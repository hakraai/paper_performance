import string
import xarray as xr
import numpy as np
import pymc as pm
from pymc.model.transform.conditioning import do
from pymc.pytensorf import convert_data
import pytensor.tensor as pt
import pytensor


def apply_aligned(function, variables, m=None):
    """
    Apply a function to a list of variables, aligning them first.
    The function returns a list of tuples containing the result and its dimensions.
    """
    m = pm.modelcontext(m)
    aligned_vars, dims = align_vars(variables, m=m)
    result = function(*aligned_vars)
    return (result, dims)


def apply_indexing_by_name(var_ids, index_parameters=None, m=None):
    """
    Apply indexing to a list of variable ids using the given index parameters.
    The function returns a dictionary mapping variable ids to their indexed values.
    """
    result = apply_indexing(var_ids, index_parameters, m=m)
    return dict(zip(var_ids, result))


def apply_indexing(var_ids=None, index_parameters=None, m=None):
    """
    Apply indexing to a list of variable ids using the given index parameters.
    The function returns a list of tuples containing the indexed variable and its dimensions.
    """
    m = pm.modelcontext(m)
    return [apply_indexing_to_var(v, index_parameters, m=m) for v in var_ids]


def apply_mixing_by_name(var_ids, mixing_parameters, m=None):
    """
    Apply mixing to a list of variable ids using the given mixing parameters.
    The function returns a dictionary mapping variable ids to their mixed values.
    """
    result = apply_mixing(var_ids, mixing_parameters, m=m)
    return dict(zip(var_ids, result))


def apply_mixing(var_ids, mixing_parameters, m=None, aggregate=True):
    """
    Apply mixing to a list of variable ids using the given mixing parameters.
    The function returns a list of tuples containing the mixed variable and its dimensions.
    """
    m = pm.modelcontext(m)
    return [
        apply_mixing_to_var(v, mixing_parameters, m=m, aggregate=aggregate) for v in var_ids
    ]


def apply_interpolation_by_name(var_ids, interpolating_parameters, m=None):
    """
    Apply interpolation to a list of variable ids using the given interpolating parameters.
    The function returns a dictionary mapping variable ids to their interpolated values.
    """
    result = apply_interpolation(var_ids, interpolating_parameters, m=m)
    return dict(zip(var_ids, result))


def apply_interpolation(var_ids, interpolating_parameters, m=None):
    """
    Apply interpolation to a list of variable ids using the given interpolating parameters.
    The function returns a list of tuples containing the interpolated variable and its dimensions.
    """
    m = pm.modelcontext(m)
    return [
        apply_interpolation_to_var(v, interpolating_parameters, m=m) for v in var_ids
    ]


def apply_indexing_to_var(var_id, index_parameters, m=None, prefix=None):
    """
    Apply indexing to a variable using the given index parameters.
    The function returns a tuple containing the indexed variable and its dimensions.
    """
    m = pm.modelcontext(m)
    if prefix is None:
        prefix = "idx_"
    var, dims = get_var_dims(var_id, m=m)

    # return if no overlap beween index_parameters and dims
    if len(set(index_parameters).intersection(dims)) == 0:
        return (var, dims)

    fancy_index = tuple()
    new_dims = tuple()
    for d in dims:
        if d in index_parameters:
            n = m.dim_lengths[d]
            i = pt.clip(m[prefix + d], 0, n - 1)
            fancy_index += (i,)
        else:
            fancy_index += (slice(None),)
            new_dims += (d,)
    new_var = var[fancy_index]

    return (new_var, new_dims)


def apply_interpolation_to_var(var_id, interpolating_parameters, m=None, prefix=None):
    """
    Apply interpolation to a variable using the given interpolating parameters.
    The function returns a tuple containing the interpolated variable and its dimensions.
    """
    m = pm.modelcontext(m)
    if prefix is None:
        prefix = "itp_"

    var, dims = get_var_dims(var_id, m=m)

    # return if no overlap beween index_parameters and dims
    if len(set(interpolating_parameters).intersection(dims)) == 0:
        return (var, dims)

    index_tuple = tuple()
    new_dims = tuple()
    w_total = pytensor.shared(np.array(1.0))
    sum_dims = []
    for i_d, d in enumerate(dims):
        if d in interpolating_parameters:
            # determine location in interpolation grid
            index_continuous = m[prefix + d]
            index_floor = pt.floor(index_continuous)
            index = pt.cast(index_floor, "int64")
            index_array = pt.as_tensor_variable([index, index + 1])

            # gather index per dimension
            index_tuple += (index_array,)

            # determine node weights
            w1 = index_continuous - index_floor
            w0 = 1 - w1
            w = pt.as_tensor_variable([w0, w1])

            # compose total weight
            w_total = pt.shape_padright(w_total) * w

            # mark summation dimensions
            sum_dims.append(i_d)
        else:
            index_tuple += (slice(None),)
            new_dims += (d,)

    # broadcast the selection vectors to construct a fancy index
    n_sum = len(sum_dims)
    pad_length = n_sum - 1
    fancy_index = tuple()
    for fi in index_tuple:
        if isinstance(fi, slice):
            fancy_index += (fi,)
        else:
            index = pt.shape_padright(fi, pad_length)
            shape = (2,) + (1,) * pad_length
            index = pt.specify_shape(index, shape)
            fancy_index += (index,)
            pad_length -= 1
        if pad_length < 0:
            break
    var_selection = var[fancy_index]

    # note that the fancy indexing can cause the dimensions to be out of order
    # this happens when the interpolation dimensions are not consecutive
    # in that case the dimensions to be interpolated are put on the front
    consecutive = sorted(sum_dims) == list(range(min(sum_dims), max(sum_dims) + 1))
    if consecutive:
        axes = [list(range(n_sum)), sum_dims]
    else:
        axes = [list(range(n_sum)), list(range(n_sum))]

    # perform interpolation by inner product
    new_var = pt.tensordot(w_total, var_selection, axes=axes)
    new_var_tpl = (new_var, new_dims)

    return new_var_tpl


def apply_mixing_to_var(var_id, mixing_parameters, m=None, prefix=None, aggregate=True):
    """
    Apply mixing to a variable using the given mixing parameters.
    The function returns a tuple containing the mixed variable and its dimensions.
    """
    m = pm.modelcontext(m)
    if prefix is None:
        prefix = "mix_"
    var, dims = get_var_dims(var_id, m=m)
    mdims = [d for d in dims if d in mixing_parameters]
    if len(mdims) == 0:
        return (var, dims)

    variables = [(var, dims)] + [(prefix + d) for d in mdims]
    variables, index_dims = align_vars(variables, m=m)
    new_var = variables[0]
    for v in variables[1:]:
        new_var = new_var * v
    if aggregate:
        axes = [index_dims.index(d) for d in mdims]
        new_var = pt.sum(new_var, axis=axes)
        new_dims = [d for d in dims if d not in mixing_parameters]
    else:
        new_dims = dims

    return (new_var, new_dims)


def apply_mixers(to_be_mixed, parameters, disagg_dims=None, prefix="mix_"):
    """
    Apply mixers to a variable using the given parameters.
    The function returns the mixed variable.
    The parameters are a dictionary of xarray datasets.
    The prefix is used to identify the mixers.
    The disagg_dims are the dimensions to keep disaggregated.
    """
    if disagg_dims is None:
        disagg_dims = []
    mixers = get_mixers(to_be_mixed, parameters, prefix)
    mdims = [d for d in mixers if d not in disagg_dims]
    if len(mdims) > 0:
        to_be_mixed = xr.dot(
            to_be_mixed,
            *mixers.values(),
            dim=mdims,
            optimize=True,
        )

    return to_be_mixed


def indexers_to_mixers(
    pars, ref_pars=None, index_prefix="idx_", mix_prefix="mix_", drop=False
):
    """
    Convert indexers to mixers.
    The indexers are used to create a mixer.
    The mixer is a weighted sum of the reference parameters.
    The output is a dictionary of mixers.
    """
    if ref_pars is None:
        ref_pars = pars
    indexed_dims = [id[len(index_prefix) :] for id in pars if index_prefix in id]
    for dim in indexed_dims:
        indexer = pars[index_prefix + dim]
        mixer = xr.zeros_like(ref_pars[dim] * indexer)
        mixer[{dim: indexer}] = 1.0
        pars[mix_prefix + dim] = mixer
        if drop:
            del pars[index_prefix + dim]

    return pars


def interpolators_to_mixers(
    pars, ref_pars=None, interp_prefix="itp_", mix_prefix="mix_", drop=False
):
    """
    Convert interpolators to mixers.
    The interpolators are used to create a mixer.
    The mixer is a weighted sum of the reference parameters.
    The output is a dictionary of mixers.
    """
    if ref_pars is None:
        ref_pars = pars
    itp_dims = [id[len(interp_prefix) :] for id in pars if interp_prefix in id]
    for dim in itp_dims:
        interpolator = pars[interp_prefix + dim]
        index = np.floor(interpolator).astype(int)
        w1 = interpolator - index
        w0 = 1 - w1
        mixer = xr.zeros_like(ref_pars[dim] * interpolator)
        mixer[{dim: index}] = w0
        mixer[{dim: index + 1}] = w1
        pars[mix_prefix + dim] = mixer
        if drop:
            del pars[interp_prefix + dim]

    return pars


def scale_interpolators(
    pars, ref_pars=None, interp_prefix="itp_", scaled_prefix="scl_"
):
    """
    Scale the interpolators to the reference parameters.
    The reference parameters are used to scale the interpolators.
    The output is a dictionary of scaled interpolators.
    """
    if ref_pars is None:
        ref_pars = pars
    itp_dims = [id[len(interp_prefix) :] for id in pars if interp_prefix in id]
    for dim in itp_dims:
        interpolator = pars[interp_prefix + dim]
        coords = ref_pars[dim]
        coords = coords.assign_coords({dim: np.arange(coords.size).astype(float)})
        pars[scaled_prefix + dim] = coords.interp({dim: interpolator}).reset_coords(
            drop=True
        )

    return pars


def get_mixers(variable, parameters, prefix="mix_"):
    """
    Get the mixers for a variable from the parameters.
    The parameters are a dictionary of xarray datasets.
    The prefix is used to identify the mixers.
    The output is an xarray dataset with the mixers.
    """
    mixers = xr.Dataset(
        {
            id[len(prefix) :]: parameters[id]
            for id in parameters
            if id.startswith(prefix) and id[len(prefix) :] in variable.dims
        }
    ).reset_coords()

    return mixers


def register_indexing_parameters(parameters, m=None, prefix="idx_"):
    """
    Register indexing parameters in the model context.
    The parameters are registered as discrete uniform distributions.
    The parameters are used to set the indexing of the variables.
    """
    m = pm.modelcontext(m)

    for name in parameters:
        if name in m.dim_lengths:
            index_name = prefix + name
            if index_name not in m:
                # register only if not already registered
                # this allows deterministic choices
                n = m.dim_lengths[name]
                pm.DiscreteUniform(index_name, lower=0, upper=n - 1)

    return


def register_interpolating_parameters(parameters, m=None, prefix="itp_"):
    """
    Register interpolating parameters in the model context.
    The parameters are registered as continuous uniform distributions.
    The parameters are used to set the interpolation grid.
    """
    m = pm.modelcontext(m)

    for name in parameters:
        if name in m.dim_lengths:
            index_name = prefix + name
            if index_name not in m:
                # use dimension lengths rather than node spacing ->
                # the prior can be steered by node spacing
                n = m.dim_lengths[name]
                # register only if not already registered
                # this allows deterministic choices
                index_continuous = pm.Uniform(index_name, lower=0, upper=n - 1)

                # register the interpolated coordinate as well
                index_0 = pt.cast(pt.floor(index_continuous), "int64")
                index_1 = pt.cast(pt.ceil(index_continuous), "int64")
                frac = index_continuous - index_0
                coords = pytensor.shared(convert_data(m.coords[name]), "coords_" + name)
                val0 = coords[index_0]
                val1 = coords[index_1]
                val = val0 + frac * (val1 - val0)
                pm.Deterministic(name, val)

    return


def register_mixing_parameters(parameters, m=None, prefix="mix_", concentration=None):
    """
    Register mixing parameters in the model context.
    The parameters are registered as Dirichlet distributions.
    The concentration parameters are used to set the Dirichlet concentration.
    """
    m = pm.modelcontext(m)
    if concentration is None:
        concentration = {}

    for name in parameters:
        if name in m.dim_lengths:
            mixture_name = prefix + name
            if mixture_name not in m:
                # register only if not already registered
                # this allows deterministic choices
                # concentration is a dictionary with the concentration parameters for each dimension
                n = m.dim_lengths[name]
                a = pt.ones(n) * concentration.get(name, 1.0)
                pm.Dirichlet(mixture_name, a=a, dims=(name,))

    return


def register_data(data, infer=True, m=None):
    """
    Register data in the model context.
    The data can be a dictionary or an xarray dataset.
    The data is registered as a pymc.Data object.
    """
    m = pm.modelcontext(m)
    if data is None:
        return
    if isinstance(data, xr.Dataset):
        data = {
            k: v
            for k, v in data.data_vars.items()
            if not np.issubdtype(v.dtype, np.str_)
        }
    for key, value in data.items():
        if isinstance(value, dict) and "dist" in value:
            value = value.copy()
            distribution = getattr(pm, value.pop("dist"))
            dims = value.pop("dims", None)
            shape = value.pop("shape", None)
            if dims is not None and shape is None:
                dims = [d for d in dims if d in m.dim_lengths]
                shape = tuple(m.dim_lengths[d] for d in dims)
            distribution(key, **value, shape=shape, dims=dims)
        elif isinstance(value, (list, str)):
            continue
        else:
            if key in m:
                # if the variable is already registered, skip it
                continue
            dims = getattr(value, "dims", None)
            pm.Data(key, value, infer_dims_and_coords=infer, dims=dims)


def align_vars(variables, dims_order=None, m=None):
    """
    Align a list of variables to a common set of dimensions.
    The variables are tuples of (variable, dimensions).
    The dimensions are the dimensions over which to align.
    The output is a tuple of (aligned variables, output dimensions).
    """
    # turn strings into (var,dims) tuple by name lookup
    var_dims_list = [get_var_dims(v, m=m) for v in variables]
    out = _align_vars(var_dims_list, dims_order)
    return out


def align_vars_with(variables, ref, m=None):
    """
    Align a list of variables to a reference variable.
    The variables are tuples of (variable, dimensions).
    The reference variable is a string or a tuple of (variable, dimensions).
    The output is a tuple of (aligned variables, output dimensions).
    """
    # turn strings into (var,dims) tuple by name lookup
    ref_var_dims = get_var_dims(ref, m=m)
    var_dims_list = [get_var_dims(v, m=m) for v in variables]
    out = _align_vars(var_dims_list, dims_order=list(ref_var_dims[1]))
    return out


def _align_vars(var_dims_list, dims_order=None):
    """
    Align a list of variables to a common set of dimensions.
    The variables are tuples of (variable, dimensions).
    The dimensions are the dimensions over which to align.
    The output is a tuple of (aligned variables, output dimensions).
    """
    # collect all dims in these vars, constructing a fixed order and dimensionality
    if dims_order is None:
        dims_order = []
    all_dims = dims_order[::-1]
    for var, dims in var_dims_list:
        for dim in reversed(dims):
            if dim not in all_dims:
                all_dims.append(dim)
    all_dims = tuple(all_dims[::-1])
    all_dims_len = len(all_dims)

    # loop over vars to be aligned
    out_vars = []
    for var, dims in var_dims_list:
        # determine the fancy index required to increase the dimensionality
        dims_len = len(dims)
        extra_dims_len = all_dims_len - dims_len
        fancy_index = dims_len * (slice(None),) + extra_dims_len * (None,)
        out_var = var[fancy_index]

        # transpose to the fixed order
        original_axes = list(range(dims_len))
        new_axes = [all_dims.index(d) for d in dims]
        out_var = np.moveaxis(out_var, original_axes, new_axes)

        # append
        out_vars.append(out_var)

    return out_vars, all_dims


def get_var_dims(var_id, m=None):
    """
    Get the variable and its dimensions from the model context.
    If var_id is a tuple, return it as is.
    If var_id is a string, look it up in the model context.
    """
    if isinstance(var_id, tuple):
        # null-op : already in the correct format
        return var_id
    m = pm.modelcontext(m)
    var = m[var_id]
    dims = m.named_vars_to_dims.get(m.name_for(var_id), tuple())
    return (var, dims)


def retrieve_parameter(par_id, m=None, **kwargs):
    """
    Retrieve a parameter from the kwargs or the model context.
    """
    m = pm.modelcontext(m)
    if par_id in kwargs:
        return kwargs[par_id]
    if par_id in m:
        return get_var_dims(par_id, m=m)

    return None


def _diff(array, axis):
    diff = pt.take(array, -1, axis=axis) - pt.take(array, 0, axis=axis)
    return diff


def einsum_multiply(multiplicands, contraction_dims=None):
    """
    Contract a list of multiplicands over the specified contraction dimensions.
    The multiplicands are tuples of (variable, dimensions).
    The contraction dimensions are the dimensions over which to contract.
    The output is a tuple of (contracted variable, output dimensions).
    """
    if contraction_dims is None:
        contraction_dims = []

    # collect all dims in these vars, constructing a fixed order and dimensionality
    terms = [item[0] for item in multiplicands]
    input_dims = [item[1] for item in multiplicands]
    all_dims = []
    for dims in input_dims:
        all_dims.extend(dims)
    unique_dims = list(dict.fromkeys(all_dims))
    output_dims = [d for d in unique_dims if d not in contraction_dims]

    # generate the einsum string
    einsum_string_lookup = dict(zip(unique_dims, list(string.ascii_lowercase)))
    output_string = "".join([einsum_string_lookup[d] for d in output_dims])
    input_strings = []
    for dims in input_dims:
        input_strings.append("".join([einsum_string_lookup[d] for d in dims]))
    input_string = ",".join(input_strings)
    einsum_string = input_string + "->" + output_string

    # perform the summation/contraction
    contracted = pt.einsum(einsum_string, *terms)
    contracted_tpl = (contracted, output_dims)

    return contracted_tpl


def diff_time(arg_tpl, time_index):
    """
    Compute the difference of a variable along the time dimension.
    The input is a tuple of (variable, dimensions).
    The output is a tuple of (variable, dimensions).
    """
    arg, input_dims = arg_tpl
    axis = input_dims.index(time_index)
    output = _diff(arg, axis)
    output_dims = tuple(d for d in input_dims if d != time_index)

    return (output, output_dims)


def lognormal_sigma(mean, std):
    """
    Compute the sigma parameter of a lognormal distribution based
    on the desired mean and standard deviation.
    """
    return pt.sqrt(pt.log(1 + (std**2 / mean**2)))


def lognormal_mu(mean, std):
    """
    Compute the mu parameter of a lognormal distribution based
    on the desired mean and standard deviation.
    """
    return pt.log(mean**2 / pt.sqrt(std**2 + mean**2))


def extract_means(idata, variables=None):
    """
    Extract the means of the posterior samples from the idata object.
    If no variables are provided, compute the mean for all variables.
    """
    if variables is None:
        means = idata["posterior"].mean(["chain", "draw"])
    else:
        if isinstance(variables, str):
            variables = [variables]
        means = (idata["posterior"][variables]).mean(["chain", "draw"])
    means_dict = means.to_dict()["data_vars"]
    return {k: v["data"] for k, v in means_dict.items()}


def do_mean(model, idata, variables=None):
    """
    Compute the mean of the posterior samples for the given variables.
    If no variables are provided, compute the mean for all variables.
    """

    means = extract_means(idata, variables)

    return do(model, means)
