"""PyMC model builders for ETF, ETAS, and related hybrid formulations."""

import pymc as pm
import pymc.model.transform.optimization as pmto
import pytensor.tensor as pt

from . import model_core as model
from .pymc_support import (
    register_data,
    register_mixing_parameters,
    register_interpolating_parameters,
    register_indexing_parameters,
    apply_aligned,
    apply_interpolation,
    apply_mixing,
    apply_indexing,
    apply_mixing_to_var,
    get_var_dims,
    retrieve_parameter,
    align_vars,
    align_vars_with,
    diff_time,
    einsum_multiply,
)


def dsm_model(dsm_parameter_data, m=None):
    """Build DSM-derived covariate, stress, and azimuth terms in a PyMC model."""
    m = pm.modelcontext(m)

    # preconditioning: initialized indexing and/or interpolation parameters
    # these parameters allow cruising through the lookup table of DSM parameters
    # that are too hard to evaluate on the fly
    indexing_dims = dsm_parameter_data.get("indexing_dims", [])
    register_indexing_parameters(indexing_dims, m=m)
    interpolation_dims = dsm_parameter_data.get("interpolation_dims", [])
    register_interpolating_parameters(interpolation_dims, m=m)

    # interpolate datasets for various parameters that are (or might be) explored in
    # a lookup table fashion - we cruise through the lookup table either by discrete
    # (random) indexing or by multilinear interpolation; the latter is more efficient
    # since it allows for NUTS sampling, JAX etc.
    dsm_datasets = [
        "covariate_exposed",  # probably not required to be interpolated
        "covariate_observed",
        "measure_exposed",
        "measure_observed",
    ]
    dsm_data = apply_indexing(dsm_datasets, indexing_dims, m=m)
    dsm_data = apply_interpolation(dsm_data, interpolation_dims, m=m)
    dsm_dict = dict(zip(dsm_datasets, dsm_data))

    # determine incremental stress
    # by default, this is the same as the main covariate
    dsm_dict["stress_exposed"] = dsm_dict["covariate_exposed"]
    dsm_dict["stress_observed"] = dsm_dict["covariate_observed"]

    # however, if "hs_exp" is present, we need to calculate the stress
    # from the covariate (which is then ~pressure) and the compressibility
    if "hs_exp" in m:
        compr_id = dsm_parameter_data.get("compressibility_id", "compressibility")
        for ext in ["_exposed", "_observed"]:
            typid = compr_id + ext
            if typid in m:
                dsm_dict["stress" + ext] = calculate_stress(
                    dsm_dict["covariate" + ext], typid
                )

    # determine the azimuthal rate dependency due to preferred orientation / elliptic stress
    # anisotropy
    if "stress_ratio" in m:
        # if not supplied before introduce a non-informative prior for the azimuth
        # of the principal horizontal stress vector
        if "stress_azimuth" not in m:
            pm.Uniform(
                "stress_azimuth",
                lower=0.0,
                upper=180.0,  # period pi
                transform=pm.distributions.transforms.circular,
            )

        # determine the azimuthal stress multiplier both for the total exposed domain and
        # the individual observed events
        azimuth_id = dsm_parameter_data.get("azimuth_id", "azimuth")
        for ext in ["_exposed", "_observed"]:
            typid = azimuth_id + ext
            if typid in m:
                dsm_dict["azimuth_multiplier" + ext] = calculate_azimuth_multiplier(
                    typid, m
                )

    return dsm_dict


def calculate_azimuth_multiplier(azimuth_id, m=None):
    """
    Calculate the azimuthal stress multiplier
    """
    m = pm.modelcontext(m)
    azimuth_exposed, dim_exp = get_var_dims(azimuth_id)

    cos_exposed = pt.cos(2 * pt.deg2rad(azimuth_exposed - m["stress_azimuth"]))
    a = m["stress_ratio"]
    multiplier_exposed = cos_exposed * (1 - a) / (1 + a)

    return (multiplier_exposed, dim_exp)


def calculate_stress(covariate_tpl, compr_id, m=None):
    """
    Calculate the stress from the covariate and the compressibility
    """
    m = pm.modelcontext(m)
    compr_tpl = get_var_dims(compr_id, m=m)

    hs = pt.exp(m["hs_exp"] * pt.log(10))
    hr = 1 / compr_tpl[0]
    hr_ref = 1 / pt.max(compr_tpl[0])

    modulator = (hr_ref + hs) / (hr + hs)
    modulator_tpl = (modulator, compr_tpl[1])

    stress_tpl = einsum_multiply([covariate_tpl, modulator_tpl])

    return stress_tpl


def size_model(size_parameter_data, m=None, **kwargs):
    """Build optional magnitude-size relation terms for exposed and observed data."""
    if size_parameter_data is None:
        return {}

    m = pm.modelcontext(m)
    functional_form = size_parameter_data["functional_form"]

    if functional_form == "constant":
        b_observed_tpl = (m["b_val"], tuple())
        b_exposed_tpl = (m["b_val"], tuple())
    else:
        size_covariate_id = size_parameter_data["covariate_id"]
        size_coveriate_exposed_id = size_covariate_id + "_exposed"
        size_covariate_observed_id = size_covariate_id + "_observed"

        sc_exposed, sc_dims_exposed = retrieve_parameter(
            size_coveriate_exposed_id, m=m, **kwargs
        )
        sc_exposed_min = pt.min(sc_exposed)
        sc_exposed_max = pt.max(sc_exposed)

        sc_observed, sc_dims_observed = retrieve_parameter(
            size_covariate_observed_id, m=m, **kwargs
        )
        sc_observed_min = pt.min(sc_observed)
        sc_observed_max = pt.max(sc_observed)

        sc_min = pt.min([sc_exposed_min, sc_observed_min])
        sc_max = pt.max([sc_exposed_max, sc_observed_max])
        sc_range = sc_max - sc_min

        b0 = m["b_val"][0]
        b1 = m["b_val"][1]
        if functional_form == "linear":
            b_observed = b0 + (b1 - b0) * (sc_observed - sc_min) / sc_range
            b_exposed = b0 + (b1 - b0) * (sc_exposed - sc_min) / sc_range
        else:
            if functional_form == "tanh0":
                if "size_covariate_scale" not in m:
                    covariate_scale_mean = m["size_covariate_scale_fraction"] * sc_max
                    # tanh with loc at covariate 0
                    mu_scale = pt.log(covariate_scale_mean)
                    pm.LogNormal("size_covariate_scale", mu=mu_scale)
                c_loc = 0.0  # no other reasonable value
            elif functional_form == "step":
                if "size_covariate_scale" in m:
                    c_scale = m["size_covariate_scale"]
                else:
                    c_scale = m["size_covariate_scale_fraction"] * sc_range
                if "c_loc" not in m:
                    c_min_buf = sc_min - 2 * c_scale
                    c_max_buf = sc_max + 2 * c_scale
                    pm.Uniform("c_loc", lower=c_min_buf, upper=c_max_buf)
                c_loc = m["c_loc"]
            elif functional_form == "tanh":
                if "size_covariate_scale" not in m:
                    covariate_scale_mean = m["size_covariate_scale_fraction"] * sc_range
                    c_scale = pm.Exponential(
                        "size_covariate_scale", scale=covariate_scale_mean
                    )
                else:
                    c_scale = m["size_covariate_scale"]
                if "c_loc" not in m:
                    c_min_buf = sc_min - 2 * c_scale
                    c_max_buf = sc_max + 2 * c_scale
                    pm.Uniform("c_loc", lower=c_min_buf, upper=c_max_buf)
                c_loc = m["c_loc"]
            else:
                raise ValueError(
                    f"Unknown functional form {size_parameter_data['functional_form']}"
                )
            b_exposed = model.hyperbolic_tangent(sc_exposed, b0, b1, c_loc, c_scale)
            b_observed = model.hyperbolic_tangent(sc_observed, b0, b1, c_loc, c_scale)
        b_exposed_tpl = (b_exposed, sc_dims_exposed)
        b_observed_tpl = (b_observed, sc_dims_observed)

    return {"b_exposed": b_exposed_tpl, "b_observed": b_observed_tpl}


def rate_model(rate_parameter_data, m=None, **kwargs):
    """Build background count, rate, and optional rate-density terms."""
    m = pm.modelcontext(m)

    # collect data from the dynamic subsurface model
    measure_exposed_tpl = retrieve_parameter("measure_exposed", m=m, **kwargs)
    measure_observed_tpl = retrieve_parameter("measure_observed", m=m, **kwargs)
    stress_exposed_tpl = retrieve_parameter("stress_exposed", m=m, **kwargs)
    stress_observed_tpl = retrieve_parameter("stress_observed", m=m, **kwargs)
    azimuth_multiplier_exposed_tpl = retrieve_parameter(
        "azimuth_multiplier_exposed", m=m, **kwargs
    )
    azimuth_multiplier_observed_tpl = retrieve_parameter(
        "azimuth_multiplier_observed", m=m, **kwargs
    )

    # collect data from the size model - these are optional and may be None
    b_exposed_tpl = retrieve_parameter("b_exposed", m=m, **kwargs)
    b_observed_tpl = retrieve_parameter("b_observed", m=m, **kwargs)

    # Determine if there are any mixing dimensions, such as bernstein_index
    # these will be used later on
    mixing_dims = rate_parameter_data.get("mixing_dims", [])

    # Determine all dimensions to be contracted from the measure
    event_id = kwargs.get("event_id", "event_id")
    epoch_exposed_id = kwargs.get("epoch_exposed_id", "epoch")
    epoch_observed_id = kwargs.get("epoch_observed_id", "datetime_bound")
    ignore_dims = [event_id, epoch_exposed_id, epoch_observed_id] + mixing_dims

    # identify radial data
    radial_id = rate_parameter_data.get("radial_distance_id", "radial_distance")
    radial_id_exposed = radial_id + "_exposed"
    radial_id_observed = radial_id + "_observed"

    # STEP 1: Determine total event count

    # collect multiplicands
    multiplicands_exposed = collect_multiplicands(
        measure_exposed_tpl,
        stress_exposed_tpl,
        azimuth_multiplier_exposed_tpl,
        b_exposed_tpl,
        radial_id_exposed,
    )

    # multiply/contract and diff the epochs
    count_tpl = contract_and_diff(multiplicands_exposed, ignore_dims, epoch_exposed_id)

    # STEP 2: Determine the m_min-exceedance rate (actually, small-interval count)
    # at the times and locations of the observed events

    # collect multiplicands
    multiplicands_observed = collect_multiplicands(
        measure_observed_tpl,
        stress_observed_tpl,
        azimuth_multiplier_observed_tpl,
        b_observed_tpl,
        radial_id_observed,
    )

    # multiply/contract and diff the small-interval epochs
    rates_tpl = contract_and_diff(
        multiplicands_observed, ignore_dims, epoch_observed_id
    )

    # prepare the return data
    return_vars = [count_tpl, rates_tpl]
    return_var_names = ["count_bg", "rates_bg"]

    # STEP 3: Determine the magnitude rate density
    # at the times, locations and magnitudes of the observed events
    if b_observed_tpl is not None:
        rate_density_multiplier_tpl = apply_aligned(
            rate_multiplier_pdf,
            [b_observed_tpl, "magnitude", "m_min"],
        )
        multiplicands_observed.append(rate_density_multiplier_tpl)

        # multiply and contract
        rate_densities_tpl = contract_and_diff(
            multiplicands_observed, ignore_dims, epoch_observed_id
        )
        return_vars.append(rate_densities_tpl)
        return_var_names.append("rate_densities")

    # STEP 4: Apply mixing (if any - e.g., Bernstein) to all two or three output
    # variables
    if mixing_dims:
        concentration = rate_parameter_data.get("mixing_concentration", {})
        register_mixing_parameters(mixing_dims, concentration=concentration, m=m)

        # for convenience register disaggregation
        count_unmixed_tpl = apply_mixing_to_var(
            count_tpl, mixing_dims, m=m, aggregate=False
        )
        pm.Deterministic(
            "event_count_bg_disaggregated",
            count_unmixed_tpl[0],
            dims=count_unmixed_tpl[1],
        )

        # apply mixing
        return_vars = apply_mixing(return_vars, mixing_dims, m=m)

    # STEP 5: Calculate the effective (average/scaled) magnitude distribution
    # so that it can be applied also to etas events and to separate it in
    # the observables
    # The average is over, e.g., Bernstein components and spatial variations
    # that are incorporated using the radial weighting
    if b_observed_tpl is not None:
        scaled_rate_densities = return_vars[-1][0] / return_vars[-2][0]
        return_vars[-1] = (scaled_rate_densities, return_vars[-1][1])

    return dict(zip(return_var_names, return_vars))


def contract_and_diff(multiplicands, ignore_dims, epoch_id):
    """Multiply aligned terms, contract nuisance dimensions, and difference epochs."""
    measure_dims = multiplicands[0][1]
    contract_dims = [d for d in measure_dims if d not in ignore_dims]
    count_tpl = einsum_multiply(multiplicands, contract_dims)
    count_tpl = diff_time(count_tpl, epoch_id)
    return count_tpl


def collect_multiplicands(
    measure_tpl,
    stress_tpl,
    azimuth_multiplier_tpl,
    b_tpl,
    radial_id,
    m=None,
):
    """Collect the multiplicative factors that define the background rate model."""
    m = pm.modelcontext(m)

    # collect multiplicands
    multiplicands = [measure_tpl]

    # rate model
    count_density_exposed_tpl = apply_aligned(
        model.extreme_threshold_failure,
        [stress_tpl, "theta0", "theta1"],
    )
    multiplicands.append(count_density_exposed_tpl)

    # append azimuth multiplier
    if azimuth_multiplier_tpl is not None:
        rate_multiplier_tpl = apply_aligned(
            lambda x, y: pt.exp(x * y),
            [azimuth_multiplier_tpl, "theta0"],
        )
        multiplicands.append(rate_multiplier_tpl)

    # append radial weights
    if "radial_sigma" in m and radial_id in m:
        radial_multiplier_tpl = apply_aligned(
            model.radial_normal_weight,
            [radial_id, "radial_sigma"],
        )
        multiplicands.append(radial_multiplier_tpl)

    # append rate multiplier

    # If the predictor for size is different from
    # the predictor for the rate, then the spatiotemporal distribution looks
    # different for each size. It would be a coincidence of galactic proportions
    # if the rate prediction would be precisely tuned at the completeness
    # magnitude m_min. It is probably useful to have the reference magnitude
    # of the exceedance rate model as a free parameter: m_rate
    if "m_rate" in m and b_tpl is not None:
        rate_multiplier_tpl = apply_aligned(
            rate_multiplier_sf,
            [b_tpl, "m_min", "m_rate"],
        )
        multiplicands.append(rate_multiplier_tpl)

    return multiplicands


def rate_multiplier_sf(b, m_target, m_ref):
    """
    Formulate to relate event rates between different magnitudes
    assuming a Gutenberg-Richter distribution with b-value b.
    """
    # TODO: how to make use of the PyMC distribution facilities?
    # such that we can easily plug in a different size distribution
    # such as a tapered GR distribution

    blog10 = b * pt.log(10)
    size_multiplier = pt.exp(-blog10 * (m_target - m_ref))

    return size_multiplier


def rate_multiplier_pdf(b, m_target, m_ref):
    """
    Formulate to relate event rates between different magnitudes
    assuming a Gutenberg-Richter distribution with b-value b.
    """
    # TODO: how to make use of the PyMC distribution facilities?
    # such that we can easily plug in a different size distribution
    # such as a tapered GR distribution

    blog10 = b * pt.log(10)
    size_multiplier = pt.exp(-blog10 * (m_target - m_ref)) * blog10

    return size_multiplier


def etas_model(etas_parameter_data, m=None):
    """Build ETAS offspring count and rate terms from parent-event data."""
    if etas_parameter_data is None:
        return {}

    m = pm.modelcontext(m)

    indexing_dims = etas_parameter_data.get("indexing_dims", [])
    register_indexing_parameters(indexing_dims, m=m)
    interpolation_dims = etas_parameter_data.get("interpolation_dims", [])
    register_interpolating_parameters(interpolation_dims, m=m)

    # interpolate datasets
    etas_dataset_ids = [
        "etas_support_coverage",
        "etas_spatial_rates",
    ]
    etas_datasets = apply_indexing(etas_dataset_ids, indexing_dims, m=m)
    etas_datasets = apply_interpolation(etas_datasets, interpolation_dims, m=m)
    etas_spatial_coverage_tpl, etas_spatial_rates_tpl = etas_datasets

    # STEP 1: Determine productivity of all parent events
    # 1A) align
    vars_to_align = [
        "parent_magnitude",
        "etas_K",
        "etas_a",
        "m_min",
    ]
    aligned_vars, dims_prod = align_vars(vars_to_align, m=m)
    (
        parent_magnitudes,
        etas_K,
        etas_a,
        m_ref,
    ) = aligned_vars

    # 1B) process
    # determine productivity of all parent events
    etas_prod = model.etas_productivity(parent_magnitudes, etas_K, etas_a, m_ref)
    etas_prod_tpl = (etas_prod, dims_prod)

    # STEP 2: Determine the cumulative count of all parent events at relevant epochs
    # 2A) align
    vars_to_align = [
        etas_spatial_coverage_tpl,
        etas_prod_tpl,
        "epoch_delays",
        "etas_c",
        "etas_p",
    ]
    aligned_vars, dims_count = align_vars(vars_to_align, m=m)
    etas_spatial_coverage, etas_prod, epoch_delays, etas_c, etas_p = aligned_vars

    # determine axis for parent events
    # we will later sum over this axis to determine the cumulative count
    # then we have to remove this axis from the dimensions
    # then we determine the axis for the epoch dimension for which
    # we will do a diff
    axis_parent = dims_count.index("parent_id")
    dims_count = tuple(d for d in dims_count if d != "parent_id")

    # 2B) process
    # determine cumulative distribution of all parent events at relevant epochs
    temp_cumul = gated_etas_temporal_cumulative(epoch_delays, etas_c, etas_p)
    cumulative_count = pt.sum(
        temp_cumul * etas_prod * etas_spatial_coverage, axis=axis_parent
    )
    count_tpl = (cumulative_count, dims_count)
    count_tpl = diff_time(count_tpl, "epoch")

    # STEP 3: Determine the rate at the relative times and locations of the observed events
    # 3A) align
    aligned_vars, dims_etas = align_vars(
        [
            etas_spatial_rates_tpl,
            etas_prod_tpl,
            "event_delays",
            "etas_c",
            "etas_p",
        ],
        m=m,
    )
    (
        etas_spatial_rates,
        etas_prod,
        event_delays,
        etas_c,
        etas_p,
    ) = aligned_vars

    # 3B) process
    # spatio-temporal rate density
    etas_temp = gated_etas_temporal(event_delays, etas_c, etas_p)
    etas_spattemp = etas_temp * etas_spatial_rates

    # STEP 4: combine parent productivity with the spatio-temporal rate
    # 4A) align
    # determine axis for parent events
    # we will later sum over this axis to determine the rate
    # then we have to remove this axis from the dimensions
    axis_parent = dims_etas.index("parent_id")
    dims_rates = tuple(d for d in dims_etas if d != "parent_id")

    # 4B) process
    # calculate the rate at the relative times and locations of the observed events
    rates = pt.sum(etas_spattemp * etas_prod, axis=axis_parent)
    rates_tpl = (rates, dims_rates)

    return {"count_etas": count_tpl, "rates_etas": rates_tpl}


def total_rate_model(m=None, **kwargs):
    """
    Combine the background and etas models to determine the total rate
    and the total event count.
    """
    # unpack kwargs
    count_bg_tpl = kwargs.get("count_bg", None)
    rates_bg_tpl = kwargs.get("rates_bg", None)
    rate_densities_tpl = kwargs.get("rate_densities", None)  # optional
    count_etas_tpl = kwargs.get("count_etas", None)  # optional
    rates_etas_tpl = kwargs.get("rates_etas", None)  # optional

    if rates_etas_tpl is None:
        count_tpl = count_bg_tpl
        rates_tpl = rates_bg_tpl
    else:
        m = pm.modelcontext(m)
        (count_bg, count_etas), dims_count = align_vars([count_bg_tpl, count_etas_tpl])
        (rates_bg, rates_etas), dims_rates = align_vars([rates_bg_tpl, rates_etas_tpl])
        count_tpl = (count_bg + count_etas, dims_count)
        rates_tpl = (rates_bg + rates_etas, dims_rates)

        pm.Deterministic("event_count_bg", count_bg)
        pm.Deterministic("event_count_etas", count_etas)
        pm.Deterministic("branching_ratio", count_etas / count_bg)
        pm.Deterministic("etas_fraction", count_etas / count_tpl[0])

    return_values = {
        "count": count_tpl,
        "rates": rates_tpl,
    }

    if rate_densities_tpl is not None:
        rates_tpl = einsum_multiply([rates_tpl, rate_densities_tpl])

        # or, leave split between XT and M?
        # return_values["rate_densities"] = rate_densities_tpl

    return return_values


def register_observables(m=None, **kwargs):
    """Register the likelihood terms for counts, spacetime, and magnitudes."""
    m = pm.modelcontext(m)

    # unpack kwargs
    count_tpl = kwargs.get("count", None)
    rates_tpl = kwargs.get("rates", None)
    rate_densities_tpl = kwargs.get("rate_densities", None)  # optional

    # Step 1: Poisson distribution for total count
    # Observed total count - integer
    event_count_obs_tpl = get_var_dims("event_count", m=m)
    event_count_obs = event_count_obs_tpl[0]
    event_count_obs_dims = event_count_obs_tpl[1]

    # Modelled total count -- possibly distributed over multiple epochs
    variables, dims = align_vars_with([count_tpl], event_count_obs_tpl, m=m)
    event_count_mod = variables[0]
    event_count_mod_dims = dims
    event_count_mod_tpl = (event_count_mod, event_count_mod_dims)

    # likelihood from Poisson distribution
    pm.Poisson(
        "event_count_total",
        event_count_mod,
        observed=event_count_obs,
        dims=event_count_obs_dims,
    )

    # Step 2: Custom distribution for space-time distribution
    # This spacetime vector is just a dummy to pretend a space-time distribution;
    # all we need to know is already in the rates parameter
    # in principle, we could do the lookup of rates here, but that would be
    # inefficient
    (rates, event_count_mod), dims = align_vars([rates_tpl, event_count_mod_tpl], m=m)

    # custom distribution for the spacetime dimensions
    def rate_logp(_dummy, rates, count):
        return pt.log(rates / count)

    xt_tpl = get_var_dims("spacetime", m=m)
    xt_dims = xt_tpl[1]
    xt_observed = xt_tpl[0]
    pm.CustomDist(
        "XT",
        rates,
        event_count_mod,
        logp=rate_logp,
        observed=xt_observed,
        signature="(),()->(3)",
        dims=xt_dims,
    )

    if rate_densities_tpl is not None:
        m_tpl = get_var_dims("magnitude", m=m)
        m_dims = m_tpl[1]
        m_observed = m_tpl[0]
        pm.CustomDist(
            "M",
            rate_densities_tpl[0],
            1,
            logp=rate_logp,
            observed=m_observed,
            signature="(),()->()",
            dims=m_dims,
        )

    return


def generate_ts_etf_etas_model(
    data,
    m=None,
    **kwargs,
):
    """Assemble the full ETF plus ETAS PyMC model from data and settings."""
    # handle model context
    if m is None:
        try:
            m = pm.Model.get_context()
        except TypeError:
            m = pm.Model()

    dsm_parameter_data = kwargs.get("dsm_parameter_data", None)
    rate_parameter_data = kwargs.get("rate_parameter_data", None)
    etas_parameter_data = kwargs.get("etas_parameter_data", None)
    size_parameter_data = kwargs.get("size_parameter_data", None)

    with m:
        # REGISTER DATA
        register_data(data)
        register_data(dsm_parameter_data)
        register_data(size_parameter_data)
        register_data(rate_parameter_data)
        register_data(etas_parameter_data)

        # PART 1: DSM - stress model
        dsm_results = dsm_model(dsm_parameter_data, m=m)

        # PART 2: MAGNITUDES / SIZE
        size_results = size_model(size_parameter_data, **dsm_results, m=m)

        # PART 3: SRM - background rate model
        background_rate_results = rate_model(
            rate_parameter_data, **size_results, **dsm_results, m=m
        )

        # PART 4: ETAS - triggered rate model
        etas_results = etas_model(etas_parameter_data, m=m)

        # PART 5: COMBINE background and ETAS
        total_rate_results = total_rate_model(
            **background_rate_results, **etas_results, m=m
        )

        # Register observables
        register_observables(**total_rate_results, **size_results, m=m)

    return m


def generate_and_test_model(
    model_f,
    data,
    settings,
    freeze=False,
):
    """Build a model, optionally freeze dims and data, and run debug checks."""
    m = model_f(data, **settings)

    if freeze:
        m = pmto.freeze_dims_and_data(m)

    m.debug(verbose=True)

    return m


def gated_etas_temporal(t, c, p):
    """Evaluate the ETAS temporal kernel only for positive delays."""
    t_local = pt.clip(t, 0.0, pt.inf)
    return pt.switch(pt.gt(t, 0.0), model.etas_temporal(t_local, c, p), 0.0)


def gated_etas_temporal_cumulative(t, c, p):
    """Evaluate the cumulative ETAS temporal kernel only for positive delays."""
    t_local = pt.clip(t, 0.0, pt.inf)
    return pt.switch(pt.gt(t, 0.0), model.etas_temporal_cumulative(t_local, c, p), 0.0)
