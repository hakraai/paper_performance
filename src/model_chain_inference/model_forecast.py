"""Forecast-generation helpers for ETF and ETAS model outputs."""

import numpy as np
import xarray as xr

import chaintools.tools_grid as tgrid
from . import model_core as model
from .catalogue import get_realisation


def generate_temporal_forecast_etf(
    covariate,
    measure,
    parameters,
    target_step=0.025,
    support=1.0,
    ensemble_dims=None,
    disagg_dims=None,
    spatial_dims="loc",
    temporal_dim="datetime",
    variance=False,
    fractiles=None,
    scale=False,
):
    """
    Generate a temporal forecast for the extreme threshold failure model.
    Parameters
    ----------
    covariate : xr.DataArray
        Covariate data, either as samples or as aggregated grid
    measure : xr.DataArray
        Measure data, either as samples or as aggregated grid
    parameters : xr.Dataset
        Parameters for the model
    target_step : float, optional
        Target step size for the temporal grid, by default 0.025
    support : float, optional
        Support factor for the measure, by default 1.0
    ensemble_dims : list, optional
        Dimensions to ensemble over, by default None
    disagg_dims : list, optional
        Dimensions to disaggregate over, by default None
    spatial_dims : str or list, optional
        Spatial dimensions to marginalize over, by default "loc"
    temporal_dim : str, optional
        Temporal dimension to use, by default "datetime"
    variance : bool, optional
        Whether to compute variances, by default False
    fractiles : list, optional
        Percentiles to compute, by default None
    scale : bool, optional
        Whether to scale the covariate, by default False
    Returns
    -------
    xr.Dataset
        Forecast data
    """
    measure = measure * support

    if scale:
        covariate_scale = parameters["covariate_scale"].load()
    else:
        covariate_scale = 1.0
    scaled_covariate = covariate.load() / covariate_scale
    aggregate_by_time = tgrid.aggregate_to_grid(
        samples=scaled_covariate.rename("covariate"),
        target_step=target_step,
        weights=measure.load().fillna(0.0),
        marginalize_dims=spatial_dims,
    )

    return generate_forecast_etf(
        aggregate_by_time["covariate"],
        aggregate_by_time,
        parameters,
        support=1.0,
        marginalize_dims=["covariate"],
        ensemble_dims=ensemble_dims,
        diff_dim=temporal_dim,
        disagg_dims=disagg_dims,
        scale=False,  # done before
        variance=variance,
        fractiles=fractiles,
    )


def generate_forecast_etf(
    covariate,
    measure,
    parameters,
    support=1.0,
    marginalize_dims=None,
    ensemble_dims=None,
    diff_dim=None,
    diff_label="lower",
    disagg_dims=None,
    scale=True,
    variance=False,
    fractiles=None,
):
    """
    Generate forecast for extreme threshold failure model
    Parameters
    ----------
    covariate : xr.DataArray
        Covariate data, either as samples or as aggregated grid
    measure : xr.DataArray
        Measure data, either as samples or as aggregated grid
    parameters : xr.Dataset
        Parameters for the model
    support : xr.DataArray, optional
        Support factor for the measure, by default 1.0
    marginalize_dims : list, optional
        Dimensions to marginalize over, by default None
    ensemble_dims : list, optional
        Dimensions to ensemble over, by default None
    diff_dim : str, optional
        Dimension to difference over before determining the statistics, by default None
    diff_label : str, optional
        Label to diff, by default "lower"
    disagg_dims : list, optional
        Dimensions to disaggregate over, by default None
    scale : bool, optional
        Whether to scale the covariate, by default True
    variance : bool, optional
        Whether to compute variances, by default False
    fractiles : list, optional
        Percentiles to compute, by default None
    Returns
    -------
    xr.Dataset
        Forecast data
    """

    # marginalize when using aggregate measures
    if marginalize_dims is None:
        marginalize_dims = []
    else:
        marginalize_dims = list(np.atleast_1d(marginalize_dims))
    if ensemble_dims is None:
        ensemble_dims = set(["draw", "chain", "sample"]) & set(parameters.dims)
    else:
        ensemble_dims = list(np.atleast_1d(ensemble_dims))
    # apply the support filter to the measure, this will often be a factor 1,
    # but can be used to allow polygon filtering
    measure = support * measure

    # determine covariate-cumulative spatial density
    print("compute forecast per ensemble member")
    forecast = forecast_etf_etas(
        covariate,
        measure,
        parameters,
        marginalize_dims,
        diff_dim,
        diff_label,
        disagg_dims=disagg_dims,
        scale=scale,
    )

    # TODO : calculate ETAS spatially resolved

    print("compute forecast ensemble mean")
    forecast_mean = forecast.mean(ensemble_dims)
    out = xr.Dataset()
    out["mean"] = forecast_mean

    if variance:
        print("compute forecast ensemble variance")
        squares = (forecast - forecast_mean) ** 2
        forecast_var = squares.mean(ensemble_dims)
        out["var"] = forecast_var

    if fractiles is not None:
        print("compute forecast ensemble fractiles")
        forecast_fractiles = forecast.quantile(fractiles, dim=ensemble_dims)
        out["fractiles"] = forecast_fractiles

    return out.compute()


def forecast_etf_etas(
    covariate,
    measure,
    parameters,
    marginalize_dims,
    diff_dim,
    diff_label,
    disagg_dims=None,
    scale=True,
):
    """Evaluate the ETF forecast and optionally augment it with ETAS scaling."""

    itp_dims = [v_id for v_id in parameters.data_vars if v_id in covariate.dims]
    if itp_dims:
        covariate = covariate.interp(
            {v_id: parameters[v_id] for v_id in itp_dims},
            method="linear",
        ).drop_vars(itp_dims)

    itp_dims = [v_id for v_id in parameters.data_vars if v_id in measure.dims]
    if itp_dims:
        measure = measure.interp(
            {v_id: parameters[v_id] for v_id in itp_dims},
            method="linear",
        ).drop_vars(itp_dims)

    covariate_scale = 1.0
    if scale:
        if "covariate_scale" in parameters:
            covariate_scale = parameters["covariate_scale"]
        else:
            print("No scaling factor found, using 1.0")

    scaled_covariate = covariate / covariate_scale

    cumulative_density = model.extreme_threshold_failure(
        scaled_covariate,
        parameters["theta0"],
        parameters["theta1"],
    )

    # apply measure / weighting
    cumulative_count = xr.dot(cumulative_density, measure, dim=marginalize_dims)

    # apply dirichlet mixers
    cumulative_count = apply_mixers(cumulative_count, parameters, disagg_dims)

    # differentiate -> to go from cumulative count to interval count
    if diff_dim is not None:
        forecast_bg = cumulative_count.diff(diff_dim, label=diff_label)
    else:
        forecast_bg = cumulative_count

    # include ETAS through multiplier
    forecast = apply_etas(forecast_bg, parameters, disagg_dims)

    return forecast


def apply_etas(forecast_bg, parameters, disagg_dims=None):
    """Add ETAS offspring counts to a background forecast when configured."""
    if disagg_dims is None:
        disagg_dims = []

    # TODO: full time-dependent
    # lazy approach
    if "branching_ratio" in parameters:
        branching_ratio = parameters["branching_ratio"]
        forecast_etas = branching_ratio * forecast_bg
        if "etas_generation" in disagg_dims:
            # if etas_generation is a disaggregation dimension, we need to
            # expand the forecast_etas to include it
            forecast_etas = forecast_etas.expand_dims("etas_generation")
            forecast_etas = forecast_etas.assign_coords(
                etas_generation=parameters["etas_generation"]
            )
            forecast = xr.Dataset(
                {
                    "background": forecast_bg,
                    "offspring": forecast_etas,
                }
            ).to_array(dim="etas_generation")
        else:
            forecast = forecast_bg + forecast_etas
    else:
        forecast = forecast_bg
    return forecast


def apply_mixers(to_be_mixed, parameters, disagg_dims=None, prefix="mix_"):
    """Apply configured mixer weights to a forecast variable."""
    if disagg_dims is None:
        disagg_dims = []
    mixers = get_mixers(to_be_mixed, parameters, prefix)
    mdims = [d for d in mixers if d not in disagg_dims]
    if len(mixers) > 0:  # also multiply if only disagg_dims
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
    """Convert discrete index selections into one-hot mixer arrays."""
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
    """Convert interpolation coordinates into linear mixer arrays."""
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
    """Map interpolation indices back to the coordinate values of reference parameters."""
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
    """Collect mixer variables that apply to the dimensions of a variable."""
    mixers = xr.Dataset(
        {
            id[len(prefix) :]: parameters[id]
            for id in parameters
            if id.startswith(prefix) and id[len(prefix) :] in variable.dims
        }
    ).reset_coords()

    return mixers


def apply_interpolators(var, post, interp_prefix="itp_", scaled_prefix="scl_"):
    """Interpolate a variable using scaled coordinate values stored in a posterior dataset."""
    interp_dict = {
        v: post[f"{scaled_prefix}{v}"]
        for v in var.dims
        if f"{interp_prefix}{v}" in post
    }
    if len(interp_dict) == 0:
        return var
    return var.interp(interp_dict).reset_coords(list(interp_dict.keys()), drop=True)


def generate_testsuite_etf(
    event_data,
    parameters,
    covariate,
    measure,
    support,
    filterset,
    target_step=0.01,
    variance=False,
    fractiles=None,
    spatiotemporal=False,
    disagg_dims=None,
    dsm_mode=None,
):
    """Build spatial and temporal forecast-observation datasets for ETF testing."""
    epochs = filterset["timeframe"].values
    epochs = np.clip(
        epochs.astype("datetime64[ns]"),
        covariate["datetime"].data[0],
        covariate["datetime"].data[-1],
    )
    spatial_support = support.sel(polygon=filterset["polygon"])

    # SPATIAL FORECAST: smashed time dimension
    # interpolate the boundaries of the test suite timeframe
    print("spatial analysis")
    covariate_at_epochs = covariate.interp({"datetime": epochs})
    forecast_spatial = generate_forecast_etf(
        covariate_at_epochs,
        measure,
        parameters,
        support=spatial_support,
        diff_dim="datetime",
        disagg_dims=disagg_dims,
        scale=False,
    ).isel({"datetime": 0}, drop=True)

    # TEMPORAL FORECAST: smashed space dimensions
    # make a slice and concat the bounds if they are not in the slice already
    # (allows epochs in between the time nodes)
    print("temporal analysis")
    covariate_time = generate_closed_time_series(covariate, epochs, covariate_at_epochs)
    forecast_temporal = generate_temporal_forecast_etf(
        covariate_time,
        measure,
        parameters,
        target_step=target_step,
        support=spatial_support,
        disagg_dims=disagg_dims,
        variance=variance,
        fractiles=fractiles,
    )

    # REALIZATION
    observation_spatiotemporal = get_realisation(covariate_time, event_data, filterset)
    observation_spatial = observation_spatiotemporal.sum("datetime")
    observation_temporal = observation_spatiotemporal.sum("loc")

    testsuite = xr.DataTree.from_dict(
        {
            "spatial/forecast": forecast_spatial,
            "temporal/forecast": forecast_temporal,
        }
    )

    testsuite["spatial/observation"] = observation_spatial
    testsuite["temporal/observation"] = observation_temporal
    testsuite["meta/time_range"] = covariate_time["datetime"]
    testsuite["meta/support"] = spatial_support
    testsuite["meta/measure"] = measure

    if spatiotemporal:
        print("spatiotemporal analysis")
        testsuite["spatiotemporal/forecast"] = generate_forecast_etf(
            covariate_time,
            measure,
            parameters,
            support=spatial_support,
            diff_dim="datetime",
            disagg_dims=disagg_dims,
        )
        testsuite["spatiotemporal/observation"] = observation_spatiotemporal

    return testsuite


def generate_closed_time_series(
    data, epochs, data_at_bounds=None, datetime_dim="datetime"
):
    """
    Generate a time series with closed boundaries, clipped to the epochs. If the epochs are
    beyond the datetime range, the epochs are clipped to the datetime range, otherwise the
    the data is interpolated at the epochs to provide the bounds. Alternatively, explicit values
    at the bounds may be provided.
    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        Data to be clipped
    epochs : array-like
        Epochs to clip the data to
    data_at_bounds : xr.Dataset, optional
        Data at the bounds, by default None
    datetime_dim : str, optional
        Name of the datetime dimension, by default "datetime"
    Returns
    -------
    xr.Dataset or xr.DataArray
        Clipped data

    """
    epochs = np.clip(
        epochs.astype("datetime64[ns]"),
        data[datetime_dim].data[0],
        data[datetime_dim].data[-1],
    )

    # interpolate data
    if data_at_bounds is None:
        data_at_bounds = data.interp({datetime_dim: epochs.data})
    data_inside = data.sel({datetime_dim: slice(epochs[0], epochs[-1])})
    bool_filter = np.logical_not(
        data_inside[datetime_dim].isin(data_at_bounds[datetime_dim])
    )
    data_inside = xr.concat(
        [data_inside.sel({datetime_dim: bool_filter}), data_at_bounds],
        dim=datetime_dim,
        data_vars="all",
    ).sortby(datetime_dim)

    return data_inside
