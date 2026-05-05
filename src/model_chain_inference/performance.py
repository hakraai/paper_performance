"""Synthetic catalogue generation and forecast performance assessment helpers."""

import numpy as np
import xarray as xr
from .statistics import (
    get_cumulative_probabilities_poisson,
    get_default_fractiles,
    compute_log_likelihood,
)
from .model_forecast import (
    get_realisation,
    generate_closed_time_series,
)


def generate_simulations(observed_counts, expected_counts, sample_size, rng=None):
    """
    Generate synthetic catalogues based on expected counts and observed counts.
    Parameters
    ----------

    observed_counts : xr.DataArray
        Observed counts, e.g. from a catalogue
    expected_counts : xr.DataArray
        Expected counts, e.g. from a forecast
    sample_size : int
        Number of synthetic catalogues to generate
    rng : np.random.Generator, optional
        Random number generator, by default None
    Returns
    -------
    xr.Dataset
        Dataset containing synthetic catalogues, with two types:
        - "normalized": normalized to the observed counts
        - "regular": regular synthetic catalogues
    """
    simulations = xr.Dataset(
        {
            "normalized": xr_synthetic_catalogues_normalized(
                expected_counts,
                observed_counts.sum(),
                sample_size=sample_size,
                rng=rng,
            ),
            "regular": xr_synthetic_catalogues(
                expected_counts,
                sample_size=sample_size,
                rng=rng,
            ),
        }
    ).to_array("simulation_type")

    return simulations


def create_adaptive_cell_covering(array, collection):
    """Construct a non-overlapping adaptive cell covering from aggregate selections."""
    def cell_covering(array):
        if len(array.dims) > 0:
            array = array.stack(cell=array.dims).dropna("cell")
        return xr.ones_like(array)

    cell_coverage_dict = {k: cell_covering(v) for k, v in collection.items()}
    cell_list = [
        level.sel(cell=[lvl]).unstack("cell")
        for level in cell_coverage_dict.values()
        if "cell" in level.dims  # and len(level["loc"]) > 0
        for lvl in level["cell"].values
    ]
    cell_list.append(xr.ones_like(array.drop_vars(["x", "y"])))
    cell_covering = xr.concat(
        cell_list,
        dim="cell",
        coords="all",
        join="outer",
    ).rename("cell_covering")
    cell_covering["x"] = array["x"]
    cell_covering["y"] = array["y"]

    first_cell = cell_covering.cumsum("cell").cumsum("cell") == 1
    cell_covering = cell_covering.where(first_cell)

    return cell_covering


def add_multiscale_index(data, spatial_coordinates=None):
    """Expand spatial coordinates into a dyadic multiscale indexing scheme."""
    if spatial_coordinates is None:
        spatial_coordinates = "x", "y"
    x, y = spatial_coordinates
    multiscale_data = data
    c_dict = {x: 2, y: 2}
    factor = 1
    while c_dict:
        factor = factor * 2
        u = f"x{factor:04d}"
        v = f"y{factor:04d}"
        multiscale_data = multiscale_data.coarsen(c_dict, boundary="pad").construct(
            {x: (x + "x", u), y: (y + "y", v)}
        )
        x = x + "x"
        y = y + "y"
        c_dict = {}
        if multiscale_data[x].size > 1:
            c_dict[x] = 2
        if multiscale_data[y].size > 1:
            c_dict[y] = 2
    multiscale_data = multiscale_data.squeeze().fillna(0)

    xs = list(np.sort(multiscale_data["x"].dims))
    ys = list(np.sort(multiscale_data["y"].dims))

    # trick to add simple [0,1] indices
    multiscale_data = multiscale_data.stack(X=xs, Y=ys).unstack("X").unstack("Y")

    return multiscale_data


def get_combined_dimensions(spatial_ds, ms_mode=None):
    """Return the aggregation order used for multiscale spatial coarsening."""
    if ms_mode is None:
        ms_mode = "rectangular"  # "square" or "rectangular"

    xs = list(np.sort(spatial_ds["x"].dims))
    ys = list(np.sort(spatial_ds["y"].dims))

    if ms_mode == "rectangular":
        combined_dimensions = (
            [v for pair in zip(ys, xs) for v in pair]
            + list(xs)[len(ys) :]
            + list(ys)[len(xs) :]
        )
    else:  # ms_mode == "square"
        combined_dimensions = (
            [pair for pair in zip(xs, ys)] + list(xs)[len(ys) :] + list(ys)[len(xs) :]
        )

    return combined_dimensions


def generate_adaptive_cell_covering(ms_spatial_ds, ms_mode=None, threshold=None):
    """Build adaptive spatial cells that meet a minimum-count threshold."""
    if ms_mode is None:
        ms_mode = "rectangular"  # "square" or "rectangular"
    combined_dimensions = get_combined_dimensions(ms_spatial_ds, ms_mode)
    n_total = ms_spatial_ds.sum().item()

    if threshold is None:
        threshold = 8

    collection_complete = {}
    to_sum = ms_spatial_ds.drop_vars(["x", "y"])
    sum_dims = [[]] + combined_dimensions
    for i, d in enumerate(sum_dims):
        to_sum = to_sum.sum(d)
        sufficient = to_sum > threshold
        if len(sufficient.dims) > 0:
            complete = to_sum.where(sufficient)
        else:
            complete = to_sum
        collection_complete[f"sum_{i}"] = complete
        to_sum = to_sum.where(np.logical_not(sufficient), 0)

    cell_covering = create_adaptive_cell_covering(ms_spatial_ds, collection_complete)

    cell_covering = stack_and_align(cell_covering).fillna(0)

    cell_covering.attrs["n_total"] = n_total
    cell_covering.attrs["threshold"] = threshold

    return cell_covering


def get_poisson_cdf_clip(observed_counts, expected_counts):
    """
    Get the tight cumulative distribution function (CDF) for Poisson-distributed counts.
    We use cdf in a tight fashion, i.e. the p-value is the
    value in the cdf interval corresponding to the observed count that is closest to 0.5.

    Parameters
    ----------
    observed_counts : xr.DataArray
        Observed counts, e.g. from a catalogue
    expected_counts : xr.DataArray
        Expected counts, e.g. from a forecast
    Returns
    -------
    xr.DataArray
        Tight CDF for Poisson-distributed counts, with dimensions of the observed counts
    """

    cum_prob = get_cumulative_probabilities_poisson(observed_counts, expected_counts)
    cdf_exc = cum_prob.sel(p_metric="cumulative_exclusive")
    cdf = cum_prob.sel(p_metric="cumulative")
    cdf_tight = xr.apply_ufunc(np.clip, 0.5, cdf_exc, cdf)
    return cdf_tight


def performance_statistics(
    observed_counts,
    expected_counts,
    observed_entropy=None,
    simulated_counts=None,
    simulated_entropy=None,
    q=None,
    bin_dims=None,
    rng=None,
    sample_size=10_000,
):
    """
    Compute performance statistics for the given observed and expected counts, and simulations.
    Parameters
    ----------
    observed_counts : xr.DataArray
        Observed counts, e.g. from a catalogue
    expected_counts : xr.DataArray
        Expected counts, e.g. from a forecast
    observed_log_expected_counts : xr.DataArray, optional
        Observed log counts, by default None. If None, it will be computed
        from the observed and expected counts.
    simulated_counts : xr.DataArray, optional
        Simulated counts, e.g. from synthetic catalogues, by default None. If None
        it will be generated from the observed and expected counts.
    simulated_log_expected_counts : xr.DataArray, optional
        Simulated log counts, by default None. If None, it will be computed
        from the simulated and expected counts.
    q : xr.DataArray, optional
        Quantiles to compute, e.g. from get_default_fractiles(), by default None
    bin_dims : list, optional
        Dimensions over which to compute the performance statistics, by default None. If None,
        it will be set to ["loc"].
    rng : np.random.Generator, optional
        Random number generator, by default None
    sample_size : int, optional
        Number of synthetic catalogues to generate if simulated_counts is None, by default 10_000

    Returns
    -------
    xr.Dataset
        Dataset containing performance statistics, including log-likelihoods, p-values, and test results
    """
    if q is None:
        q = get_default_fractiles()
    if bin_dims is None:
        bin_dims = ["loc"]

    if simulated_counts is None:
        simulated_counts = generate_simulations(
            observed_counts,
            expected_counts,
            sample_size,
            rng,
        )

    present = expected_counts > 0
    if observed_entropy is None:
        observed_entropy = xr.dot(
            observed_counts,
            -np.log(expected_counts.where(present, 1)),
            dims=bin_dims,
        )
    if simulated_entropy is None:
        simulated_entropy = xr.dot(
            simulated_counts,
            -np.log(expected_counts.where(present, 1)),
            dims=bin_dims,
        )

    ll_simulations = compute_log_likelihood(
        expected_counts,
        simulated_counts,
        simulated_entropy,
        bin_dims,
    )
    ll_observed = compute_log_likelihood(
        expected_counts,
        observed_counts,
        observed_entropy,
        bin_dims,
    )

    p_value = (ll_observed < ll_simulations).mean(dim="catalogue")

    # test result 1: p vs q
    test_result_1 = np.logical_and(
        p_value >= q.isel(fractile=0),
        p_value <= q.isel(fractile=-1),
    )

    # test result 2: ll vs q-quantiles
    ll_fractiles = (
        ll_simulations.quantile(q, dim="catalogue")
        .rename({"quantile": "fractile"})
        .assign_coords(fractile=["lower", "upper"])
    )
    test_result_2 = np.logical_and(
        ll_observed >= ll_fractiles.isel(fractile=0),
        ll_observed <= ll_fractiles.isel(fractile=-1),
    )

    metrics_c = xr.Dataset(
        {
            "ll_fractiles": ll_fractiles,
            "ll_observed": ll_observed,
            "p_value": p_value,
            "p_test_result": test_result_1,
            "ll_test_result": test_result_2,
            "q": q,
        }
    )

    return metrics_c


def temporal_performance_assessment(
    temporal_ds: xr.Dataset,
    sample_size=10_000,
    rng=None,
) -> xr.Dataset:
    """
    Perform temporal performance assessment on the given dataset.

    Parameters:
        temporal_ds (xr.Dataset): The dataset containing observations and forecasts.

    Returns:
        xr.Dataset: The performance statistics.
    """
    # TODO:  do not use this as a special case, just marginalize, and identify the bin dims

    perfstat = performance_statistics(
        temporal_ds["observations"],
        temporal_ds["forecast"],
        bin_dims=["datetime"],
        sample_size=sample_size,
        rng=rng,
    )

    perfstat["temporal_statistics"] = cell_statistics(temporal_ds).to_array("type")

    return perfstat


def adaptive_spatial_performance_assessment(
    spatial_ds: xr.Dataset, cell_covering: xr.DataArray, sample_size=10_000, rng=None
) -> xr.Dataset:
    """
    Perform adaptive spatial performance assessment on the given dataset.

    Parameters:
        spatial_ds (xr.Dataset): The dataset containing observations and forecasts.
        adaptive_cell_covering (xr.DataArray): The adaptive cell covering.

    Returns:
        xr.Dataset: The performance statistics.
    """
    spatial_ds_xy = stack_and_align(spatial_ds)
    present = spatial_ds_xy["forecast"].fillna(0) > 0
    spatial_ds_xy["simulations"] = generate_simulations(
        spatial_ds_xy["observations"].fillna(0),
        spatial_ds_xy["forecast"].fillna(0),
        sample_size,
        rng=rng,
    )
    observed_entropy = xr.dot(
        spatial_ds_xy["observations"].fillna(0),
        -np.log(spatial_ds_xy["forecast"].fillna(0).where(present, 1)),
        dims=["X", "Y"],
    )
    simulated_entropy = xr.dot(
        spatial_ds_xy["simulations"],
        -np.log(spatial_ds_xy["forecast"].fillna(0).where(present, 1)),
        dims=["X", "Y"],
    )

    cells = spatial_ds_xy.map(
        lambda v: v.dot(cell_covering.fillna(0)),
    )
    perfstat = performance_statistics(
        cells["observations"],
        cells["forecast"],
        bin_dims=["cell"],
        observed_entropy=observed_entropy,
        simulated_counts=cells["simulations"],
        simulated_entropy=simulated_entropy,
        rng=rng,
        sample_size=sample_size,
    )
    perfstat = perfstat.merge(cells)

    perfstat["cell_statistics"] = cell_statistics(cells).to_array("type")
    perfstat["spatial_statistics"] = xr.dot(perfstat["cell_statistics"], cell_covering)

    return perfstat


def multiscale_spatial_performance_assessment(
    ms_spatial_ds, ms_mode=None, sample_size=10_000, rng=None
):
    """Evaluate spatial performance statistics over successive coarsening levels."""
    # many scale collection
    to_sum = ms_spatial_ds.drop_vars(["x", "y"])
    combined_dims = get_combined_dimensions(ms_spatial_ds, ms_mode=ms_mode)
    sum_dims = [[]] + combined_dims
    bin_dims = ["loc"]

    to_sum["simulations"] = generate_simulations(
        to_sum["observations"].fillna(0),
        to_sum["forecast"].fillna(0),
        sample_size,
    )

    def __f__(v):
        ds = cell_statistics(v).to_array("type")
        if len(ds["loc"]) > 1:
            ds = ds.unstack("loc")
        else:
            ds = ds.squeeze("loc")
        return ds

    first_pass = True
    perf_stats = []
    spatial_stats = []
    for d in sum_dims:
        to_sum = to_sum.sum(d)
        stack_dims = [dim for dim in to_sum.dims if dim in combined_dims]
        if len(stack_dims) == 0:
            to_assess = to_sum.expand_dims("loc", axis=0)
        else:
            to_assess = to_sum.stack(loc=stack_dims)
            present = to_assess["forecast"] > 0
            to_assess = to_assess.where(present, drop=True)
        if first_pass:  # first iteration, full resolution
            observed_entropy = xr.dot(
                to_assess["observations"],
                -np.log(to_assess["forecast"]),
                dims=bin_dims,
            )
            simulated_entropy = xr.dot(
                to_assess["simulations"],
                -np.log(to_assess["forecast"]),
                dims=bin_dims,
            )
            first_pass = False
        stats = performance_statistics(
            to_assess["observations"],
            to_assess["forecast"],
            observed_entropy=observed_entropy,
            simulated_counts=to_assess["simulations"],
            simulated_entropy=simulated_entropy,
            bin_dims=bin_dims,
            rng=rng,
            sample_size=sample_size,
        )
        perf_stats.append(stats)
        spatial_stats.append(__f__(to_assess))
    perf_stats = xr.concat(perf_stats, dim="level")

    spatial_stats = xr.concat(
        spatial_stats,
        dim="level",
        coords="minimal",
        compat="override",
    )
    spatial_stats["x"] = ms_spatial_ds["x"]
    spatial_stats["y"] = ms_spatial_ds["y"]

    spatial_stats = stack_and_align(spatial_stats).rename("spatial_statistics")

    perf_stats = perf_stats.merge(spatial_stats)

    return perf_stats


def cell_statistics(v):
    """Derive forecast, observation, density, and CDF summaries for each cell."""

    n_obs = v["observations"].sum()
    n_exp = v["forecast"].sum()
    expected_counts_normalized = (n_obs / n_exp) * v["forecast"]
    cdf_clip = get_poisson_cdf_clip(v["observations"], expected_counts_normalized)

    support = v.get("support_fraction", 1)
    result = xr.Dataset(
        {
            "forecast": v["forecast"],
            "observations": v["observations"],
            "forecast_density": v["forecast"] / support,
            "observation_density": v["observations"] / support,
            "cdf_clip": cdf_clip,
        }
    )
    for key in ["forecast", "observations", "forecast_density", "observation_density"]:
        result[f"normalized_{key}"] = result[key] / result[key].max()
    return result


def xr_synthetic_catalogues_normalized(rates, count, sample_size, rng=None):
    """Wrap fixed-size catalogue simulation so it broadcasts cleanly over xarray dims."""
    return xr.apply_ufunc(
        simulate_catalogues_fixed_size,
        rates,
        count,
        kwargs={"sample_size": sample_size, "rng": rng},
        output_core_dims=[["catalogue"]],
    ).rename("event_count")


def xr_synthetic_catalogues(rates, sample_size, rng=None):
    """Wrap Poisson catalogue simulation so it broadcasts cleanly over xarray dims."""
    return xr.apply_ufunc(
        simulate_catalogues_poisson,
        rates,
        kwargs={"sample_size": sample_size, "rng": rng},
        output_core_dims=[["catalogue"]],
    ).rename("event_count")


def simulate_catalogues_fixed_size(rates, count=None, sample_size=10_000, rng=None):
    """
    Simulate synthetic catalogues of fixed count, using the multinomial distribution.
    Parameters
    ----------
    rates : xr.DataArray
        Rates for each bin, e.g. from a forecast
    count : int, optional
        Total count for the multinomial distribution, by default None. In that case the total count
        is taken as the round sum of the rates.
    sample_size : int, optional
        Number of synthetic catalogues to generate, by default 10_000
    rng : np.random.Generator, optional
        Random number generator, by default None
    Returns
    -------
    np.ndarray
        Array of shape rates.shape + (sample_size,) containing the simulated catalogues
    """
    rng = np.random.default_rng(rng)

    expectation = np.nansum(rates)
    if count is None:
        count = expectation.round().astype(int)
    pmf = rates / expectation
    count = np.full(sample_size, count).astype(int)

    new_shape = pmf.shape + count.shape
    simulations = np.zeros(new_shape, dtype=int)

    simulations[pmf > 0] = np.moveaxis(
        rng.multinomial(count, pvals=pmf[pmf > 0]), -1, 0
    )

    return simulations


def simulate_catalogues_poisson(rates, sample_size=10_000, rng=None):
    """Simulate synthetic catalogues with Poisson-distributed total event counts."""
    rng = np.random.default_rng(rng)

    expectation = np.nansum(rates)
    pmf = rates / expectation

    count = rng.poisson(expectation, size=sample_size)

    new_shape = pmf.shape + count.shape
    simulations = np.zeros(new_shape, dtype=int)

    simulations[pmf > 0] = np.moveaxis(
        rng.multinomial(count, pvals=pmf[pmf > 0]), -1, 0
    )

    return simulations


def stack_and_align(input_ds):
    """Stack x and y dimensions into aligned sparse X and Y axes."""
    output_ds = (
        input_ds.stack(
            {
                "X": input_ds["x"].dims,
                "Y": input_ds["y"].dims,
            }
        )
        .sortby("x")
        .sortby("y")
    )
    x_present = output_ds["x"].notnull()
    y_present = output_ds["y"].notnull()
    output_ds = output_ds.where(x_present & y_present, drop=True)
    return output_ds


def create_spatial_ds(testsuite):
    """Extract spatial forecasts, observations, and entropy terms from a testsuite."""
    spatial_ds = xr.Dataset(
        {
            "observations": testsuite["spatial/observation"],
            "forecast": testsuite["spatial/forecast/mean"],
        }
    )
    spatial_ds["support_fraction"] = testsuite["meta/support"]
    spatial_ds["observed_entropy"] = observed_entropy(
        spatial_ds["forecast"], spatial_ds["observations"]
    )
    spatial_ds = (
        spatial_ds.unstack("loc")
        .dropna("x", how="all")
        .dropna("y", how="all")
        .reset_coords(drop=True)
    ).fillna(0)
    # spatial_ds["node_count"] = xr.ones_like(spatial_ds["forecast"])
    return spatial_ds


def observed_entropy(forecast, observations):
    """Compute the entropy-like term used by heterogeneous Poisson likelihoods."""
    present = forecast > 0
    log_expected = np.log(forecast.where(present, 1))
    return forecast - observations * log_expected


def create_temporal_ds(testsuite):
    """Extract temporal forecasts, observations, and entropy terms from a testsuite."""
    temporal_ds = xr.Dataset(
        {
            "observations": testsuite["temporal/observation"],
            "forecast": testsuite["temporal/forecast/mean"],
        }
    )
    temporal_ds["observed_entropy"] = observed_entropy(
        temporal_ds["forecast"], temporal_ds["observations"]
    )

    return temporal_ds


def generate_cell_covering(
    grid_data, event_data, filterset, ms_mode="rectangular", threshold=None
):
    """Generate an adaptive cell covering from grid data, events, and a filter set."""
    data = generate_closed_time_series(grid_data, filterset["timeframe"])
    realisation = get_realisation(data, event_data, filterset)
    realisation, data = xr.align(realisation, data)
    spatial_ds = realisation.sum(["datetime"])
    ms_spatial_ds = add_multiscale_index(spatial_ds)
    cell_covering = generate_adaptive_cell_covering(
        ms_spatial_ds, ms_mode=ms_mode, threshold=threshold
    )
    return cell_covering


__all__ = [
    "adaptive_spatial_performance_assessment",
    "add_multiscale_index",
    "cell_statistics",
    "create_adaptive_cell_covering",
    "create_spatial_ds",
    "create_temporal_ds",
    "generate_adaptive_cell_covering",
    "generate_cell_covering",
    "generate_simulations",
    "get_combined_dimensions",
    "get_poisson_cdf_clip",
    "multiscale_spatial_performance_assessment",
    "observed_entropy",
    "performance_statistics",
    "simulate_catalogues_fixed_size",
    "simulate_catalogues_poisson",
    "stack_and_align",
    "temporal_performance_assessment",
    "xr_synthetic_catalogues",
    "xr_synthetic_catalogues_normalized",
]
