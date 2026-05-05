"""Testsuite construction and diagnostic orchestration for calibrated models."""

import arviz as az
import xarray as xr
import matplotlib.pyplot as plt
import model_chain_inference as mci


def prepare_testsuite(
    idata,
    grid_data,
    event_data,
    filterset,
    rng=None,
    n_posterior_samples=None,
):
    """Build a forecast-vs-observation testsuite from posterior samples and inputs."""
    if rng is None:
        rng = 42
    if n_posterior_samples is None:
        n_posterior_samples = 1_000

    posterior_samples = az.extract(idata, num_samples=n_posterior_samples, rng=rng)
    constant_data = idata["constant_data"].squeeze(drop=True)
    covariate_scale = constant_data.get("covariate_scale", 1.0)

    def itp_f(v):
        itp_dims = [v_id for v_id in posterior_samples.data_vars if v_id in v.dims]
        return v.interp(
            {v_id: posterior_samples[v_id] for v_id in itp_dims},
            method="linear",
        ).drop_vars(itp_dims)

    constant_data = constant_data.map(itp_f)

    parameters = xr.merge(
        [
            posterior_samples,
            constant_data,
        ],
    )

    cov_id = idata.attrs["covariate_id"]
    meas_id = idata.attrs["measure_id"]
    supp_id = idata.attrs["support_id"]

    testsuite = mci.generate_testsuite_etf(
        event_data,
        parameters,
        grid_data[cov_id] / covariate_scale,
        grid_data[meas_id],
        grid_data[supp_id],
        filterset=filterset,
        variance=True,  # only for temporal
    )
    testsuite.attrs.update(idata.attrs)

    return testsuite


def analyze_and_display(
    model_id,
    testsuite,
    run_id,
    purpose,
    cell_covering,
    groningen_contour,
    plotting_timeframe,
):
    """Run the standard diagnostics and render the main investigation plots."""
    title = f"m:{model_id} - c:{run_id} - f:{purpose}"
    spatial_ds = mci.create_spatial_ds(testsuite)
    temporal_ds = mci.create_temporal_ds(testsuite)
    ms_spatial_ds = mci.add_multiscale_index(spatial_ds)
    ms_perf_stats = mci.multiscale_spatial_performance_assessment(ms_spatial_ds)

    adaptive_perf_stats = mci.adaptive_spatial_performance_assessment(
        ms_spatial_ds, cell_covering
    )
    time_perf_stats = mci.temporal_performance_assessment(temporal_ds)

    mci.plot_spatial_statistics(
        ms_perf_stats.thin(level=2), groningen_contour, title=title
    )
    g = (
        adaptive_perf_stats["spatial_statistics"]
        .sel(
            type=[
                "normalized_observation_density",
                "normalized_forecast_density",
                "cdf_clip",
            ]
        )
        .plot(x="x", y="y", col="type")
    )
    mci.postprocess_facets(g, groningen_contour)
    mci.plot_test_results(ms_perf_stats.thin(level=2), dim="level", title=title)
    # mci.plot_spatial_coarsening(perfass, groningen_contour, title)
    # mci.plot_test(perfass, title)
    mci.plot_test_results(
        xr.concat(
            [time_perf_stats, adaptive_perf_stats],
            dim="test",
        ).assign_coords(
            test=["temporal", "spatial - adaptive"],
        ),
        title=title,
        dim="test",
    )
    mci.plot_time_series(
        testsuite,
        title,
        timeframe=plotting_timeframe,
        top=40,
    )
    plt.show()


def perf_assessment(testsuite, cell_covering, sample_size=10_000, rng=None):
    """Run temporal, multiscale spatial, and adaptive spatial assessments."""
    # temporal testing
    temporal_ds = mci.create_temporal_ds(testsuite)
    time_perf_stats = mci.temporal_performance_assessment(
        temporal_ds,
        sample_size=sample_size,
        rng=rng,
    )
    out_t = time_perf_stats

    # spatial testing
    spatial_ds = mci.create_spatial_ds(testsuite)
    ms_spatial_ds = mci.add_multiscale_index(spatial_ds)

    # spatial multiscale testing
    ms_perf_stats = mci.multiscale_spatial_performance_assessment(
        ms_spatial_ds,
        sample_size=sample_size,
        rng=rng,
    )
    out_s = ms_perf_stats

    # spatial adaptive testing
    adaptive_perf_stats = mci.adaptive_spatial_performance_assessment(
        ms_spatial_ds,
        cell_covering,
        sample_size=sample_size,
        rng=rng,
    )
    out_sa = adaptive_perf_stats

    return out_t, out_s, out_sa
