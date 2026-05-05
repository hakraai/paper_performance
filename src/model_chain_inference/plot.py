"""Plotting helpers for forecast diagnostics and performance summaries."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib.patches as patches
import model_chain_inference as mci


def create_contour_patch(polygon, **kwargs):
    """Create a Matplotlib polygon patch from a contour geometry."""
    x, y = polygon.exterior.coords.xy
    return patches.Polygon(np.array([x, y]).T, fc="none", **kwargs)


def postprocess_ax(ax, polygon, xticks=None, yticks=None, **kwargs):
    """Clip spatial plot content to a polygon and remove axis decoration."""
    # return for empty axes
    if len(ax.collections) == 0:
        return
    gpatch = create_contour_patch(polygon, **kwargs)
    xy = gpatch.xy
    xmin = xy[:, 0].min()
    xmax = xy[:, 0].max()
    ymin = xy[:, 1].min()
    ymax = xy[:, 1].max()
    ax.set_xticks(xticks if xticks is not None else [])
    ax.set_yticks(yticks if yticks is not None else [])
    ax.add_patch(gpatch)
    ax.collections[0].set_clip_path(gpatch)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", "box")
    ax.set_axis_off()


def postprocess_facets(g, polygon, **kwargs):
    """Apply polygon clipping and axis cleanup to every facet in a grid."""
    for x in g.axs.flat:
        postprocess_ax(x, polygon, **kwargs)
    return g


def make_test_axis(
    ax, obs, fractiles, test_result, xlabel, ylabel, ms=10, capsize=6, pval=None
):
    """Draw a horizontal interval plot for one set of likelihood test results."""
    stacked_fractiles = fractiles.stack({"q": [...]})
    mn = np.amin([obs, stacked_fractiles.min()])
    mx = np.amax([obs, stacked_fractiles.max()])
    if mx == mn:
        mn = mn - 0.5
        mx = mn + 1.0
    mn = mn - 0.1 * (mx - mn)
    mx = mx + 0.1 * (mx - mn)
    nbar = fractiles.shape[0]
    frac_lo = stacked_fractiles.sel(fractile="lower")
    frac_up = stacked_fractiles.sel(fractile="upper")
    for i, (q0, q1, result) in enumerate(zip(frac_lo, frac_up, test_result)):
        color = "green" if result else "red"
        ax.errorbar(
            0.5 * (q0 + q1),
            i,
            xerr=0.5 * (q1 - q0),
            fmt="",
            capsize=capsize,
            ms=5,
            c="k",
        )
        ax.plot(obs, i, "o", ms=ms, c=color)
        if pval is not None:
            offsetx = (mx - mn) * 0.05
            p = str()
            ax.annotate(
                text=f"{pval[i].item():.3G}" + p,
                xy=(mx - offsetx, i + 0.2),
                ha="right",
                size=7,
            )

    ax.set_xlim(mn, mx)
    ax.set_xticks([])
    ax.set_ylim([-1.0, nbar])
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_time_series(
    suite,
    title,
    timeframe=None,
    top=None,
    legend=False,
    ax=None,
    legend_loc="upper right",
    separation_date=None,
):
    """Plot temporal forecasts and observed counts for a testsuite."""
    if timeframe is None:
        t0, t1 = suite["meta/time_range"]["datetime"].data[[0, -1]]
    else:
        t0, t1 = timeframe
    if ax is None:
        ax = plt.subplots()[1]
    isel = {}
    sel = {}
    fc = suite["temporal/forecast"]
    if "strain_mode" in fc.dims:
        isel["strain_mode"] = -1
    if "bernstein_index" in fc.dims:
        isel["bernstein_index"] = -1
    if "etas_generation" in fc.dims:
        fc = fc.sum("etas_generation")
    if "var" in suite["temporal/forecast"]:
        var = suite["temporal/forecast/var"].isel(isel).sel(sel)
    else:
        var = None
    plot_time_forecast(
        suite["temporal/forecast/mean"].isel(isel).sel(sel),
        variance=var,
        ax=ax,
        final_date=suite["meta/time_range"]["datetime"].data[-1],
    )
    plot_time_realisation(
        suite["temporal/observation"],
        ax=ax,
        final_date=suite["meta/time_range"]["datetime"].data[-1],
        label="observed count",
    )
    # add vertical line at the end of the forecast period
    if separation_date is not None:
        ax.axvline(
            x=separation_date,
            color="k",
            linestyle="--",
            linewidth=1,
        )

    ax.grid(True, which="both", axis="both", color="lightgray")
    ax.set_xlim(left=np.datetime64(t0), right=np.datetime64(t1))
    ax.set_ylim(bottom=0, top=top)
    if legend:
        ax.legend(loc=legend_loc)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Annual rate/count")


def plot_time_realisation(realisation, ax, label=None, c="k", final_date=None):
    """Plot an observed realization as a stepwise time series."""
    if final_date is not None:
        final_date = np.datetime64(final_date)
        realisation_extended = realisation.isel(
            {"datetime": -1}, drop=True
        ).expand_dims({"datetime": [final_date]})
        realisation = xr.concat([realisation, realisation_extended], dim="datetime")

    if label is None:
        label = "realisation"
    realisation.plot.step(
        x="datetime",
        where="post",
        ax=ax,
        lw=2,
        c=c,
        label=label,
    )


def plot_time_forecast(
    mean, variance=None, ax=None, label=None, q=None, final_date=None
):
    """Plot the mean temporal forecast with count and rate prediction intervals."""
    if label is None:
        label = [
            "mean annual earthquake rate",
            "95% PI annual earthquake count",
            "95% PI annual earthquake rate",
        ]
    if final_date is not None:
        final_date = np.datetime64(final_date)
        mean_extended = mean.isel({"datetime": -1}, drop=True).expand_dims(
            {"datetime": [final_date]}
        )
        mean = xr.concat([mean, mean_extended], dim="datetime")

        if variance is not None:
            variance_extended = variance.isel({"datetime": -1}, drop=True).expand_dims(
                {"datetime": [final_date]}
            )
            variance = xr.concat([variance, variance_extended], dim="datetime")

    count_fractiles = mci.statistics.get_count_fractiles(mean, variance, q)
    dt = count_fractiles["datetime"]

    forecast_line = mean.plot.step(
        x="datetime",
        where="post",
        ax=ax,
        c="black",
        linestyle=(0, (1.0, 1.2)),
        lw=2.5,
        label=label[0],
    )
    forecast_line[0].set_dash_capstyle("round")
    if variance is None:
        count_percs = count_fractiles.sel(distribution="poisson")
        ax.fill_between(
            dt,
            count_percs[0],
            count_percs[-1],
            facecolor="silver",
            step="post",
            label=label[1],
        )
    else:
        count_percs = count_fractiles.sel(distribution="nbinom")
        norm_percs = count_fractiles.sel(distribution="norm")
        ax.fill_between(
            dt,
            count_percs[0],
            count_percs[-1],
            facecolor="silver",
            step="post",
            label=label[1],
        )
        ax.fill_between(
            dt,
            norm_percs[0],
            norm_percs[-1],
            facecolor="grey",
            step="post",
            label=label[2],
        )


__all__ = [
    "create_contour_patch",
    "make_test_axis",
    "plot_time_forecast",
    "plot_time_realisation",
    "plot_time_series",
    "postprocess_ax",
    "postprocess_facets",
]
