from __future__ import annotations

from pathlib import Path
from typing import Callable

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr

from workflow_support import assessment_runtime as runtime
from workflow_support.logging import get_logger


mci = runtime.mci
DEFAULT_FIGURE_FORMATS = [".pdf", ".png", ".eps"]
PDF_RASTER_DPI = 300
LOGGER = get_logger(__name__)


def normalize_figure_formats(value: object) -> list[str]:
    if value is None:
        return list(DEFAULT_FIGURE_FORMATS)
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    normalized = []
    for item in values:
        extension = str(item).strip()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        normalized.append(extension.lower())
    if not normalized:
        return list(DEFAULT_FIGURE_FORMATS)
    return list(dict.fromkeys(normalized))


def get_figure_paths(figpath: Path, base_name: str, figure_formats: list[str]) -> list[Path]:
    return [figpath / f"{base_name}{extension}" for extension in figure_formats]


def ensure_figure_outputs(figpath: Path, base_name: str, figure_formats: list[str], cache_mode: str) -> bool:
    paths = get_figure_paths(figpath, base_name, figure_formats)
    existing = [path for path in paths if path.exists()]
    if not existing:
        return True
    if len(existing) == len(paths) and cache_mode == "reuse":
        for path in existing:
            LOGGER.info("cached %s", path)
        return False
    if cache_mode == "refresh":
        return True
    if cache_mode == "reuse":
        return True
    existing_text = ", ".join(str(path) for path in existing)
    raise FileExistsError(
        f"Refusing to overwrite existing output: {existing_text}. Use --force or --cache refresh to replace it, or --cache reuse to keep it."
    )


def save_current_figure(figpath: Path, base_name: str, figure_formats: list[str]) -> None:
    figpath.mkdir(parents=True, exist_ok=True)
    for path in get_figure_paths(figpath, base_name, figure_formats):
        save_kwargs = {"dpi": PDF_RASTER_DPI} if path.suffix.lower() in {".pdf", ".eps"} else {}
        plt.savefig(path, **save_kwargs)


def rasterize_primary_facet_collections(facets: object) -> object:
    for ax in facets.axs.flat:
        if ax.collections:
            ax.collections[0].set_rasterized(True)
    return facets


def maybe_render_formats(
    figpath: Path,
    base_name: str,
    figure_formats: list[str],
    cache_mode: str,
    render: Callable[[], None],
) -> None:
    if not ensure_figure_outputs(figpath, base_name, figure_formats, cache_mode):
        return
    render()


def has_models(store: dict[str, object], required_model_ids: list[str]) -> bool:
    return all(model_id in store for model_id in required_model_ids)


def has_perspective_models(
    store: dict[str, dict[str, object]], perspective: str, required_model_ids: list[str]
) -> bool:
    return perspective in store and has_models(store[perspective], required_model_ids)


def log_skipped_figure(base_name: str, reason: str) -> None:
    LOGGER.info("stage=assessment status=skip-figure figure=%s reason=%s", base_name, reason)


def generate_model_time_series_plot(
    model_ids: list[str],
    base_name: str,
    testsuite_dict: dict[str, xr.DataTree],
    model_names: dict[str, str],
    figpath: Path,
    timeframe_plotting: list[str],
    figure_formats: list[str],
    figsize: tuple[float, float] = (10, 7),
) -> None:
    plt.close("all")
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharey=True, sharex=True)
    axsf = axs.ravel()
    for ax, mid in zip(axsf, model_ids):
        forecast = testsuite_dict[mid]
        separation_date = np.datetime64(forecast.attrs["timeframe"][1])
        mci.plot_time_series(
            forecast,
            model_names[mid],
            timeframe=timeframe_plotting,
            top=45,
            legend=False,
            ax=ax,
            separation_date=separation_date,
        )
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        fontsize=9,
        framealpha=1,
    )
    letters = ["(a)", "(b)", "(c)", "(d)"]
    for ax, letter in zip(axsf, letters):
        ax.text(
            -0.06,
            1.02,
            letter,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontweight="bold",
            clip_on=False,
        )
    for r in range(2):
        for c in range(2):
            if r < 1:
                axs[r, c].set_xlabel("")
            if c > 0:
                axs[r, c].set_ylabel("")
    plt.tight_layout()
    save_current_figure(figpath, base_name, figure_formats)


def plot_likelihood_tests_2x4(
    model_ids: list[str],
    base_name: str,
    t_prospective: dict[str, xr.Dataset],
    t_retrospective: dict[str, xr.Dataset],
    model_names: dict[str, str],
    figpath: Path,
    figure_formats: list[str],
    figsize: tuple[float, float] = (12, 4.8),
) -> None:
    plt.close("all")
    ncols = len(model_ids)
    fig, axs = plt.subplots(2, ncols, figsize=figsize, sharex=False, sharey=True)
    if ncols == 1:
        axs = np.array([[axs[0]], [axs[1]]])

    def draw_panel(ax: object, mid: str, tdict: dict[str, xr.Dataset]) -> None:
        ds = tdict[mid].sel({"likelihood_function": "multi_hom_poisson"})
        mci.plot.make_test_axis(
            ax,
            ds["ll_observed"],
            ds["ll_fractiles"],
            ds["ll_test_result"],
            "",
            "",
            pval=ds["p_value"],
            capsize=4,
        )

    scenarios = [("prospective", t_prospective), ("retrospective", t_retrospective)]
    for r, (row_label, tdict) in enumerate(scenarios):
        for c, mid in enumerate(model_ids):
            ax = axs[r, c]
            draw_panel(ax, mid, tdict)
            ax.set_title(f"{model_names[mid]}\n{row_label}" if r == 0 else row_label)
    for r in range(2):
        for c in range(ncols):
            if c > 0:
                axs[r, c].set_yticklabels([])
                axs[r, c].set_ylabel("")
    for r in range(2):
        axs[r, 0].set_yticks([0, 1])
        axs[r, 0].set_yticklabels(["conditional", "unconditional"])
    letters = [f"({chr(97 + i)})" for i in range(2 * ncols)]
    for r in range(2):
        for c in range(ncols):
            axs[r, c].text(
                -0.06,
                1.02,
                letters[r * ncols + c],
                transform=axs[r, c].transAxes,
                ha="right",
                va="bottom",
                fontweight="bold",
                clip_on=False,
            )
    save_current_figure(figpath, base_name, figure_formats)


def plot_spatial_statistics_paper(
    perf_stats: dict[str, xr.Dataset],
    polygon: object,
    type_list: list[str],
    title: str | None = None,
) -> object:
    thinlevel = 2
    merge_list = []
    for i, key in enumerate(perf_stats.keys()):
        type_list_local = type_list if i == 0 else type_list[1:]
        plot_stats = perf_stats[key]["spatial_statistics"].sel(type=type_list_local)
        new_labels = [label + key for label in type_list_local]
        plot_stats = plot_stats.assign_coords({"type": new_labels})
        merge_list.append(plot_stats)
    plot_dat = xr.concat(merge_list, dim="type")
    g = plot_dat.thin(level=thinlevel).plot(
        x="x",
        y="y",
        col="level",
        row="type",
        add_colorbar=False,
        figsize=(10.5, 12),
    )
    g = rasterize_primary_facet_collections(g)
    if title is not None:
        g.fig.subplots_adjust(top=0.94)
        g.fig.suptitle(title, x=0.5)
    return mci.postprocess_facets(g, polygon)


def generate_spatial_performance_plots(
    base_name: str,
    model_id_1: str,
    model_id_2: str,
    spatial_stats: dict[str, xr.Dataset],
    model_names: dict[str, str],
    groningen_contour: object,
    figpath: Path,
    figure_formats: list[str],
    title: str | None = None,
) -> None:
    type_list = ["normalized_observations", "normalized_forecast", "cdf_clip"]
    level_list = [
        "L[0,0] - 0.5 km",
        "L[1,1] - 1 km",
        "L[2,2] - 2 km",
        "L[3,3] - 4 km",
        "L[4,4] - 8 km",
        "L[5,5] - 16 km",
        "L[6,6] - 32 km",
        "L[7,7] - 64 km",
    ]
    labels = [
        "observed counts",
        "expected counts",
        "$F_\\mathrm{clip}$",
        "expected counts",
        "$F_\\mathrm{clip}$",
    ]
    plt.close("all")
    selected = {model_id_1: spatial_stats[model_id_1], model_id_2: spatial_stats[model_id_2]}
    g = plot_spatial_statistics_paper(selected, groningen_contour, type_list, title=" ")
    g.fig.subplots_adjust(hspace=-0.01)
    g.fig.add_artist(mpl.lines.Line2D([0.08, 1], [0.77, 0.77], linewidth=0.5, color="k"))
    g.fig.add_artist(mpl.lines.Line2D([0.08, 1], [0.42, 0.42], linewidth=0.5, color="k"))
    for i, label in enumerate(g.row_labels):
        label.update({"text": labels[i], "ma": "center", "x": 1.03})
    for i, ax in enumerate(g.axs.flatten()[:8]):
        ax.set_title("")
        ax.annotate(xy=(0.5, 1.29), ha="center", text=level_list[i], clip_on=False, xycoords="axes fraction")
    axesf = g.axs.flatten()
    offset = 0.07
    tr = selected[model_id_1].thin({"level": 2}).sel({"likelihood_function": "multi_hom_poisson"})
    for i, ax in enumerate(axesf[16:24]):
        pos = ax.get_position()
        pos.y0 = pos.y0 + offset
        pos.y1 = pos.y1 + offset
        ax.set_position(pos)
        rax = g.fig.add_axes([pos.x0, pos.y0 - offset, (pos.x1 - pos.x0), (pos.y1 - pos.y0) * 0.5])
        mci.plot.make_test_axis(
            rax,
            tr.isel(level=i)["ll_observed"],
            tr.isel(level=i)["ll_fractiles"],
            tr.isel(level=i)["ll_test_result"],
            "",
            "",
            pval=tr.isel(level=i)["p_value"],
            ms=7,
            capsize=4,
        )
    tr = selected[model_id_2].thin({"level": 2}).sel({"likelihood_function": "multi_hom_poisson"})
    for i, ax in enumerate(axesf[-8:]):
        pos = ax.get_position()
        pos.y0 = pos.y0 + offset
        pos.y1 = pos.y1 + offset
        ax.set_position(pos)
        rax = g.fig.add_axes([pos.x0, pos.y0 - offset, (pos.x1 - pos.x0), (pos.y1 - pos.y0) * 0.5])
        mci.plot.make_test_axis(
            rax,
            tr.isel(level=i)["ll_observed"],
            tr.isel(level=i)["ll_fractiles"],
            tr.isel(level=i)["ll_test_result"],
            "",
            "",
            pval=tr.isel(level=i)["p_value"],
            ms=7,
            capsize=4,
        )
    if title is None:
        title = "multiresolution spatial tests"
    g.fig.text(x=0.5, y=0.98, s=title, va="center", ha="center", size=12)
    g.fig.text(x=0.5, y=0.955, s="observations", va="center", ha="center", size=12)
    g.fig.text(x=0.5, y=0.75, s=model_names[model_id_1], va="center", ha="center", size=12)
    g.fig.text(x=0.5, y=0.40, s=model_names[model_id_2], va="center", ha="center", size=12)
    cax = g.fig.add_axes([0.81, 0.958, 0.12, 0.012])
    g.add_colorbar(cax=cax, orientation="horizontal", ticks=[0, 0.5, 1], label="")
    g.fig.text(
        x=0.87,
        y=0.982,
        s="normalized count / $F_\\mathrm{clip}$",
        ha="center",
        va="center",
        size=9,
    )
    save_current_figure(figpath, base_name, figure_formats)


def generate_adaptive_analysis_plots_all_models(
    model_ids: list[str],
    base_name: str,
    adaptive_stats: dict[str, xr.Dataset],
    model_names: dict[str, str],
    groningen_contour: object,
    figpath: Path,
    figure_formats: list[str],
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    plt.close("all")
    ncols = 1 + len(model_ids)
    if figsize is None:
        figsize = (2.2 * ncols, 6.5)
    fig, axs = plt.subplots(2, ncols, figsize=figsize)
    ax_top = axs[0, :]
    ax_bot = axs[1, :]
    obs = adaptive_stats[model_ids[0]]["spatial_statistics"].sel(type="normalized_observation_density")
    mappable = obs.plot(x="x", y="y", ax=ax_top[0], add_colorbar=False)
    mappable.set_rasterized(True)
    ax_top[0].set_title("observed")
    mci.postprocess_ax(ax_top[0], groningen_contour)
    for j, mid in enumerate(model_ids, start=1):
        fd = adaptive_stats[mid]["spatial_statistics"].sel(type="normalized_forecast_density")
        forecast_map = fd.plot(x="x", y="y", ax=ax_top[j], add_colorbar=False)
        forecast_map.set_rasterized(True)
        ax_top[j].set_title(model_names[mid])
        mci.postprocess_ax(ax_top[j], groningen_contour)
        fc = adaptive_stats[mid]["spatial_statistics"].sel(type="cdf_clip")
        cdf_map = fc.plot(x="x", y="y", ax=ax_bot[j], add_colorbar=False)
        cdf_map.set_rasterized(True)
        ax_bot[j].set_title("")
        mci.postprocess_ax(ax_bot[j], groningen_contour)
    ax_bot[0].set_axis_off()
    cb = plt.colorbar(mappable, ax=ax_bot[0], orientation="horizontal")
    cb.set_label("normalized count density\nor $F_\\mathrm{clip}$", fontsize=8)
    pos_cb = ax_bot[0].get_position()
    cb.ax.set_position([
        pos_cb.x0 + 0.02,
        pos_cb.y0 + 0.25 * (pos_cb.y1 - pos_cb.y0),
        pos_cb.x1 - pos_cb.x0 - 0.04,
        0.15 * (pos_cb.y1 - pos_cb.y0),
    ])
    offset = 0.08
    for j, mid in enumerate(model_ids, start=1):
        tr = adaptive_stats[mid].sel({"likelihood_function": "multi_hom_poisson"})
        pos = ax_bot[j].get_position()
        ax_bot[j].set_position([pos.x0, pos.y0 + offset, pos.x1 - pos.x0, pos.y1 - pos.y0])
        rax = fig.add_axes([
            pos.x0,
            pos.y0 - offset * 0.95,
            pos.x1 - pos.x0,
            (pos.y1 - pos.y0) * 0.5,
        ])
        mci.plot.make_test_axis(
            rax,
            tr["ll_observed"],
            tr["ll_fractiles"],
            tr["ll_test_result"],
            "",
            "",
            pval=tr["p_value"],
            ms=7,
            capsize=4,
        )
    pos_col0 = ax_top[0].get_position()
    fig.add_artist(
        mpl.lines.Line2D([pos_col0.x1, pos_col0.x1], [0.04, 0.96], transform=fig.transFigure, color="k", linewidth=0.7)
    )
    ax_top[-1].annotate(
        "normalized count density",
        xy=(1.12, 0.5),
        xycoords="axes fraction",
        rotation=270,
        va="center",
        ha="center",
        fontsize=11,
        clip_on=False,
    )
    ax_bot[-1].annotate(
        "$F_\\mathrm{clip}$",
        xy=(1.12, 0.5),
        xycoords="axes fraction",
        rotation=270,
        va="center",
        ha="center",
        fontsize=11,
        clip_on=False,
    )
    if title is None:
        title = "adaptive-resolution spatial tests"
    fig.text(x=0.5, y=0.98, s=title, va="center", ha="center", size=12)
    save_current_figure(figpath, base_name, figure_formats)


def generate_bs_basis_plot(fault_data: xr.Dataset, figpath: Path, figure_formats: list[str]) -> None:
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.0, 1.0, 5))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    data1 = (
        fault_data.reset_index(["ID"])
        .sel(thickness_origin="faults")
        .set_coords("throw_thickness_ratio")
        .set_xindex("throw_thickness_ratio")["weights_bs"]
        .sortby("throw_thickness_ratio")
        .sel(bernstein_basis=fault_data["bernstein_degree"].load() == 4, drop=True)
        .drop_vars(["bernstein_basis", "bernstein_degree", "bernstein_index"])
        .assign_coords({"bernstein_basis": [0, 1, 2, 3, 4]})
        .drop_vars(["spatial_ref", "thickness_origin"])
    )
    data1 = data1.assign_coords({"ID": data1["ID"] / data1["ID"].max()})
    labels = [r"$B_{0,4}$", r"$B_{1,4}$", r"$B_{2,4}$", r"$B_{3,4}$", r"$B_{4,4}$"]
    for b, label, color in zip(data1["bernstein_basis"].values, labels, colors):
        data1.sel(bernstein_basis=b).plot.line(x="ID", ax=ax1, label=label, color=color)
    ax1.set_xlabel("$u$")
    ax1.set_title("$B_{k,4}(u)$")
    ax1.set_ylabel("weight")
    ax1.legend(framealpha=1.0)
    data2 = (
        fault_data.sel(thickness_origin="faults")
        .drop_vars("thickness_origin")
        .reset_index(["ID"])
        .set_coords("throw_thickness_ratio")
        .set_xindex("throw_thickness_ratio")["weights_bs"]
        .sortby("throw_thickness_ratio")
        .sel(bernstein_basis=fault_data["bernstein_degree"] == 4, drop=True)
        .drop_vars(["bernstein_basis", "bernstein_degree", "bernstein_index"])
        .assign_coords({"bernstein_basis": [0, 1, 2, 3, 4]})
        .drop_vars("spatial_ref")
    )
    labels = [r"$W_{0,4}$", r"$W_{1,4}$", r"$W_{2,4}$", r"$W_{3,4}$", r"$W_{4,4}$"]
    for b, label, color in zip(data2["bernstein_basis"].values, labels, colors):
        data2.sel(bernstein_basis=b).plot.line(
            x="throw_thickness_ratio",
            ax=ax2,
            label=label,
            color=color,
            xscale="log",
        )
    ax2.set_xlabel("$r=T/h$")
    ax2.set_title("$W_{k,4}(r)$")
    ax2.set_ylabel("weight")
    ax2.legend(framealpha=1.0)
    for ax, letter in zip([ax1, ax2], ["(a)", "(b)"]):
        ax.text(-0.08, 1.02, letter, transform=ax.transAxes, ha="right", va="bottom", fontweight="bold", fontsize=12, clip_on=False)
    plt.tight_layout()
    save_current_figure(figpath, "bs_basis", figure_formats)


def generate_bs_spatial_plot(
    grid_data: xr.Dataset,
    groningen_contour: object,
    figpath: Path,
    figure_formats: list[str],
    degree: int = 4,
) -> None:
    titles = {
        0: [r"$W_{0,0}$"],
        1: [r"$W_{0,1}$", r"$W_{1,1}$"],
        2: [r"$W_{0,2}$", r"$W_{1,2}$", r"$W_{2,2}$"],
        4: [r"$W_{0,4}$", r"$W_{1,4}$", r"$W_{2,4}$", r"$W_{3,4}$", r"$W_{4,4}$"],
    }
    subset = (
        grid_data["bernstein_fraction"]
        .rename("Bernstein weight")
        .sel(bernstein_basis=grid_data["bernstein_degree"] == degree)
        .sel(thickness_origin="grid")
        .reset_index("bernstein_basis")
        .reset_coords(drop=True)
        .unstack("loc")
        .dropna("x", how="all")
        .dropna("y", how="all")
        .sortby("x")
        .sortby("y")
    )
    facets = subset.T.plot(col="bernstein_basis", aspect=0.7, cmap="gray_r")
    facets = rasterize_primary_facet_collections(facets)
    facets = mci.postprocess_facets(facets, groningen_contour, edgecolor="black", facecolor="none", linewidth=1)
    for ax, title in zip(facets.axs.flat, titles[degree]):
        ax.set_title(title)
    save_current_figure(figpath, f"bs_spatial_{degree}", figure_formats)


def generate_groningen_three_panel_plot(
    event_data: xr.Dataset,
    fault_data: xr.Dataset,
    grid_data: xr.Dataset,
    groningen_contour: object,
    figpath: Path,
    figure_formats: list[str],
) -> None:
    plt.close("all")
    scale = 1 / 1000.0
    axis_label_fs = 13
    tick_fs = 12
    cbar_tick_fs = 11
    cbar_label_fs = 14
    grid_fields = grid_data[["reservoir_thickness", "pressure_drop", "compaction_coefficient"]].unstack("loc")
    thickness_field = grid_fields.get("reservoir_thickness", None)
    press_field = grid_fields.get("pressure_drop", None)
    comp_field = grid_fields.get("compaction_coefficient", None)
    press_date = "2025-10-01"
    if press_field is not None and "datetime" in press_field.dims:
        press_field = press_field.sel(datetime=np.datetime64(press_date), method="nearest")
    ev_df = (
        event_data[["x", "y", "magnitude"]]
        .to_dataframe()
        .reset_index()
        .dropna(subset=["x", "y", "magnitude"])
    )
    fault_df = pd.DataFrame(
        {
            "FAULT_ID": fault_data["FAULT_ID"].values,
            "PILLAR_ID": fault_data["PILLAR_ID"].values,
            "x": fault_data["x"].values,
            "y": fault_data["y"].values,
        }
    ).dropna(subset=["x", "y"])
    fault_groups = []
    for fault_id, group in fault_df.groupby("FAULT_ID"):
        fault_groups.append((fault_id, group.sort_values("PILLAR_ID")[["x", "y"]].values))
    extents = []
    for data_array in [press_field, thickness_field, comp_field]:
        if data_array is not None:
            x_vals = data_array["x"].values
            y_vals = data_array["y"].values
            extents.append((x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()))
    if extents:
        minx = min(extent[0] for extent in extents)
        maxx = max(extent[1] for extent in extents)
        miny = min(extent[2] for extent in extents)
        maxy = max(extent[3] for extent in extents)
    else:
        minx, miny, maxx, maxy = groningen_contour.bounds
    dx = (maxx - minx) * scale
    dy = (maxy - miny) * scale
    pad_frac = 0.02
    xlim = (minx * scale - dx * pad_frac, maxx * scale + dx * pad_frac)
    ylim = (miny * scale - dy * pad_frac, maxy * scale + dy * pad_frac)
    legend_buffer_frac = 0.12
    xlim = (xlim[0], xlim[1] + dx * legend_buffer_frac)
    ylim = (ylim[0], ylim[1] + dy * legend_buffer_frac)
    fig = plt.figure(figsize=(18, 9), constrained_layout=True)
    cbar_height_ratio = 0.1 / 3.0
    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[1.0, cbar_height_ratio], hspace=0.0, wspace=0.06)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    caxes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.0)
    titles = [
        f"pressure drop ({press_date}) [$\\mathrm{{MPa}}$]",
        "reservoir thickness [$\\mathrm{m}$]",
        "compaction coefficient [$\\mathrm{MPa^{-1}}$]",
    ]
    fields = [press_field, thickness_field, comp_field]
    mag_bin_defs = []
    if len(ev_df):
        mag_bin_defs = [
            (-np.inf, 1.5, "<1.5", 12.0),
            (1.5, 2.5, "1.5-2.5", 28.0),
            (2.5, np.inf, ">=2.5", 44.0),
        ]
    legend_params = {
        "loc": "upper right",
        "fontsize": 11,
        "title": "Overlays",
        "frameon": True,
        "framealpha": 1.0,
        "borderpad": 0.6,
        "labelspacing": 0.6,
        "handlelength": 1.4,
        "handletextpad": 0.8,
    }
    imgs: dict[object, tuple[object, object]] = {}
    for ax, field, title in zip(axs, fields, titles):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.margins(x=0.0, y=0.0)
        if field is not None:
            x_vals = field["x"].values
            y_vals = field["y"].values
            values = field.values
            if x_vals.ndim == 1 and y_vals.ndim == 1:
                xx, yy = np.meshgrid(x_vals * scale, y_vals * scale)
                img = ax.pcolormesh(xx, yy, values.T, cmap="viridis", shading="auto", rasterized=True)
            else:
                img = ax.pcolormesh(x_vals * scale, y_vals * scale, values, cmap="viridis", shading="auto", rasterized=True)
            imgs[ax] = (img, field)
        else:
            ax.text(0.5, 0.5, "field not found", transform=ax.transAxes, ha="center", va="center")
        fault_proxy = Line2D([0], [0], color="black", linewidth=0.6, label="Fault")
        boundary_proxy = Line2D([0], [0], color="black", linewidth=2.0, label="Field boundary")
        quake_proxies = []
        for low, high, label, size in mag_bin_defs:
            quake_proxies.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markersize=np.sqrt(size) * 0.85,
                    markerfacecolor="red",
                    markeredgecolor="k",
                    markeredgewidth=0.25,
                    label=f"M {label}",
                )
            )
        if ax is axs[0] or ax is axs[1]:
            for low, high, _, size in mag_bin_defs:
                mask = (ev_df["magnitude"] >= (low if low != -np.inf else -1e9)) & (
                    ev_df["magnitude"] < (high if high != np.inf else 1e9)
                )
                if mask.any():
                    ax.scatter(
                        ev_df.loc[mask, "x"] * scale,
                        ev_df.loc[mask, "y"] * scale,
                        s=size,
                        c="red",
                        edgecolor="k",
                        linewidth=0.25,
                        zorder=4,
                    )
        if ax is axs[0]:
            for _, coords in fault_groups:
                ax.plot(coords[:, 0] * scale, coords[:, 1] * scale, color="black", linewidth=0.6, zorder=3)
            legend = ax.legend(handles=[fault_proxy] + quake_proxies + [boundary_proxy], **legend_params)
            legend.get_title().set_fontsize(12)
        elif ax is axs[1]:
            legend = ax.legend(handles=quake_proxies + [boundary_proxy], **legend_params)
            legend.get_title().set_fontsize(12)
        elif ax is axs[2]:
            for _, coords in fault_groups:
                ax.plot(coords[:, 0] * scale, coords[:, 1] * scale, color="black", linewidth=0.6, zorder=3)
            legend = ax.legend(handles=[fault_proxy, boundary_proxy], **legend_params)
            legend.get_title().set_fontsize(12)
        xpoly, ypoly = groningen_contour.exterior.xy
        ax.plot(np.asarray(xpoly) * scale, np.asarray(ypoly) * scale, color="black", linewidth=2.0, zorder=6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [km]", fontsize=axis_label_fs)
        ax.set_ylabel("y [km]", fontsize=axis_label_fs)
        ax.tick_params(labelsize=tick_fs)
    for i, ax in enumerate(axs):
        if ax in imgs:
            img, _ = imgs[ax]
            cb = fig.colorbar(img, cax=caxes[i], orientation="horizontal")
            cb.ax.tick_params(labelsize=cbar_tick_fs)
            cb.set_label(titles[i], fontsize=cbar_label_fs)
        else:
            caxes[i].set_visible(False)
    for ax, letter in zip(axs, ["(a)", "(b)", "(c)"]):
        ax.text(-0.06, 1.02, letter, transform=ax.transAxes, ha="right", va="bottom", fontweight="bold", fontsize=14, clip_on=False)
    save_current_figure(figpath, "groningen_three_panel", figure_formats)


def generate_cell_covering_plot(
    cell_covering: xr.DataArray,
    adaptive_stats: dict[str, dict[str, xr.Dataset]],
    model_ids: list[str],
    groningen_contour: object,
    figpath: Path,
    figure_formats: list[str],
) -> None:
    g = (
        cell_covering.where(cell_covering > 0)
        .rename(cell="bin")
        .plot(
            x="x",
            y="y",
            cmap="grey_r",
            add_colorbar=False,
            col="bin",
            col_wrap=7,
            aspect="equal",
            figsize=(10, 8),
        )
    )
    g = rasterize_primary_facet_collections(g)
    mci.postprocess_facets(g, groningen_contour, edgecolor="black", facecolor="none", linewidth=1)
    plt.tight_layout()
    obs_counts = adaptive_stats["retrospective"][model_ids[0]]["observations"].sortby("cell")
    for ax, count in zip(g.axs.flatten(), obs_counts.values):
        ax.set_title(f"n={int(count)}")
    save_current_figure(figpath, "cell_covering", figure_formats)


def generate_loo_plot(
    rc: dict[str, dict[str, az.InferenceData]],
    model_names: dict[str, str],
    figpath: Path,
    figure_formats: list[str],
    artifact_dir: Path,
) -> None:
    rc_local = {model_names[key]: value for key, value in rc["retrospective"].items()}
    azc_tot = az.compare(rc_local, ic="loo", var_name="XT")
    azc_tot.to_csv(artifact_dir / "loo_compare_retrospective.csv")
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
    az.plot_compare(azc_tot, insample_dev=True, plot_ic_diff=True, title="", ax=ax)
    ax.set_ylabel("Ranked models")
    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    weights = pd.Series(azc_tot["weight"].values, index=azc_tot.index).to_dict()
    for y, name in zip(ax.get_yticks(), labels):
        if name in weights:
            ax.text(
                0.02,
                y + 0.04,
                f"w= {weights[name]:.2f}",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
            )
    plt.tight_layout()
    save_current_figure(figpath, "loo-cv", figure_formats)


__all__ = [
    "generate_adaptive_analysis_plots_all_models",
    "generate_bs_basis_plot",
    "generate_bs_spatial_plot",
    "generate_cell_covering_plot",
    "generate_groningen_three_panel_plot",
    "generate_loo_plot",
    "generate_model_time_series_plot",
    "generate_spatial_performance_plots",
    "has_perspective_models",
    "log_skipped_figure",
    "maybe_render_formats",
    "normalize_figure_formats",
    "plot_likelihood_tests_2x4",
]