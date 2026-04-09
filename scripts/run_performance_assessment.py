from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import arviz as az
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
CHAINTOOLS_ROOT = SRC_ROOT / "chaintools"
for path in [SRC_ROOT, CHAINTOOLS_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import chaintools.tools_grid as tgrid  # noqa: E402
import model_chain_inference as mci  # noqa: E402
from workflow_support.logging import configure_logging, get_logger, progress  # noqa: E402


DEFAULT_MODEL_NAMES = {
    "ETS_rmax_etasKac": "ETS-ETF",
    "EVTS_rmax_etasKac": "EVTS-ETF",
    "ETS_bs4_etasKac": "ETS-BS4-ETF",
    "EVTS_bs4_etasKac": "EVTS-BS4-ETF",
}

DEFAULT_FIGURE_FORMATS = [".pdf", ".png", ".eps"]
PDF_RASTER_DPI = 300
REQUIRED_FILTER_ATTRS = {"timeframe", "timeframe_etas", "polygon", "polygon_etas", "mmin", "mmin_etas"}
LOGGER = get_logger(__name__)


def build_testsuite_and_assessment(
    model_id: str,
    idata: az.InferenceData,
    model_specs: dict[str, dict[str, object]],
    grid_data: xr.Dataset,
    event_data: xr.Dataset,
    filterset_used: xr.Dataset,
    cell_covering: xr.DataArray,
    pa_sample_size: int,
    seed: int,
) -> tuple[str, xr.DataTree, xr.Dataset, xr.Dataset, xr.Dataset]:
    testsuite = build_testsuite_artifact(
        model_id,
        idata,
        model_specs,
        grid_data,
        event_data,
        filterset_used,
    )
    temporal, spatial, adaptive = build_assessment_from_testsuite(
        testsuite,
        cell_covering,
        pa_sample_size,
        seed,
    )
    return model_id, testsuite, temporal, spatial, adaptive


def build_testsuite_artifact(
    model_id: str,
    idata: az.InferenceData,
    model_specs: dict[str, dict[str, object]],
    grid_data: xr.Dataset,
    event_data: xr.Dataset,
    filterset_used: xr.Dataset,
) -> xr.DataTree:
    specs = model_specs[model_id]
    return mci.prepare_testsuite(
        idata,
        grid_data.sel(specs.get("sel", {})),
        event_data,
        filterset_used,
    )


def build_assessment_from_testsuite(
    testsuite: xr.DataTree,
    cell_covering: xr.DataArray,
    pa_sample_size: int,
    seed: int,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    return mci.perf_assessment(
        testsuite,
        cell_covering,
        sample_size=pa_sample_size,
        rng=seed,
    )


def build_and_write_assessment_artifact(
    model_id: str,
    idata_path: Path,
    source_data_root: Path,
    cell_covering_path: Path,
    artifact_dir: Path,
    experiment: str,
    perspective: str,
    model_specs: dict[str, dict[str, object]],
    filter_attrs: dict[str, object],
    pa_sample_size: int,
    seed: int,
) -> str:
    idata = az.from_netcdf(idata_path).load()
    _, event_data, _, grid_data, _ = load_context(source_data_root)
    cell_covering = load_cell_covering(cell_covering_path)
    filterset_used = filter_from_attrs(filter_attrs).sel(purpose="calibration")
    _, testsuite, temporal, spatial, adaptive = build_testsuite_and_assessment(
        model_id,
        idata,
        model_specs,
        grid_data,
        event_data,
        filterset_used,
        cell_covering,
        pa_sample_size,
        seed,
    )
    save_testsuite(artifact_dir, experiment, perspective, model_id, testsuite, True)
    save_assessment(artifact_dir, experiment, perspective, model_id, temporal, spatial, adaptive, seed, True)
    return model_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the performance assessment artifact generation from YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "performance_assessment.yaml",
        help="YAML configuration file.",
    )
    parser.add_argument(
        "--cache",
        choices=["error", "reuse", "refresh"],
        default="error",
        help="How to handle existing outputs: error, reuse them, or refresh them.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(repo_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def get_cache_mode(args: argparse.Namespace) -> str:
    return "refresh" if args.force else args.cache


def should_write_output(path: Path, cache_mode: str) -> bool:
    if not path.exists():
        return True
    if cache_mode == "refresh":
        return True
    if cache_mode == "reuse":
        LOGGER.info("cached %s", path)
        return False
    raise FileExistsError(
        f"Refusing to overwrite existing output: {path}. Use --force or --cache refresh to replace it, or --cache reuse to keep it."
    )


def get_idata_path(path: Path, experiment: str, perspective: str, model: str) -> Path:
    return path / f"model_calibration-{experiment}-{perspective}-{model}.h5"


def get_testsuite_path(path: Path, experiment: str, perspective: str, model: str) -> Path:
    return path / f"testsuite-{experiment}-{perspective}-{model}.h5"


def get_assessment_path(path: Path, experiment: str, perspective: str, model: str) -> Path:
    return path / f"performance_assessment-{experiment}-{perspective}-{model}.h5"


def get_cell_covering_path(path: Path, experiment: str) -> Path:
    return path / f"cell_covering-{experiment}.h5"


def normalize_cell_covering(cell_covering: xr.DataArray) -> xr.DataArray:
    index_dims = [dim for dim in ("X", "Y") if dim in cell_covering.dims and dim in cell_covering.indexes]
    if not index_dims:
        return cell_covering
    return cell_covering.reset_index(index_dims)


def load_cell_covering(path: Path) -> xr.DataArray:
    try:
        return normalize_cell_covering(xr.open_dataarray(path, engine="h5netcdf").load())
    except ValueError as exc:
        dataset = xr.open_dataset(path, engine="h5netcdf")
        try:
            dataset.load()
            if len(dataset.data_vars) != 1:
                raise ValueError(
                    f"Expected exactly one cell_covering data variable in {path}, found {len(dataset.data_vars)}"
                ) from exc
            variable_name = next(iter(dataset.data_vars))
            return normalize_cell_covering(dataset[variable_name])
        finally:
            dataset.close()


def normalize_serializable_dataset(dataset: xr.Dataset) -> xr.Dataset:
    index_dims = [
        name
        for name, index in dataset.indexes.items()
        if name in dataset.dims and isinstance(index, pd.MultiIndex)
    ]
    if not index_dims:
        return dataset
    return dataset.reset_index(index_dims)


def normalize_serializable_datatree(tree: xr.DataTree) -> xr.DataTree:
    return xr.DataTree.from_dict(
        {
            key: normalize_serializable_dataset(node.to_dataset(inherit=False))
            for key, node in tree.subtree_with_keys
        }
    )


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


def filter_from_attrs(attrs: dict[str, object]) -> xr.Dataset:
    return xr.Dataset(
        {
            "timeframe": xr.DataArray(
                data=[attrs["timeframe"], attrs["timeframe_etas"]],
                dims=["purpose", "epoch"],
                coords={
                    "purpose": ["calibration", "etas"],
                    "epoch": ["start", "finish"],
                },
            ),
            "polygon": xr.DataArray(
                data=[attrs["polygon"], attrs["polygon_etas"]],
                dims=["purpose"],
            ),
            "mmin": xr.DataArray(
                data=[attrs["mmin"], attrs["mmin_etas"]],
                dims=["purpose"],
            ),
        }
    )


def load_context(source_data_root: Path) -> tuple[object, xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    groningen_contour = (
        gpd.read_file(source_data_root / "groningen_polygons.shp")
        .set_index("polygon")
        .geometry["GroningenFieldGWC"]
    )
    event_data = xr.open_dataset(source_data_root / "event_data.h5", decode_coords="all").load()
    fault_data = (
        xr.open_dataset(
            source_data_root / "fault_data.h5",
            decode_coords="all",
            decode_timedelta=False,
        )
        .set_xindex(["bernstein_degree", "bernstein_index"])
        .set_xindex(["FAULT_ID", "PILLAR_ID"])
    )
    grid_data = (
        xr.open_dataset(
            source_data_root / "grid_data.h5",
            decode_coords="all",
            decode_timedelta=False,
        )
        .set_xindex(["x", "y"])
        .set_xindex(["bernstein_degree", "bernstein_index"])
    )
    grid_specs = grid_data[["x", "y", "datetime"]].unstack("loc")
    event_data = event_data.merge(
        tgrid.prepare_grid_selection(event_data, grid_specs, time_conversion_factor=None)
    )
    return groningen_contour, event_data, fault_data, grid_data, grid_specs


def ensure_required_calibration_attrs(
    idata: az.InferenceData,
    perspective: str,
    model_id: str,
) -> az.InferenceData:
    attrs = dict(idata.attrs)
    missing = REQUIRED_FILTER_ATTRS.difference(attrs)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(
            f"Calibration artifact for {perspective}/{model_id} is missing required attrs: {missing_text}"
        )
    return idata


def collect_calibrations(
    calibration_root: Path,
    experiment: str,
    perspectives: list[str],
    model_ids: list[str],
) -> dict[str, dict[str, az.InferenceData]]:
    rc: dict[str, dict[str, az.InferenceData]] = {}
    for perspective in perspectives:
        rc[perspective] = {}
        for model_id in model_ids:
            path = get_idata_path(calibration_root, experiment, perspective, model_id)
            if not path.exists():
                raise FileNotFoundError(f"Missing calibration artifact: {path}")
            idata = az.from_netcdf(path).load()
            rc[perspective][model_id] = ensure_required_calibration_attrs(
                idata,
                perspective,
                model_id,
            )
    return rc


def load_testsuite(path: Path) -> xr.DataTree:
    tree = xr.open_datatree(path, engine="h5netcdf")
    tree.load()
    tree.close()
    return tree


def load_assessment(path: Path) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    tree = xr.open_datatree(path, engine="h5netcdf")
    tree.load()
    temporal = tree["temporal"].to_dataset()
    spatial = tree["spatial"].to_dataset()
    adaptive = tree["adaptive"].to_dataset()
    tree.close()
    return temporal, spatial, adaptive


def save_testsuite(
    outdir: Path,
    experiment: str,
    perspective: str,
    model_id: str,
    testsuite: xr.DataTree,
    force: bool,
) -> None:
    path = outdir / f"testsuite-{experiment}-{perspective}-{model_id}.h5"
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}. Use --force to replace it.")
    normalize_serializable_datatree(testsuite).to_netcdf(path, engine="h5netcdf")


def save_assessment(
    outdir: Path,
    experiment: str,
    perspective: str,
    model_id: str,
    temporal: xr.Dataset,
    spatial: xr.Dataset,
    adaptive: xr.Dataset,
    seed: int,
    force: bool,
) -> None:
    path = outdir / f"performance_assessment-{experiment}-{perspective}-{model_id}.h5"
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}. Use --force to replace it.")
    temporal = temporal.assign_attrs({**temporal.attrs, "assessment_seed": seed})
    spatial = spatial.assign_attrs({**spatial.attrs, "assessment_seed": seed})
    adaptive = adaptive.assign_attrs({**adaptive.attrs, "assessment_seed": seed})
    normalize_serializable_datatree(
        xr.DataTree.from_dict(
            {
                "temporal": temporal,
                "spatial": spatial,
                "adaptive": adaptive,
            }
        )
    ).to_netcdf(path, engine="h5netcdf")


def maybe_render_figure(path: Path, cache_mode: str, render: Callable[[], None]) -> None:
    if not should_write_output(path, cache_mode):
        return
    render()


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
    perf_stats: dict[str, xr.Dataset], polygon: object, type_list: list[str], title: str | None = None
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


def main() -> None:
    args = parse_args()
    configure_logging()
    config = yaml.safe_load(args.config.read_text())
    cache_mode = get_cache_mode(args)

    repo_root = resolve_path(REPO_ROOT, config.get("repo_root")) or REPO_ROOT
    source_data_root = resolve_path(repo_root, config.get("source_data_root")) or (repo_root / "data" / "resources")
    calibration_root = resolve_path(repo_root, config.get("calibration_root")) or (repo_root / "data" / "generated_calibrations")
    artifact_dir = resolve_path(repo_root, config.get("artifact_dir")) or (repo_root / "data" / "generated_assessment")
    experiment = config.get("experiment", "groningen_1995_2025")
    perspectives = config.get("perspectives", ["prospective", "retrospective"])
    model_names = config.get("model_names", DEFAULT_MODEL_NAMES)
    model_ids = config.get("models", list(model_names.keys()))
    pa_sample_size = int(config.get("pa_sample_size", 10_000))
    seed = int(config.get("seed", 42))
    workers = max(1, int(config.get("workers", 1)))
    timeframe_testing = config.get(
        "timeframe_testing",
        config.get("timeframe_prospective_testing", ["2020-10-01", "2025-10-01"]),
    )
    timeframe_forecast = config.get("timeframe_forecast")

    LOGGER.info(
        "stage=assessment configured experiment=%s perspectives=%s models=%s workers=%s cache=%s calibration_root=%s artifact_dir=%s",
        experiment,
        perspectives,
        model_ids,
        workers,
        cache_mode,
        calibration_root,
        artifact_dir,
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_specs = yaml.safe_load((source_data_root / f"model_specs-{experiment}.yaml").read_text())
    _, event_data, _, grid_data, grid_specs = load_context(source_data_root)
    rc = collect_calibrations(
        calibration_root,
        experiment,
        perspectives,
        model_ids,
    )

    forecast_source = "retrospective" if "retrospective" in rc else "prospective"
    forecast_filter_attrs = dict(rc[forecast_source][model_ids[0]].attrs)
    retrospective_filter_attrs = dict(forecast_filter_attrs)
    if timeframe_forecast is not None:
        retrospective_filter_attrs["timeframe"] = timeframe_forecast
    filterset_retrospective = filter_from_attrs(retrospective_filter_attrs).sel(purpose="calibration")
    prospective_filter_attrs = dict(forecast_filter_attrs)
    prospective_filter_attrs["timeframe"] = timeframe_testing
    filterset_prospective = filter_from_attrs(prospective_filter_attrs).sel(purpose="calibration")
    LOGGER.info(
        "stage=assessment status=filters retrospective_timeframe=%s prospective_timeframe=%s",
        retrospective_filter_attrs["timeframe"],
        prospective_filter_attrs["timeframe"],
    )

    cell_covering_path = get_cell_covering_path(artifact_dir, experiment)
    cell_covering: xr.DataArray | None = None
    if cell_covering_path.exists() and cache_mode == "reuse":
        try:
            cell_covering = load_cell_covering(cell_covering_path)
            LOGGER.info("cached %s", cell_covering_path)
        except (OSError, ValueError, KeyError) as exc:
            LOGGER.warning(
                "stage=assessment status=cache-invalid output=%s reason=%s",
                cell_covering_path,
                exc,
            )
    if cell_covering is None:
        if cell_covering_path.exists() and cache_mode == "error":
            raise FileExistsError(
                f"Refusing to overwrite existing output: {cell_covering_path}. Use --force or --cache refresh to replace it, or --cache reuse to keep it."
            )
        cell_covering = normalize_cell_covering(
            mci.generate_cell_covering(
                grid_specs,
                event_data,
                filterset_retrospective,
                ms_mode="rectangular",
                threshold=10,
            )
        )
        cell_covering.to_netcdf(cell_covering_path, engine="h5netcdf")

    testsuite_dict: dict[str, dict[str, xr.DataTree]] = {}
    temporal_results: dict[str, dict[str, xr.Dataset]] = {}
    spatial_results: dict[str, dict[str, xr.Dataset]] = {}
    adaptive_results: dict[str, dict[str, xr.Dataset]] = {}

    total_perspectives = len(rc)
    for perspective_index, perspective in enumerate(list(rc.keys()), start=1):
        LOGGER.info(
            "stage=assessment perspective=%s status=start index=%s/%s",
            perspective,
            perspective_index,
            total_perspectives,
        )
        idatas = rc[perspective]
        filterset_used = filterset_prospective if perspective == "prospective" else filterset_retrospective
        filter_attrs_used = prospective_filter_attrs if perspective == "prospective" else retrospective_filter_attrs
        testsuite_dict[perspective] = {}
        temporal_results[perspective] = {}
        spatial_results[perspective] = {}
        adaptive_results[perspective] = {}
        pending_models: list[tuple[str, az.InferenceData, bool, bool]] = []
        cached_models: list[str] = []
        partial_models: list[str] = []
        for model_id in idatas.keys():
            idata = idatas[model_id]
            testsuite_path = get_testsuite_path(artifact_dir, experiment, perspective, model_id)
            assessment_path = get_assessment_path(artifact_dir, experiment, perspective, model_id)

            reusable_artifacts = cache_mode == "reuse"
            cached_testsuite = testsuite_path.exists() and reusable_artifacts
            cached_assessment = assessment_path.exists() and reusable_artifacts

            if cached_testsuite:
                LOGGER.info("cached %s", testsuite_path)
                testsuite_dict[perspective][model_id] = load_testsuite(testsuite_path)
            elif testsuite_path.exists() and cache_mode == "error":
                raise FileExistsError(
                    f"Refusing to overwrite existing output: {testsuite_path}. Use --force or --cache refresh to replace it, or --cache reuse to keep it."
                )

            if cached_assessment:
                LOGGER.info("cached %s", assessment_path)
                temporal, spatial, adaptive = load_assessment(assessment_path)
                temporal_results[perspective][model_id] = temporal
                spatial_results[perspective][model_id] = spatial
                adaptive_results[perspective][model_id] = adaptive
            elif assessment_path.exists() and cache_mode == "error":
                raise FileExistsError(
                    f"Refusing to overwrite existing output: {assessment_path}. Use --force or --cache refresh to replace it, or --cache reuse to keep it."
                )

            if not cached_testsuite or not cached_assessment:
                pending_models.append((model_id, idata, not cached_testsuite, not cached_assessment))
            if cached_testsuite != cached_assessment:
                partial_models.append(model_id)
            if cached_testsuite and cached_assessment:
                cached_models.append(model_id)
                LOGGER.info("stage=assessment perspective=%s model=%s status=cached", perspective, model_id)

        if not pending_models:
            LOGGER.info(
                "stage=assessment perspective=%s status=cached models=%s",
                perspective,
                sorted(cached_models),
            )
            continue

        effective_workers = workers
        if workers > 1 and partial_models:
            effective_workers = 1
            LOGGER.info(
                "stage=assessment perspective=%s status=sequential reason=partial-cache-reuse models=%s",
                perspective,
                sorted(set(partial_models)),
            )

        if effective_workers == 1:
            total_models = len(pending_models)
            for model_index, (model_id, idata, need_testsuite, need_assessment) in enumerate(pending_models, start=1):
                LOGGER.info(
                    "stage=assessment perspective=%s model=%s status=build index=%s/%s need_testsuite=%s need_assessment=%s",
                    perspective,
                    model_id,
                    model_index,
                    total_models,
                    need_testsuite,
                    need_assessment,
                )
                testsuite = testsuite_dict[perspective].get(model_id)
                if need_testsuite:
                    testsuite = build_testsuite_artifact(
                        model_id,
                        idata,
                        model_specs,
                        grid_data,
                        event_data,
                        filterset_used,
                    )
                    save_testsuite(artifact_dir, experiment, perspective, model_id, testsuite, True)
                if testsuite is not None:
                    testsuite_dict[perspective][model_id] = testsuite
                if need_assessment:
                    if testsuite is None:
                        raise RuntimeError(
                            f"Missing testsuite for {perspective}/{model_id} while building assessment"
                        )
                    if cell_covering is None:
                        raise RuntimeError("cell_covering is required to compute assessment artifacts")
                    temporal, spatial, adaptive = build_assessment_from_testsuite(
                        testsuite,
                        cell_covering,
                        pa_sample_size,
                        seed,
                    )
                    save_assessment(
                        artifact_dir,
                        experiment,
                        perspective,
                        model_id,
                        temporal,
                        spatial,
                        adaptive,
                        seed,
                        True,
                    )
                    temporal_results[perspective][model_id] = temporal
                    spatial_results[perspective][model_id] = spatial
                    adaptive_results[perspective][model_id] = adaptive
                testsuite_dict[perspective][model_id] = testsuite
                LOGGER.info("stage=assessment perspective=%s model=%s status=wrote", perspective, model_id)
        else:
            futures = {}
            max_workers = min(effective_workers, len(pending_models))
            LOGGER.info(
                "stage=assessment perspective=%s status=parallel workers=%s models=%s",
                perspective,
                max_workers,
                sorted(model_id for model_id, _, _, _ in pending_models),
            )
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for model_id, _, _, _ in pending_models:
                    futures[
                        executor.submit(
                            build_and_write_assessment_artifact,
                            model_id,
                            get_idata_path(calibration_root, experiment, perspective, model_id),
                            source_data_root,
                            cell_covering_path,
                            artifact_dir,
                            experiment,
                            perspective,
                            model_specs,
                            filter_attrs_used,
                            pa_sample_size,
                            seed,
                        )
                    ] = model_id
                for future in progress(
                    as_completed(futures),
                    desc=f"{perspective} assessment",
                    total=len(futures),
                ):
                    model_id = future.result()
                    testsuite_dict[perspective][model_id] = load_testsuite(
                        get_testsuite_path(artifact_dir, experiment, perspective, model_id)
                    )
                    temporal, spatial, adaptive = load_assessment(
                        get_assessment_path(artifact_dir, experiment, perspective, model_id)
                    )
                    temporal_results[perspective][model_id] = temporal
                    spatial_results[perspective][model_id] = spatial
                    adaptive_results[perspective][model_id] = adaptive
                    LOGGER.info("stage=assessment perspective=%s model=%s status=wrote", perspective, model_id)
        LOGGER.info(
            "stage=assessment perspective=%s status=done computed=%s cached=%s",
            perspective,
            sorted(model_id for model_id, _, _, _ in pending_models),
            sorted(cached_models),
        )

    LOGGER.info("assessment artifacts: %s", artifact_dir)


if __name__ == "__main__":
    main()