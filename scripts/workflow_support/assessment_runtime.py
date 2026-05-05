from __future__ import annotations

from pathlib import Path
import sys

import arviz as az
import geopandas as gpd
import xarray as xr

from workflow_support.paths import REPO_ROOT, get_idata_path


def _import_local_runtime_dependencies() -> tuple[object, object]:
    try:
        import chaintools.tools_grid as tools_grid
        import model_chain_inference as package
    except ModuleNotFoundError:
        src_root = REPO_ROOT / "src"
        chaintools_root = src_root / "chaintools"
        for path in [src_root, chaintools_root]:
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
        import chaintools.tools_grid as tools_grid
        import model_chain_inference as package
    return tools_grid, package


tgrid, mci = _import_local_runtime_dependencies()

DEFAULT_MODEL_NAMES = {
    "ETS_rmax_etasKac": "ETS-ETF",
    "EVTS_rmax_etasKac": "EVTS-ETF",
    "ETS_bs4_etasKac": "ETS-BS4-ETF",
    "EVTS_bs4_etasKac": "EVTS-BS4-ETF",
}
REQUIRED_FILTER_ATTRS = {"timeframe", "timeframe_etas", "polygon", "polygon_etas", "mmin", "mmin_etas"}


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


__all__ = [
    "DEFAULT_MODEL_NAMES",
    "build_assessment_from_testsuite",
    "build_testsuite_artifact",
    "collect_calibrations",
    "filter_from_attrs",
    "load_assessment",
    "load_cell_covering",
    "load_context",
    "load_testsuite",
    "normalize_cell_covering",
]
