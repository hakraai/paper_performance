from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import arviz as az
import geopandas as gpd
import pandas as pd
import xarray as xr
import yaml

import chaintools.tools_grid as tgrid

from .testsuite import prepare_testsuite


DEFAULT_MODEL_NAMES = {
    "ETS_rmax_etasKac": "ETS-ETF",
    "EVTS_rmax_etasKac": "EVTS-ETF",
    "ETS_bs4_etasKac": "ETS-BS4-ETF",
    "EVTS_bs4_etasKac": "EVTS-BS4-ETF",
}

REQUIRED_FILTER_ATTRS = {
    "timeframe",
    "timeframe_etas",
    "polygon",
    "polygon_etas",
    "mmin",
    "mmin_etas",
}


@dataclass(frozen=True)
class InvestigationPaths:
    repo_root: Path
    source_data_root: Path
    calibration_root: Path
    artifact_root: Path
    figure_root: Path


def infer_repo_root(start: Path | None = None) -> Path:
    candidate = (start or Path.cwd()).resolve()
    for root in [candidate, *candidate.parents]:
        if (root / "src" / "model_chain_inference").exists() and (root / "configs").exists():
            return root
    raise FileNotFoundError(f"Could not infer repository root from {candidate}")


def default_paths(repo_root: Path | None = None) -> InvestigationPaths:
    root = infer_repo_root(repo_root)
    return InvestigationPaths(
        repo_root=root,
        source_data_root=root / "data" / "resources",
        calibration_root=root / "data" / "generated_calibrations",
        artifact_root=root / "data" / "generated_assessment",
        figure_root=root / "figures" / "generated_paper",
    )


def load_model_specs(source_data_root: Path, experiment: str) -> dict[str, dict[str, object]]:
    specs_path = source_data_root / f"model_specs-{experiment}.yaml"
    return yaml.safe_load(specs_path.read_text())


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


def get_calibration_path(
    calibration_root: Path,
    experiment: str,
    perspective: str,
    model_id: str,
) -> Path:
    return calibration_root / f"model_calibration-{experiment}-{perspective}-{model_id}.h5"


def load_calibration(
    calibration_root: Path,
    experiment: str,
    perspective: str,
    model_id: str,
) -> az.InferenceData:
    path = get_calibration_path(calibration_root, experiment, perspective, model_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing calibration artifact: {path}")
    return az.from_netcdf(path).load()


def load_calibrations(
    calibration_root: Path,
    experiment: str,
    perspectives: list[str],
    model_ids: list[str],
) -> dict[str, dict[str, az.InferenceData]]:
    calibrations: dict[str, dict[str, az.InferenceData]] = {}
    for perspective in perspectives:
        calibrations[perspective] = {}
        for model_id in model_ids:
            calibrations[perspective][model_id] = load_calibration(
                calibration_root,
                experiment,
                perspective,
                model_id,
            )
    return calibrations


def get_assessment_path(
    artifact_root: Path,
    experiment: str,
    perspective: str,
    model_id: str,
) -> Path:
    return artifact_root / f"performance_assessment-{experiment}-{perspective}-{model_id}.h5"


def get_cell_covering_path(artifact_root: Path, experiment: str) -> Path:
    return artifact_root / f"cell_covering-{experiment}.h5"


def _normalize_indexed_dataset(dataset: xr.Dataset) -> xr.Dataset:
    index_dims = [
        name
        for name, index in dataset.indexes.items()
        if name in dataset.dims and isinstance(index, pd.MultiIndex)
    ]
    if not index_dims:
        return dataset
    return dataset.reset_index(index_dims)


def load_assessment(
    artifact_root: Path,
    experiment: str,
    perspective: str,
    model_id: str,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    path = get_assessment_path(artifact_root, experiment, perspective, model_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing assessment artifact: {path}")
    tree = xr.open_datatree(path, engine="h5netcdf")
    tree.load()
    temporal = _normalize_indexed_dataset(tree["temporal"].to_dataset())
    spatial = _normalize_indexed_dataset(tree["spatial"].to_dataset())
    adaptive = _normalize_indexed_dataset(tree["adaptive"].to_dataset())
    tree.close()
    return temporal, spatial, adaptive


def load_assessments(
    artifact_root: Path,
    experiment: str,
    perspectives: list[str],
    model_ids: list[str],
) -> dict[str, dict[str, tuple[xr.Dataset, xr.Dataset, xr.Dataset]]]:
    assessments: dict[str, dict[str, tuple[xr.Dataset, xr.Dataset, xr.Dataset]]] = {}
    for perspective in perspectives:
        assessments[perspective] = {}
        for model_id in model_ids:
            assessments[perspective][model_id] = load_assessment(
                artifact_root,
                experiment,
                perspective,
                model_id,
            )
    return assessments


def load_cell_covering(artifact_root: Path, experiment: str) -> xr.DataArray:
    path = get_cell_covering_path(artifact_root, experiment)
    if not path.exists():
        raise FileNotFoundError(f"Missing cell covering artifact: {path}")
    cell_covering = xr.open_dataarray(path, engine="h5netcdf").load()
    index_dims = [dim for dim in ("X", "Y") if dim in cell_covering.dims and dim in cell_covering.indexes]
    if index_dims:
        cell_covering = cell_covering.reset_index(index_dims)
    return cell_covering


def filter_from_attrs(attrs: dict[str, object], timeframe: list[str] | None = None) -> xr.Dataset:
    attrs_local = dict(attrs)
    if timeframe is not None:
        attrs_local["timeframe"] = timeframe
    missing = REQUIRED_FILTER_ATTRS.difference(attrs_local)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"Missing required calibration attrs: {missing_text}")

    return xr.Dataset(
        {
            "timeframe": xr.DataArray(
                data=[attrs_local["timeframe"], attrs_local["timeframe_etas"]],
                dims=["purpose", "epoch"],
                coords={
                    "purpose": ["calibration", "etas"],
                    "epoch": ["start", "finish"],
                },
            ),
            "polygon": xr.DataArray(
                data=[attrs_local["polygon"], attrs_local["polygon_etas"]],
                dims=["purpose"],
            ),
            "mmin": xr.DataArray(
                data=[attrs_local["mmin"], attrs_local["mmin_etas"]],
                dims=["purpose"],
            ),
        }
    )


def build_testsuite_from_calibration(
    idata: az.InferenceData,
    model_specs: dict[str, dict[str, object]],
    model_id: str,
    grid_data: xr.Dataset,
    event_data: xr.Dataset,
    timeframe: list[str] | None = None,
) -> xr.DataTree:
    specs = model_specs[model_id]
    filterset = filter_from_attrs(idata.attrs, timeframe=timeframe).sel(purpose="calibration")
    return prepare_testsuite(
        idata,
        grid_data.sel(specs.get("sel", {})),
        event_data,
        filterset,
    )


__all__ = [
    "DEFAULT_MODEL_NAMES",
    "InvestigationPaths",
    "build_testsuite_from_calibration",
    "default_paths",
    "filter_from_attrs",
    "get_assessment_path",
    "get_calibration_path",
    "get_cell_covering_path",
    "infer_repo_root",
    "load_assessment",
    "load_assessments",
    "load_calibration",
    "load_calibrations",
    "load_cell_covering",
    "load_context",
    "load_model_specs",
]