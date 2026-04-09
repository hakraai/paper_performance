from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
CHAINTOOLS_ROOT = SRC_ROOT / "chaintools"
for path in [SRC_ROOT, CHAINTOOLS_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import xarray as xr
import yaml

import model_chain_inference as mci


PERSPECTIVE_TIMEFRAMES = {
    "prospective": {
        "calibration": ["1995-01-01", "2020-10-01"],
        "etas": ["1990-01-01", "2020-10-01"],
    },
    "retrospective": {
        "calibration": ["1995-01-01", "2025-10-01"],
        "etas": ["1990-01-01", "2025-10-01"],
    },
}


def load_inputs(data_dir: Path) -> tuple[xr.Dataset, xr.Dataset]:
    event_data = xr.open_dataset(data_dir / "event_data.h5", decode_coords="all").load()
    grid_data = (
        xr.open_dataset(
            data_dir / "grid_data.h5",
            decode_coords="all",
            decode_timedelta=False,
            engine="h5netcdf",
        )
        .set_xindex(["x", "y"])
        .set_xindex(["bernstein_degree", "bernstein_index"])
    )
    return event_data, grid_data


def build_filterset(perspective: str, polygon: str, mmin: float) -> xr.Dataset:
    timeframe = PERSPECTIVE_TIMEFRAMES[perspective]
    return xr.Dataset(
        {
            "timeframe": xr.DataArray(
                data=[timeframe["calibration"], timeframe["etas"]],
                dims=["purpose", "epoch"],
                coords={"purpose": ["calibration", "etas"], "epoch": ["start", "finish"]},
            ),
            "polygon": xr.DataArray(data=[polygon, polygon], dims=["purpose"]),
            "mmin": xr.DataArray(data=[mmin, mmin], dims=["purpose"]),
        }
    )


def load_scenarios(path: Path) -> dict[str, dict[str, object]]:
    return yaml.safe_load(path.read_text())


def generate_calibration_dataset(
    event_data: xr.Dataset,
    grid_data: xr.Dataset,
    filterset: xr.Dataset,
    scenario_kwargs: dict[str, object],
    target_step: float,
    etas_d: float,
    etas_q: float,
    attribute_step: float,
) -> xr.Dataset:
    kwargs = {
        "event_data": event_data,
        "grid_data": grid_data,
        "filterset": filterset.sel(purpose="calibration", drop=True),
        "filterset_etas": filterset.sel(purpose="etas", drop=True),
        "support_id": "support_fraction",
        "target_step": target_step,
        "attributes": {"reservoir_thickness": attribute_step},
        "etas_d": etas_d,
        "etas_q": etas_q,
    }
    kwargs.update(scenario_kwargs)
    dataset = mci.generate_inference_data_local(**kwargs)
    if "bernstein_basis" in dataset.indexes:
        dataset = dataset.reset_index("bernstein_basis")
    return dataset
