from __future__ import annotations

from pathlib import Path
import sys

import xarray as xr
import yaml


def _import_model_chain_inference():
    try:
        import model_chain_inference as package
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[2]
        src_root = repo_root / "src"
        chaintools_root = src_root / "chaintools"
        for path in [src_root, chaintools_root]:
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
        import model_chain_inference as package
    return package


mci = _import_model_chain_inference()


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


def build_filterset(
    perspective: str,
    polygon: str,
    mmin: float,
    perspective_timeframes: dict[str, dict[str, list[str]]],
) -> xr.Dataset:
    try:
        timeframe = perspective_timeframes[perspective]
        calibration_timeframe = timeframe["calibration"]
        etas_timeframe = timeframe["etas"]
    except KeyError as exc:
        raise ValueError(
            f"Missing configured timeframes for perspective '{perspective}'. "
            "Expected 'calibration' and 'etas' entries under 'perspective_timeframes'."
        ) from exc

    return xr.Dataset(
        {
            "timeframe": xr.DataArray(
                data=[calibration_timeframe, etas_timeframe],
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
