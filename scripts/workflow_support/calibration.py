from __future__ import annotations

from pathlib import Path

import xarray as xr


def get_inference_data(path: Path, experiment: str, perspective: str, model: str) -> xr.Dataset:
    ids = xr.open_dataset(
        path / f"model_data-{experiment}-{perspective}-{model}.h5",
        decode_timedelta=False,
        engine="h5netcdf",
    )
    if "bernstein_basis" in ids.dims:
        ids = ids.set_xindex(["bernstein_degree", "bernstein_index"])
    return ids


def get_settings() -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    settings_dsm = {
        "default": {
            "interpolation_dims": [
                "rmax",
                "sigma",
                "hs_exp",
                "hs_exp_alt",
                "M_plastic",
            ],
        },
    }
    settings_rate = {
        "default": {
            "mixing_dims": ["bernstein_index"],
            "theta0": {
                "dist": "Normal",
                "mu": -17.0,
                "sigma": 5.0,
            },
            "theta1": {
                "dist": "Exponential",
                "scale": 5.0,
                "dims": ["bernstein_index"],
            },
        },
    }
    settings_etas = {
        "default": None,
        "Kac": {
            "interpolation_dims": ["etas_d", "etas_q"],
            "etas_c": {"dist": "LogNormal", "mu": 0.0, "sigma": 1.0},
            "etas_p": 1.35,
            "etas_K": {"dist": "Exponential", "scale": 0.1},
            "etas_a": {"dist": "Uniform", "lower": 0.0, "upper": 2.0},
        },
    }
    settings_size = {"default": None}
    return settings_dsm, settings_rate, settings_etas, settings_size
