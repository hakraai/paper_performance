from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import time

import numpy as np
import pymc as pm
import xarray as xr
import yaml
from pymc.model.transform.optimization import freeze_dims_and_data


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
CHAINTOOLS_ROOT = SRC_ROOT / "chaintools"
for path in [SRC_ROOT, CHAINTOOLS_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import model_chain_inference as mci  # noqa: E402
from workflow_support.logging import configure_logging, get_logger  # noqa: E402
from workflow_support.paths import default_model_specs_path, default_source_data_root  # noqa: E402


LOGGER = get_logger(__name__)


def log_multiline_info(prefix: str, text: str) -> None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            LOGGER.info("%s | %s", prefix, stripped)


def load_inference_data(path: Path, experiment: str, perspective: str, model: str) -> xr.Dataset:
    ids = xr.open_dataset(
        path / f"model_data-{experiment}-{perspective}-{model}.h5",
        decode_timedelta=False,
        engine="h5netcdf",
    )
    if "bernstein_basis" in ids.dims:
        ids = ids.set_xindex(["bernstein_degree", "bernstein_index"])
    return ids


def load_calibration_settings(config: dict[str, object], repo_root: Path) -> dict[str, dict[str, object] | None]:
    settings_path = resolve_path(repo_root, config.get("settings_file"))
    if settings_path is None:
        settings_path = repo_root / "configs" / "model_calibration_settings.yaml"
    settings = yaml.safe_load(settings_path.read_text()) or {}
    return {
        "dsm": settings.get("dsm", {}),
        "rate": settings.get("rate", {}),
        "etas": settings.get("etas", {}),
        "size": settings.get("size", {}),
    }


def resolve_model_settings(
    settings_catalog: dict[str, dict[str, object] | None],
    model_spec: dict[str, object],
) -> dict[str, object]:
    return {
        "rate_parameter_data": settings_catalog["rate"][model_spec.get("rate_model_id", "default")],
        "size_parameter_data": settings_catalog["size"][model_spec.get("size_model_id", "default")],
        "dsm_parameter_data": settings_catalog["dsm"][model_spec.get("dsm_model_id", "default")],
        "etas_parameter_data": settings_catalog["etas"][model_spec.get("etas_model_id", "default")],
    }


def build_validated_model(
    model_name: str,
    perspective: str,
    ids: object,
    settings_local: dict[str, object],
) -> object:
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        model = mci.generate_and_test_model(
            mci.generate_ts_etf_etas_model,
            data=ids,
            settings=settings_local,
        )
    model_debug_output = buffer.getvalue().strip()
    if model_debug_output:
        log_multiline_info(f"model debug [{perspective}/{model_name}]", model_debug_output)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the calibration workflow from YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "model_calibration.yaml",
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


def main() -> None:
    args = parse_args()
    configure_logging()
    config = yaml.safe_load(args.config.read_text())
    cache_mode = get_cache_mode(args)

    repo_root = resolve_path(REPO_ROOT, config.get("repo_root")) or REPO_ROOT
    source_data_root = resolve_path(repo_root, config.get("source_data_root"))
    if source_data_root is None:
        source_data_root = default_source_data_root(repo_root)
    experiment = config.get("experiment", "groningen_1995_2025")
    data_root = resolve_path(repo_root, config.get("data_root"))
    if data_root is None:
        data_root = repo_root / "data" / "generated_model_data"
    outdir = resolve_path(repo_root, config.get("outdir"))
    if outdir is None:
        outdir = repo_root / "data" / "generated_calibrations"
    perspectives = config.get("perspectives", ["prospective", "retrospective"])
    draws = int(config.get("draws", 1000))
    chains = int(config.get("chains", 8))
    cores = int(config.get("cores", chains))
    seed = int(config.get("seed", 1234))
    target_accept = float(config.get("target_accept", 0.9))
    nuts_sampler = config.get("nuts_sampler")
    show_sampling_progress = bool(config.get("show_sampling_progress", True))

    specs_path = resolve_path(repo_root, config.get("model_specs_file"))
    if specs_path is None:
        specs_path = default_model_specs_path(repo_root, experiment)
    model_specs = yaml.safe_load(specs_path.read_text())
    models = config.get("models", list(model_specs.keys()))

    settings_catalog = load_calibration_settings(config, repo_root)
    outdir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "stage=calibration configured experiment=%s perspectives=%s models=%s draws=%s chains=%s cores=%s cache=%s source_data_root=%s",
        experiment,
        perspectives,
        models,
        draws,
        chains,
        cores,
        cache_mode,
        source_data_root,
    )

    total_perspectives = len(perspectives)
    total_models = len(models)
    for perspective_index, perspective in enumerate(perspectives, start=1):
        LOGGER.info(
            "stage=calibration perspective=%s status=start index=%s/%s",
            perspective,
            perspective_index,
            total_perspectives,
        )
        for model_index, model_name in enumerate(models, start=1):
            outpath = outdir / f"model_calibration-{experiment}-{perspective}-{model_name}.h5"
            LOGGER.info(
                "stage=calibration perspective=%s model=%s status=prepare index=%s/%s output=%s",
                perspective,
                model_name,
                model_index,
                total_models,
                outpath,
            )
            if not should_write_output(outpath, cache_mode):
                LOGGER.info(
                    "stage=calibration perspective=%s model=%s status=cached",
                    perspective,
                    model_name,
                )
                continue

            model_spec = model_specs[model_name]
            data_id = model_spec["data_id"]
            settings_local = resolve_model_settings(settings_catalog, model_spec)

            ids = load_inference_data(data_root, experiment, perspective, data_id)
            ids = ids.sel(model_spec.get("sel", {})).isel(model_spec.get("isel", {}))
            LOGGER.info(
                "stage=calibration perspective=%s model=%s status=build data_id=%s",
                perspective,
                model_name,
                data_id,
            )

            model = build_validated_model(
                model_name,
                perspective,
                ids,
                settings_local,
            )

            chosen_sampler = nuts_sampler
            if chosen_sampler is None:
                chosen_sampler = "blackjax" if len(model.discrete_value_vars) == 0 else "pymc"

            LOGGER.info(
                "stage=calibration perspective=%s model=%s status=sampling sampler=%s chains=%s cores=%s draws=%s target_accept=%s progressbar=%s",
                perspective,
                model_name,
                chosen_sampler,
                chains,
                cores,
                draws,
                target_accept,
                show_sampling_progress,
            )
            started = time.monotonic()

            with freeze_dims_and_data(model):
                idata = pm.sample(
                    draws=draws,
                    chains=chains,
                    cores=cores,
                    target_accept=target_accept,
                    idata_kwargs={"log_likelihood": True},
                    random_seed=np.random.default_rng(seed),
                    nuts_sampler=chosen_sampler,
                    progressbar=show_sampling_progress,
                )

            LOGGER.info(
                "stage=calibration perspective=%s model=%s status=sampled elapsed=%.1fs",
                perspective,
                model_name,
                time.monotonic() - started,
            )

            idata.attrs.update(dict(ids.attrs))
            idata.to_netcdf(outpath)
            LOGGER.info(
                "stage=calibration perspective=%s model=%s status=wrote output=%s",
                perspective,
                model_name,
                outpath,
            )
        LOGGER.info("stage=calibration perspective=%s status=done", perspective)


if __name__ == "__main__":
    main()