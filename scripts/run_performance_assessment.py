from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import arviz as az
import pandas as pd
import xarray as xr
import yaml

from workflow_support import assessment_runtime, paths as workflow_paths


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
CHAINTOOLS_ROOT = SRC_ROOT / "chaintools"
for path in [SRC_ROOT, CHAINTOOLS_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import model_chain_inference as mci  # noqa: E402
from workflow_support.filtering import build_perspective_filter_attrs, require_timeframe_config  # noqa: E402
from workflow_support.logging import configure_logging, get_logger, progress  # noqa: E402
from workflow_support.paths import default_model_specs_path, default_source_data_root  # noqa: E402


LOGGER = get_logger(__name__)


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
    _, event_data, _, grid_data, _ = assessment_runtime.load_context(source_data_root)
    cell_covering = assessment_runtime.load_cell_covering(cell_covering_path)
    filterset_used = assessment_runtime.filter_from_attrs(filter_attrs).sel(purpose="calibration")
    testsuite = assessment_runtime.build_testsuite_artifact(
        model_id,
        idata,
        model_specs,
        grid_data,
        event_data,
        filterset_used,
    )
    temporal, spatial, adaptive = assessment_runtime.build_assessment_from_testsuite(
        testsuite,
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


def get_cache_mode(args: argparse.Namespace) -> str:
    return "refresh" if args.force else args.cache


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


def main() -> None:
    args = parse_args()
    configure_logging()
    config = yaml.safe_load(args.config.read_text())
    cache_mode = get_cache_mode(args)

    repo_root = workflow_paths.resolve_path(REPO_ROOT, config.get("repo_root")) or REPO_ROOT
    source_data_root = workflow_paths.resolve_path(repo_root, config.get("source_data_root")) or default_source_data_root(repo_root)
    calibration_root = workflow_paths.resolve_path(repo_root, config.get("calibration_root")) or (repo_root / "data" / "generated_calibrations")
    artifact_dir = workflow_paths.resolve_path(repo_root, config.get("artifact_dir")) or (repo_root / "data" / "generated_assessment")
    experiment = config.get("experiment", "groningen_1995_2025")
    model_specs_file = workflow_paths.resolve_path(repo_root, config.get("model_specs_file")) or default_model_specs_path(repo_root, experiment)
    perspectives = config.get("perspectives", ["prospective", "retrospective"])
    model_names = config.get("model_names", assessment_runtime.DEFAULT_MODEL_NAMES)
    model_ids = config.get("models", list(model_names.keys()))
    pa_sample_size = int(config.get("pa_sample_size", 10_000))
    seed = int(config.get("seed", 42))
    workers = max(1, int(config.get("workers", 1)))
    timeframe_testing = require_timeframe_config(config, "timeframe_testing", args.config)
    timeframe_forecast = require_timeframe_config(config, "timeframe_forecast", args.config)

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

    model_specs = yaml.safe_load(model_specs_file.read_text())
    _, event_data, _, grid_data, grid_specs = assessment_runtime.load_context(source_data_root)
    rc = assessment_runtime.collect_calibrations(
        calibration_root,
        experiment,
        perspectives,
        model_ids,
    )

    forecast_source = "retrospective" if "retrospective" in rc else "prospective"
    forecast_filter_attrs = dict(rc[forecast_source][model_ids[0]].attrs)
    filter_attrs = build_perspective_filter_attrs(
        forecast_filter_attrs,
        timeframe_testing,
        timeframe_forecast,
    )
    retrospective_filter_attrs = filter_attrs["retrospective"]
    filterset_retrospective = assessment_runtime.filter_from_attrs(retrospective_filter_attrs).sel(purpose="calibration")
    prospective_filter_attrs = filter_attrs["prospective"]
    filterset_prospective = assessment_runtime.filter_from_attrs(prospective_filter_attrs).sel(purpose="calibration")
    LOGGER.info(
        "stage=assessment status=filters retrospective_timeframe=%s prospective_timeframe=%s",
        retrospective_filter_attrs["timeframe"],
        prospective_filter_attrs["timeframe"],
    )

    cell_covering_path = workflow_paths.get_cell_covering_path(artifact_dir, experiment)
    cell_covering: xr.DataArray | None = None
    if cell_covering_path.exists() and cache_mode == "reuse":
        try:
            cell_covering = assessment_runtime.load_cell_covering(cell_covering_path)
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
        cell_covering = assessment_runtime.normalize_cell_covering(
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
            testsuite_path = workflow_paths.get_testsuite_path(artifact_dir, experiment, perspective, model_id)
            assessment_path = workflow_paths.get_assessment_path(artifact_dir, experiment, perspective, model_id)

            reusable_artifacts = cache_mode == "reuse"
            cached_testsuite = testsuite_path.exists() and reusable_artifacts
            cached_assessment = assessment_path.exists() and reusable_artifacts

            if cached_testsuite:
                LOGGER.info("cached %s", testsuite_path)
                testsuite_dict[perspective][model_id] = assessment_runtime.load_testsuite(testsuite_path)
            elif testsuite_path.exists() and cache_mode == "error":
                raise FileExistsError(
                    f"Refusing to overwrite existing output: {testsuite_path}. Use --force or --cache refresh to replace it, or --cache reuse to keep it."
                )

            if cached_assessment:
                LOGGER.info("cached %s", assessment_path)
                temporal, spatial, adaptive = assessment_runtime.load_assessment(assessment_path)
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
                    testsuite = assessment_runtime.build_testsuite_artifact(
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
                    temporal, spatial, adaptive = assessment_runtime.build_assessment_from_testsuite(
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
                            workflow_paths.get_idata_path(calibration_root, experiment, perspective, model_id),
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
                    testsuite_dict[perspective][model_id] = assessment_runtime.load_testsuite(
                        workflow_paths.get_testsuite_path(artifact_dir, experiment, perspective, model_id)
                    )
                    temporal, spatial, adaptive = assessment_runtime.load_assessment(
                        workflow_paths.get_assessment_path(artifact_dir, experiment, perspective, model_id)
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