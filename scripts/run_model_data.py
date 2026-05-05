from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
CHAINTOOLS_ROOT = SRC_ROOT / "chaintools"
for path in [SRC_ROOT, CHAINTOOLS_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from workflow_support.logging import configure_logging, get_logger, progress  # noqa: E402
from workflow_support.model_data import (  # noqa: E402
    build_filterset,
    generate_calibration_dataset,
    load_inputs,
    load_scenarios,
)
from workflow_support.paths import default_data_scenarios_path, default_source_data_root  # noqa: E402


CURRENT_SCENARIOS = ["ETS_rmax", "EVTS_rmax", "ETS_bs", "EVTS_bs"]
LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model-data generation from YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "model_data.yaml",
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


def get_model_data_path(outdir: Path, experiment: str, perspective: str, scenario_name: str) -> Path:
    return outdir / f"model_data-{experiment}-{perspective}-{scenario_name}.h5"


def validate_scenario_names(scenario_names: list[str], scenarios: dict[str, object], scenarios_file: Path) -> None:
    missing = [name for name in scenario_names if name not in scenarios]
    if missing:
        raise ValueError(
            f"Undefined scenarios in {scenarios_file}: {missing}. "
            f"Available scenarios: {sorted(scenarios.keys())}"
        )


def validate_perspective_timeframes(
    perspectives: list[str],
    perspective_timeframes: dict[str, object],
    config_path: Path,
) -> None:
    missing = [perspective for perspective in perspectives if perspective not in perspective_timeframes]
    if missing:
        raise ValueError(
            f"Missing perspective_timeframes entries in {config_path} for: {missing}. "
            f"Available entries: {sorted(perspective_timeframes.keys())}"
        )

    for perspective in perspectives:
        timeframe_config = perspective_timeframes[perspective]
        if not isinstance(timeframe_config, dict):
            raise ValueError(
                f"Expected perspective_timeframes['{perspective}'] in {config_path} to be a mapping."
            )
        missing_purposes = [purpose for purpose in ["calibration", "etas"] if purpose not in timeframe_config]
        if missing_purposes:
            raise ValueError(
                f"Missing perspective_timeframes['{perspective}'] entries in {config_path}: {missing_purposes}"
            )


def build_and_write_scenario_dataset(
    output_path: Path,
    data_dir: Path,
    scenarios_file: Path,
    perspective: str,
    polygon: str,
    mmin: float,
    perspective_timeframes: dict[str, dict[str, list[str]]],
    scenario_name: str,
    target_step: float,
    etas_d: float,
    etas_q: float,
    attribute_step: float,
) -> str:
    scenarios = load_scenarios(scenarios_file)
    validate_scenario_names([scenario_name], scenarios, scenarios_file)
    event_data, grid_data = load_inputs(data_dir)
    filterset = build_filterset(perspective, polygon, mmin, perspective_timeframes)
    dataset = generate_calibration_dataset(
        event_data=event_data,
        grid_data=grid_data,
        filterset=filterset,
        scenario_kwargs=scenarios[scenario_name],
        target_step=target_step,
        etas_d=etas_d,
        etas_q=etas_q,
        attribute_step=attribute_step,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path, engine="h5netcdf")
    return scenario_name


def main() -> None:
    args = parse_args()
    configure_logging()
    config = yaml.safe_load(args.config.read_text()) or {}
    cache_mode = get_cache_mode(args)

    repo_root = resolve_path(REPO_ROOT, config.get("repo_root")) or REPO_ROOT
    experiment = config.get("experiment", "groningen_1995_2025")
    source_data_root = resolve_path(repo_root, config.get("source_data_root"))
    if source_data_root is None:
        source_data_root = default_source_data_root(repo_root)
    outdir = resolve_path(repo_root, config.get("outdir"))
    if outdir is None:
        outdir = repo_root / "data" / "generated_model_data"

    scenarios_file = resolve_path(repo_root, config.get("data_scenarios_file"))
    if scenarios_file is None:
        scenarios_file = default_data_scenarios_path(repo_root, experiment)

    perspectives = config.get("perspectives", ["prospective", "retrospective"])
    polygon = config.get("polygon", "GroningenFieldGWC")
    mmin = float(config.get("mmin", 1.45))
    perspective_timeframes = config.get("perspective_timeframes") or {}
    target_step = float(config.get("target_step", 0.02))
    etas_d = float(config.get("etas_d", (2000.0) ** 2))
    etas_q = float(config.get("etas_q", 3.16))
    attribute_step = float(config.get("attribute_step", 25.0))
    scenario_names = config.get("scenarios", CURRENT_SCENARIOS)
    workers = max(1, int(config.get("workers", 1)))

    validate_perspective_timeframes(list(perspectives), perspective_timeframes, args.config)
    scenarios = load_scenarios(scenarios_file)
    validate_scenario_names(list(scenario_names), scenarios, scenarios_file)
    event_data, grid_data = load_inputs(source_data_root)

    LOGGER.info(
        "stage=model_data configured experiment=%s perspectives=%s scenarios=%s workers=%s cache=%s source_data_root=%s outdir=%s",
        experiment,
        perspectives,
        scenario_names,
        workers,
        cache_mode,
        source_data_root,
        outdir,
    )

    total_perspectives = len(perspectives)
    for perspective_index, perspective in enumerate(perspectives, start=1):
        LOGGER.info(
            "stage=model_data perspective=%s status=start index=%s/%s",
            perspective,
            perspective_index,
            total_perspectives,
        )
        filterset = build_filterset(perspective, polygon, mmin, perspective_timeframes)
        pending_outputs: list[tuple[str, Path]] = []
        cached_outputs: list[str] = []
        for scenario_name in scenario_names:
            scenario_path = get_model_data_path(outdir, experiment, perspective, scenario_name)
            if should_write_output(scenario_path, cache_mode):
                pending_outputs.append((scenario_name, scenario_path))
            else:
                cached_outputs.append(scenario_name)

        if not pending_outputs:
            LOGGER.info(
                "stage=model_data perspective=%s status=cached scenarios=%s",
                perspective,
                sorted(cached_outputs),
            )
            continue

        if workers == 1:
            total_scenarios = len(pending_outputs)
            for scenario_index, (scenario_name, scenario_path) in enumerate(pending_outputs, start=1):
                LOGGER.info(
                    "stage=model_data perspective=%s scenario=%s status=build index=%s/%s output=%s",
                    perspective,
                    scenario_name,
                    scenario_index,
                    total_scenarios,
                    scenario_path,
                )
                dataset = generate_calibration_dataset(
                    event_data=event_data,
                    grid_data=grid_data,
                    filterset=filterset,
                    scenario_kwargs=scenarios[scenario_name],
                    target_step=target_step,
                    etas_d=etas_d,
                    etas_q=etas_q,
                    attribute_step=attribute_step,
                )
                scenario_path.parent.mkdir(parents=True, exist_ok=True)
                dataset.to_netcdf(scenario_path, engine="h5netcdf")
                LOGGER.info(
                    "stage=model_data perspective=%s scenario=%s status=wrote output=%s",
                    perspective,
                    scenario_name,
                    scenario_path,
                )
        else:
            futures = {}
            max_workers = min(workers, len(pending_outputs))
            LOGGER.info(
                "stage=model_data perspective=%s status=parallel workers=%s scenarios=%s",
                perspective,
                max_workers,
                sorted(name for name, _ in pending_outputs),
            )
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for scenario_name, scenario_path in pending_outputs:
                    futures[
                        executor.submit(
                            build_and_write_scenario_dataset,
                            scenario_path,
                            source_data_root,
                            scenarios_file,
                            perspective,
                            polygon,
                            mmin,
                            perspective_timeframes,
                            scenario_name,
                            target_step,
                            etas_d,
                            etas_q,
                            attribute_step,
                        )
                    ] = scenario_name
                for future in progress(as_completed(futures), desc=f"{perspective} scenarios", total=len(futures)):
                    scenario_name = future.result()
                    LOGGER.info(
                        "stage=model_data perspective=%s scenario=%s status=wrote output=%s",
                        perspective,
                        scenario_name,
                        get_model_data_path(outdir, experiment, perspective, scenario_name),
                    )
        LOGGER.info(
            "stage=model_data perspective=%s status=done computed=%s cached=%s",
            perspective,
            sorted(name for name, _ in pending_outputs),
            sorted(cached_outputs),
        )


if __name__ == "__main__":
    main()
