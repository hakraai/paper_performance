from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

import yaml

from workflow_support.logging import configure_logging, get_logger


REPO_ROOT = Path(__file__).resolve().parents[1]
LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full paper workflow from YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "paper_workflow.yaml",
        help="YAML configuration file.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["data", "calibration", "assessment", "figures", "all"],
        default=["all"],
        help="Workflow stages to run.",
    )
    parser.add_argument(
        "--cache",
        choices=["error", "reuse", "refresh"],
        default="reuse",
        help="How to handle existing stage outputs: error, reuse them, or refresh them.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_config_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def run_step(script_name: str, config_path: Path, force: bool, cache_mode: str) -> None:
    stage_name = script_name.removesuffix(".py").removeprefix("run_")
    command = [sys.executable, str(REPO_ROOT / "scripts" / script_name), "--config", str(config_path)]
    command.extend(["--cache", "refresh" if force else cache_mode])
    if force:
        command.append("--force")
    LOGGER.info("stage=%s status=start command=%s", stage_name, " ".join(command))
    started = time.monotonic()
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        LOGGER.exception("stage=%s status=failed", stage_name)
        raise
    LOGGER.info("stage=%s status=done elapsed=%.1fs", stage_name, time.monotonic() - started)


def main() -> None:
    args = parse_args()
    configure_logging()
    config = yaml.safe_load(args.config.read_text())
    steps = set(args.steps)
    if "all" in steps:
        steps = {"data", "calibration", "assessment", "figures"}
    ordered_steps = [step for step in ["data", "calibration", "assessment", "figures"] if step in steps]
    for step in ordered_steps:
        if step == "data":
            run_step("run_model_data.py", resolve_config_path(config["model_data_config"], REPO_ROOT), args.force, args.cache)
        elif step == "calibration":
            run_step("run_model_calibration.py", resolve_config_path(config["calibration_config"], REPO_ROOT), args.force, args.cache)
        elif step == "assessment":
            run_step("run_performance_assessment.py", resolve_config_path(config["performance_assessment_config"], REPO_ROOT), args.force, args.cache)
        elif step == "figures":
            run_step("run_figure_generation.py", resolve_config_path(config["figure_generation_config"], REPO_ROOT), args.force, args.cache)
    LOGGER.info("stage=workflow status=done steps=%s cache=%s force=%s", ordered_steps, args.cache, args.force)


if __name__ == "__main__":
    main()