from __future__ import annotations

from pathlib import Path


def default_source_data_root(repo_root: Path) -> Path:
    return repo_root / "data" / "generated_source_data"


def default_experiment_config_dir(repo_root: Path, experiment: str) -> Path:
    return repo_root / "configs" / "experiments" / experiment


def default_model_specs_path(repo_root: Path, experiment: str) -> Path:
    return default_experiment_config_dir(repo_root, experiment) / "model_specs.yaml"


def default_data_scenarios_path(repo_root: Path, experiment: str) -> Path:
    return default_experiment_config_dir(repo_root, experiment) / "data_scenarios.yaml"