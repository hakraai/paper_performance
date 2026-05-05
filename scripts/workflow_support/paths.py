from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(repo_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def default_source_data_root(repo_root: Path) -> Path:
    return repo_root / "data" / "generated_source_data"


def default_experiment_config_dir(repo_root: Path, experiment: str) -> Path:
    return repo_root / "configs" / "experiments" / experiment


def default_model_specs_path(repo_root: Path, experiment: str) -> Path:
    return default_experiment_config_dir(repo_root, experiment) / "model_specs.yaml"


def default_data_scenarios_path(repo_root: Path, experiment: str) -> Path:
    return default_experiment_config_dir(repo_root, experiment) / "data_scenarios.yaml"


def get_idata_path(path: Path, experiment: str, perspective: str, model: str) -> Path:
    return path / f"model_calibration-{experiment}-{perspective}-{model}.h5"


def get_testsuite_path(path: Path, experiment: str, perspective: str, model: str) -> Path:
    return path / f"testsuite-{experiment}-{perspective}-{model}.h5"


def get_assessment_path(path: Path, experiment: str, perspective: str, model: str) -> Path:
    return path / f"performance_assessment-{experiment}-{perspective}-{model}.h5"


def get_cell_covering_path(path: Path, experiment: str) -> Path:
    return path / f"cell_covering-{experiment}.h5"


__all__ = [
    "REPO_ROOT",
    "default_data_scenarios_path",
    "default_experiment_config_dir",
    "default_model_specs_path",
    "default_source_data_root",
    "get_assessment_path",
    "get_cell_covering_path",
    "get_idata_path",
    "get_testsuite_path",
    "resolve_path",
]