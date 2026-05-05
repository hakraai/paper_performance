from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "zenodo_downloads.yaml"


def resolve_path(repo_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def load_config(config_path: Path | None = None) -> dict[str, object]:
    path = config_path or DEFAULT_CONFIG
    return yaml.safe_load(path.read_text()) or {}


def resolve_archive_config(archive_key: str, config_path: Path | None = None) -> dict[str, object]:
    config = load_config(config_path)
    archives = config.get("archives") or {}
    archive_config = archives.get(archive_key)
    if not isinstance(archive_config, dict):
        raise ValueError(
            f"No archive configuration found for key '{archive_key}' in {config_path or DEFAULT_CONFIG}."
        )
    return dict(archive_config)


def resolve_archive_name(archive_key: str, config_path: Path | None = None) -> str:
    archive_config = resolve_archive_config(archive_key, config_path)
    archive_name = archive_config.get("archive_name")
    if not archive_name:
        raise ValueError(
            f"No archive configured for key '{archive_key}' in {config_path or DEFAULT_CONFIG}."
        )
    return str(archive_name)


def build_archive_url(archive_name: str, config_path: Path | None = None) -> str:
    config = load_config(config_path)
    record_api_url = str(config.get("record_api_url") or "").rstrip("/")
    if not record_api_url:
        raise ValueError(f"No record_api_url configured in {config_path or DEFAULT_CONFIG}.")
    return f"{record_api_url}/{quote(archive_name)}/content"


def resolve_archive(archive_key: str, config_path: Path | None = None) -> tuple[str, str]:
    archive_name = resolve_archive_name(archive_key, config_path)
    return archive_name, build_archive_url(archive_name, config_path)