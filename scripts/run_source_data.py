from __future__ import annotations

import argparse
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

from workflow_support.logging import configure_logging, get_logger  # noqa: E402
from workflow_support.paths import default_source_data_root  # noqa: E402
from workflow_support.source_data import (  # noqa: E402
    expected_source_data_outputs,
    generate_source_data,
)


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate prepared source-data artifacts from the raw resources bundle."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "source_data.yaml",
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


def should_generate(outputs: list[Path], cache_mode: str) -> bool:
    existing = [path.exists() for path in outputs]
    if not any(existing):
        return True
    if all(existing):
        if cache_mode == "refresh":
            return True
        if cache_mode == "reuse":
            return False
        raise FileExistsError(
            "Refusing to overwrite existing source-data outputs. Use --force or --cache refresh to replace them, or --cache reuse to keep them."
        )
    if cache_mode == "error":
        raise FileExistsError(
            "Found a partial source-data output set. Use --force or --cache refresh to rebuild it, or --cache reuse after completing the set."
        )
    return True


def main() -> None:
    args = parse_args()
    configure_logging()
    config = yaml.safe_load(args.config.read_text())
    cache_mode = get_cache_mode(args)

    repo_root = resolve_path(REPO_ROOT, config.get("repo_root")) or REPO_ROOT
    resources_root = resolve_path(repo_root, config.get("resources_root")) or (repo_root / "data" / "resources")
    source_data_root = resolve_path(repo_root, config.get("source_data_root")) or default_source_data_root(repo_root)

    LOGGER.info(
        "stage=source-data configured cache=%s resources_root=%s source_data_root=%s",
        cache_mode,
        resources_root,
        source_data_root,
    )

    expected_outputs = expected_source_data_outputs(source_data_root)
    if not should_generate(expected_outputs, cache_mode):
        LOGGER.info("stage=source-data status=cached output_root=%s", source_data_root)
        return

    generate_source_data(resources_root, source_data_root, config)
    LOGGER.info("stage=source-data status=done output_root=%s", source_data_root)


if __name__ == "__main__":
    main()