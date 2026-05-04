from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from workflow_support.logging import configure_logging, get_logger
from workflow_support.zenodo import (
    DEFAULT_CONFIG as DEFAULT_ZENODO_CONFIG,
    resolve_archive_config,
    resolve_archive_name,
    resolve_path as resolve_shared_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = DEFAULT_ZENODO_CONFIG
LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package downstream generated workflow artifacts into a Zenodo-ready cache archive."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML configuration file.",
    )
    parser.add_argument(
        "--archive-key",
        default="generated_artifacts",
        help="Archive key from the shared Zenodo configuration.",
    )
    parser.add_argument(
        "--cache",
        choices=["error", "reuse", "refresh"],
        default="reuse",
        help="How to handle an existing package: error, reuse it, or rebuild it.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(repo_root: Path, value: str | None) -> Path | None:
    return resolve_shared_path(repo_root, value)


def get_cache_mode(args: argparse.Namespace) -> str:
    return "refresh" if args.force else args.cache


def resolve_expected_paths(repo_root: Path, config: dict[str, object]) -> list[Path]:
    values = config.get(
        "expected_cache_paths",
        [
            "data/generated_model_data",
            "data/generated_calibrations",
            "data/generated_assessment",
            "figures/generated_paper",
        ],
    )
    return [resolve_path(repo_root, str(value)) or repo_root for value in values]


def should_write_output(path: Path, cache_mode: str) -> bool:
    if not path.exists():
        return True
    if cache_mode == "refresh":
        return True
    if cache_mode == "reuse":
        LOGGER.info("cached %s", path)
        return False
    raise FileExistsError(
        f"Refusing to overwrite existing package: {path}. Use --force or --cache refresh to replace it, or --cache reuse to keep it."
    )


def iter_artifact_files(expected_paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    missing: list[Path] = []
    for path in expected_paths:
        if not path.exists():
            missing.append(path)
            continue
        if path.is_file():
            files.append(path)
            continue
        files.extend(sorted(candidate for candidate in path.rglob("*") if candidate.is_file()))
    if missing:
        raise FileNotFoundError(
            "Cannot package generated artifacts because these expected cache paths are missing: "
            + ", ".join(str(path) for path in missing)
        )
    if not files:
        raise FileNotFoundError("No generated artifact files found under the expected cache paths.")
    return files


def package_files(repo_root: Path, archive_path: Path, files: list[Path]) -> list[dict[str, object]]:
    members: list[dict[str, object]] = []
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            relative_path = path.relative_to(repo_root)
            archive.write(path, arcname=relative_path.as_posix())
            members.append(
                {
                    "path": relative_path.as_posix(),
                    "size_bytes": path.stat().st_size,
                }
            )
    return members


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sidecars(archive_path: Path, checksum: str, members: list[dict[str, object]]) -> tuple[Path, Path]:
    checksum_path = archive_path.with_name(f"{archive_path.name}.sha256")
    manifest_path = archive_path.with_name(f"{archive_path.name}.manifest.json")
    checksum_path.write_text(f"{checksum}  {archive_path.name}\n")
    manifest_path.write_text(
        json.dumps(
            {
                "archive": archive_path.name,
                "sha256": checksum,
                "file_count": len(members),
                "members": members,
            },
            indent=2,
        )
        + "\n"
    )
    return checksum_path, manifest_path


def main() -> None:
    args = parse_args()
    configure_logging()
    cache_mode = get_cache_mode(args)
    archive_config = resolve_archive_config(args.archive_key, args.config)

    repo_root = resolve_path(REPO_ROOT, archive_config.get("repo_root")) or REPO_ROOT
    expected_paths = resolve_expected_paths(repo_root, archive_config)
    output_dir = resolve_path(repo_root, archive_config.get("package_output_dir")) or (
        repo_root / "data" / "release" / "packages"
    )
    archive_name = str(
        archive_config.get("archive_name")
        or resolve_archive_name(args.archive_key, args.config)
    )
    archive_path = output_dir / archive_name

    LOGGER.info(
        "stage=package-artifacts configured cache=%s output_dir=%s archive=%s",
        cache_mode,
        output_dir,
        archive_path,
    )

    if not should_write_output(archive_path, cache_mode):
        return

    files = iter_artifact_files(expected_paths)
    members = package_files(repo_root, archive_path, files)
    checksum = sha256sum(archive_path)
    checksum_path, manifest_path = write_sidecars(archive_path, checksum, members)

    LOGGER.info(
        "stage=package-artifacts status=done archive=%s files=%s sha256=%s checksum_file=%s manifest=%s",
        archive_path,
        len(members),
        checksum,
        checksum_path,
        manifest_path,
    )


if __name__ == "__main__":
    main()