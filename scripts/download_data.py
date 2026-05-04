from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

import requests
from tqdm import tqdm

from workflow_support.zenodo import (
    DEFAULT_CONFIG as DEFAULT_ZENODO_CONFIG,
    resolve_archive,
    resolve_archive_config,
    resolve_path as resolve_shared_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = DEFAULT_ZENODO_CONFIG
DEFAULT_DOWNLOAD_DIR = REPO_ROOT / "data" / "release" / "downloads"
DEFAULT_EXTRACT_ROOT = REPO_ROOT / "data" / "resources"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the published raw-resource bundle."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML configuration file.",
    )
    parser.add_argument(
        "--archive-key",
        default="resources",
        help="Archive key from the shared Zenodo configuration.",
    )
    parser.add_argument(
        "--cache",
        choices=["error", "reuse", "refresh"],
        default="reuse",
        help="How to handle existing extracted resources: error, reuse them, or refresh them.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(repo_root: Path, value: str | None) -> Path | None:
    return resolve_shared_path(repo_root, value)


def get_cache_mode(args: argparse.Namespace) -> str:
    return "refresh" if args.force else args.cache


def resolve_expected_paths(repo_root: Path, config: dict[str, object]) -> list[Path]:
    values = config.get("expected_cache_paths", ["data/resources"])
    return [resolve_path(repo_root, str(value)) or repo_root for value in values]


def download_file(url: str, dest_path: Path) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        total=total_size_in_bytes, unit="iB", unit_scale=True, desc=dest_path.name
    )

    with dest_path.open("wb") as file_handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            progress_bar.update(len(chunk))
            file_handle.write(chunk)
    progress_bar.close()


def ensure_within_directory(root: Path, candidate: Path) -> None:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    if resolved_candidate != resolved_root and resolved_root not in resolved_candidate.parents:
        raise ValueError(f"Refusing to extract outside {root}: {candidate}")


def extract_zip_archive(archive_path: Path, extract_dir: Path) -> None:
    print(f"Extracting {archive_path.name}...")
    with ZipFile(archive_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            if not member:
                continue
            ensure_within_directory(extract_dir, extract_dir / member)
        zip_ref.extractall(extract_dir)
    print(f"Extracted {archive_path.name}.")


def archive_members_present(archive_path: Path, extract_root: Path) -> bool:
    with ZipFile(archive_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            if not member or member.endswith("/"):
                continue
            candidate = extract_root / member
            ensure_within_directory(extract_root, candidate)
            if not candidate.exists():
                return False
    return True


def main() -> None:
    args = parse_args()
    cache_mode = get_cache_mode(args)
    archive_config = resolve_archive_config(args.archive_key, args.config)
    archive_name, artifact_url = resolve_archive(args.archive_key, args.config)

    download_dir = resolve_path(REPO_ROOT, archive_config.get("download_dir")) or DEFAULT_DOWNLOAD_DIR
    extract_root = resolve_path(REPO_ROOT, archive_config.get("extract_root")) or DEFAULT_EXTRACT_ROOT
    expected_paths = resolve_expected_paths(REPO_ROOT, archive_config)
    archive_path = download_dir / archive_name
    legacy_archive_path = extract_root / archive_name

    download_dir.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    ready_archive_path = archive_path if archive_path.exists() else None
    if ready_archive_path is None and legacy_archive_path.exists() and legacy_archive_path != archive_path:
        ready_archive_path = legacy_archive_path

    extraction_ready = False
    if ready_archive_path is not None and all(path.exists() for path in expected_paths):
        extraction_ready = archive_members_present(ready_archive_path, extract_root)

    if extraction_ready:
        if cache_mode == "reuse":
            print(f"Cached raw resources already available via {ready_archive_path}. Skipping download and extraction.")
            return
        if cache_mode == "error":
            raise FileExistsError(
                "Refusing to overwrite existing raw-resource extraction. "
                "Use --force or --cache refresh to replace it, or --cache reuse to keep it."
            )

    if archive_path.exists():
        print(f"File {archive_name} already exists. Skipping download.")
    elif legacy_archive_path.exists() and legacy_archive_path != archive_path:
        print(f"File {archive_name} already exists at {legacy_archive_path}. Reusing legacy archive location.")
        archive_path = legacy_archive_path
    else:
        print(f"Downloading {archive_name}...")
        download_file(str(artifact_url), archive_path)

    extract_zip_archive(archive_path, extract_root)


if __name__ == "__main__":
    main()
