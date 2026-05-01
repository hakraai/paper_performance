from __future__ import annotations

import argparse
from pathlib import Path
import tarfile
from urllib.parse import urlparse
from zipfile import ZipFile

import requests
from tqdm import tqdm
import yaml

from workflow_support.logging import configure_logging, get_logger


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "generated_artifacts_download.yaml"
LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract an optional generated-artifact cache bundle."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML configuration file.",
    )
    parser.add_argument(
        "--url",
        help="Direct archive URL. Overrides artifact_archive_url from the config file.",
    )
    parser.add_argument(
        "--cache",
        choices=["error", "reuse", "refresh"],
        default="reuse",
        help="How to handle existing extracted artifacts: error, reuse them, or refresh them.",
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


def archive_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        raise ValueError(f"Could not derive an archive name from URL: {url}")
    return name


def resolve_expected_paths(repo_root: Path, config: dict[str, object]) -> list[Path]:
    values = config.get(
        "expected_cache_paths",
        [
            "data/generated_source_data",
            "data/generated_model_data",
            "data/generated_calibrations",
            "data/generated_assessment",
            "figures/generated_paper",
        ],
    )
    return [resolve_path(repo_root, str(value)) or repo_root for value in values]


def validate_cache_state(expected_paths: list[Path], cache_mode: str) -> bool:
    existing = [path for path in expected_paths if path.exists()]
    if len(existing) == len(expected_paths) and cache_mode == "reuse":
        for path in existing:
            LOGGER.info("cached %s", path)
        return False
    if existing and cache_mode == "error":
        raise FileExistsError(
            "Refusing to overwrite existing generated-artifact cache paths: "
            + ", ".join(str(path) for path in existing)
        )
    return True


def download_file(url: str, dest_path: Path) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
        desc=dest_path.name,
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


def extract_zip_archive(archive_path: Path, extract_root: Path) -> None:
    with ZipFile(archive_path, "r") as archive:
        for member in archive.namelist():
            if not member:
                continue
            ensure_within_directory(extract_root, extract_root / member)
        archive.extractall(extract_root)


def extract_tar_archive(archive_path: Path, extract_root: Path) -> None:
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            if not member.name:
                continue
            ensure_within_directory(extract_root, extract_root / member.name)
        archive.extractall(extract_root)


def extract_archive(archive_path: Path, extract_root: Path) -> None:
    suffixes = archive_path.suffixes
    if archive_path.suffix == ".zip":
        extract_zip_archive(archive_path, extract_root)
        return
    if suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix in {".tgz", ".tar"}:
        extract_tar_archive(archive_path, extract_root)
        return
    if tarfile.is_tarfile(archive_path):
        extract_tar_archive(archive_path, extract_root)
        return
    raise ValueError(f"Unsupported archive format: {archive_path}")


def main() -> None:
    args = parse_args()
    configure_logging()
    config = yaml.safe_load(args.config.read_text()) or {}
    cache_mode = get_cache_mode(args)

    repo_root = resolve_path(REPO_ROOT, config.get("repo_root")) or REPO_ROOT
    extract_root = resolve_path(repo_root, config.get("extract_root")) or repo_root
    download_dir = resolve_path(repo_root, config.get("download_dir")) or (repo_root / "data" / "obsolete" / "downloads")
    expected_paths = resolve_expected_paths(repo_root, config)
    artifact_url = args.url or config.get("artifact_archive_url")

    if not artifact_url:
        raise ValueError(
            "No generated-artifact archive URL configured. Fill in artifact_archive_url in "
            f"{args.config} or pass --url."
        )

    LOGGER.info(
        "stage=download-artifacts configured cache=%s extract_root=%s download_dir=%s",
        cache_mode,
        extract_root,
        download_dir,
    )

    if not validate_cache_state(expected_paths, cache_mode):
        LOGGER.info("stage=download-artifacts status=cached")
        return

    archive_name = str(config.get("archive_name") or archive_name_from_url(str(artifact_url)))
    archive_path = download_dir / archive_name
    download_dir.mkdir(parents=True, exist_ok=True)

    if cache_mode == "refresh" or not archive_path.exists():
        LOGGER.info("stage=download-artifacts status=download url=%s archive=%s", artifact_url, archive_path)
        download_file(str(artifact_url), archive_path)
    else:
        LOGGER.info("stage=download-artifacts status=reuse-archive archive=%s", archive_path)

    LOGGER.info("stage=download-artifacts status=extract archive=%s", archive_path)
    extract_archive(archive_path, extract_root)

    missing = [path for path in expected_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Artifact archive extraction completed, but expected cache paths were not created: "
            + ", ".join(str(path) for path in missing)
        )

    LOGGER.info("stage=download-artifacts status=done expected_paths=%s", [str(path) for path in expected_paths])


if __name__ == "__main__":
    main()