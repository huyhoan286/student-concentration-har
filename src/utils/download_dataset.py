"""
Tải dataset từ Kaggle về thư mục dataset/.

Cấu trúc sau khi tải:
    dataset/train|val|test/{class}/*.mp4

Chạy từ thư mục gốc project:
    python src/utils/download_dataset.py
    python src/utils/download_dataset.py --check-only
    python src/utils/download_dataset.py --force-download

Cấu hình (configs/dataset_config.yaml):
    kaggle_dataset_slug: "owner/dataset-name"
Hoặc: KAGGLE_DATASET_SLUG=owner/dataset-name

Yêu cầu: pip install kaggle và ~/.kaggle/kaggle.json
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

VIDEO_EXTENSION = ".mp4"
SPLITS = ("train", "val", "test")
DEFAULT_DATA_DIR = "dataset"
KAGGLE_CACHE_DIRNAME = "_kaggle_cache"
MIN_TOTAL_VIDEOS = 10


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_dataset_config(project_root: Path) -> dict:
    config_path = project_root / "configs" / "dataset_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_classes(config: dict) -> list[str]:
    classes = config.get("classes")
    if not classes:
        from constants import CLASSES  # noqa: PLC0415

        return list(CLASSES)
    return list(classes)


def get_dataset_dir(project_root: Path, config: dict) -> Path:
    key = config.get("data_dir") or config.get("split_data_dir", DEFAULT_DATA_DIR)
    key = str(key).replace("\\", "/").rstrip("/")
    if key.endswith("split"):
        key = DEFAULT_DATA_DIR
    return (project_root / key).resolve()


def get_kaggle_slug(config: dict) -> str:
    return (
        os.environ.get("KAGGLE_DATASET_SLUG", "").strip()
        or str(config.get("kaggle_dataset_slug", "")).strip()
    )


def count_mp4_in_dir(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(
        1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() == VIDEO_EXTENSION
    )


def count_videos(dataset_dir: Path, classes: list[str]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    for split in SPLITS:
        for class_name in classes:
            folder = dataset_dir / split / class_name
            stats[split][class_name] = count_mp4_in_dir(folder)
    return stats


def total_videos(stats: dict[str, dict[str, int]]) -> int:
    return sum(n for split in stats.values() for n in split.values())


def has_dataset(
    dataset_dir: Path,
    classes: list[str],
    min_total: int = MIN_TOTAL_VIDEOS,
) -> bool:
    """True nếu dataset/ có train/val/test và đủ file .mp4."""
    if not dataset_dir.is_dir():
        return False

    for split in SPLITS:
        if not (dataset_dir / split).is_dir():
            return False

    stats = count_videos(dataset_dir, classes)
    if total_videos(stats) < min_total:
        return False

    return any(stats["train"].get(c, 0) > 0 for c in classes)


def print_dataset_summary(dataset_dir: Path, classes: list[str]) -> None:
    stats = count_videos(dataset_dir, classes)
    print(f"Dataset dir: {dataset_dir}")
    for split in SPLITS:
        per_class = stats[split]
        split_total = sum(per_class.values())
        print(f"  {split}: {split_total} mp4")
        for class_name in classes:
            n = per_class.get(class_name, 0)
            if n > 0:
                print(f"    {class_name}: {n}")
    print(f"  Tổng: {total_videos(stats)} mp4")


def is_dataset_root(path: Path, classes: list[str]) -> bool:
    if not path.is_dir():
        return False
    for split in SPLITS:
        if not (path / split).is_dir():
            return False
    return total_videos(count_videos(path, classes)) >= 1


def locate_dataset_root(search_root: Path, classes: list[str]) -> Path | None:
    """Tìm thư mục chứa train/, val/, test/ sau khi giải nén Kaggle."""
    candidates = [search_root, search_root / "split", search_root / "dataset"]
    for path in candidates:
        if is_dataset_root(path, classes):
            return path

    for train_dir in search_root.rglob("train"):
        if not train_dir.is_dir():
            continue
        parent = train_dir.parent
        if is_dataset_root(parent, classes):
            return parent

    return None


def copy_to_dataset(source_root: Path, dataset_dir: Path) -> None:
    """Copy train/val/test từ cache Kaggle sang dataset/."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        src = source_root / split
        if not src.is_dir():
            continue
        dest = dataset_dir / split
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    print(f"[OK] Đã copy train/val/test -> {dataset_dir}")


def fetch_from_kaggle(slug: str, dest_dir: Path) -> Path:
    """Tải và giải nén dataset từ Kaggle."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle  # noqa: F401, PLC0415
    except ImportError as exc:
        raise RuntimeError("Thiếu package 'kaggle'. Cài: pip install kaggle") from exc

    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        slug,
        "-p",
        str(dest_dir),
        "--unzip",
    ]
    print(f"[download] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Tải Kaggle thất bại (slug={slug}).\n{msg}\n\n"
            "Kiểm tra: pip install kaggle, ~/.kaggle/kaggle.json, quyền dataset."
        )
    print(f"[OK] Đã tải và giải nén: {dest_dir}")
    return dest_dir


def install_from_kaggle_cache(cache_dir: Path, dataset_dir: Path, classes: list[str]) -> None:
    root = locate_dataset_root(cache_dir, classes)
    if root is None:
        raise RuntimeError(
            f"Không tìm thấy train/val/test trong {cache_dir}.\n"
            "Dataset Kaggle cần: train|val|test/{{class}}/*.mp4"
        )
    copy_to_dataset(root, dataset_dir)


def download_dataset(
    project_root: Path | None = None,
    *,
    force_download: bool = False,
    check_only: bool = False,
    min_total: int = MIN_TOTAL_VIDEOS,
) -> bool:
    """
    Kiểm tra dataset/; nếu chưa có thì tải từ Kaggle.

    Returns:
        True nếu dataset đã sẵn sàng.
    """
    root = project_root or get_project_root()
    config = load_dataset_config(root)
    classes = get_classes(config)
    dataset_dir = get_dataset_dir(root, config)

    print("=== Tải dataset từ Kaggle ===\n")

    if has_dataset(dataset_dir, classes, min_total=min_total) and not force_download:
        print("[OK] dataset/ đã có dữ liệu — bỏ qua tải Kaggle.")
        print_dataset_summary(dataset_dir, classes)
        return True

    if check_only:
        print("[MISSING] dataset/ chưa đủ dữ liệu.")
        if dataset_dir.is_dir():
            print_dataset_summary(dataset_dir, classes)
        else:
            print(f"  Thư mục chưa có: {dataset_dir}")
        return False

    slug = get_kaggle_slug(config)
    if not slug:
        raise ValueError(
            "Chưa cấu hình Kaggle slug.\n"
            '  kaggle_dataset_slug: "owner/dataset-name"\n'
            "  hoặc KAGGLE_DATASET_SLUG=owner/dataset-name"
        )

    cache_dir = root / "dataset" / KAGGLE_CACHE_DIRNAME
    if force_download and cache_dir.exists():
        shutil.rmtree(cache_dir)

    fetch_from_kaggle(slug, cache_dir)
    install_from_kaggle_cache(cache_dir, dataset_dir, classes)

    if not has_dataset(dataset_dir, classes, min_total=min_total):
        raise RuntimeError(
            "Sau khi tải, dataset/ vẫn chưa đủ video. "
            f"Kiểm tra slug và {cache_dir}"
        )

    print("\n[OK] Đã tải dataset từ Kaggle.")
    print_dataset_summary(dataset_dir, classes)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tải dataset từ Kaggle về dataset/")
    parser.add_argument("--check-only", action="store_true", help="Chỉ kiểm tra, không tải.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Tải lại dù dataset đã có.",
    )
    parser.add_argument(
        "--min-total",
        type=int,
        default=MIN_TOTAL_VIDEOS,
        help=f"Số .mp4 tối thiểu (mặc định {MIN_TOTAL_VIDEOS}).",
    )
    return parser.parse_args()


def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def main() -> None:
    _configure_stdout()
    args = parse_args()
    utils_dir = Path(__file__).resolve().parent
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))

    ok = download_dataset(
        force_download=args.force_download,
        check_only=args.check_only,
        min_total=args.min_total,
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
