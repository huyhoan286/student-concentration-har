"""
Thiết lập project sau khi clone — chạy một mạch từ thư mục gốc.

    python scripts/setup_project.py
    python scripts/setup_project.py --skip-install
    python scripts/setup_project.py --skip-download

Windows (khuyến nghị):
    .\\setup.ps1

Linux / macOS:
    ./setup.sh
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = PROJECT_ROOT / "src" / "utils"
MODELS_DIR = PROJECT_ROOT / "src" / "models"


def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def _run(cmd: list[str], *, step: str) -> None:
    print(f"\n>>> {step}")
    print(f"    {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Bước thất bại: {step} (exit {result.returncode})")


def _check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        need = ".".join(map(str, MIN_PYTHON))
        raise RuntimeError(f"Cần Python >={need}, hiện tại: {sys.version.split()[0]}")


def _pip_install() -> None:
    _run(
        [sys.executable, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")],
        step="Cài dependencies (requirements.txt)",
    )


def _ensure_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    example = PROJECT_ROOT / ".env.example"
    if env_path.is_file():
        print(f"[OK] Đã có {env_path}")
        return
    if example.is_file():
        env_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[OK] Đã tạo {env_path} từ .env.example — hãy điền KAGGLE_* trước khi tải data.")
    else:
        print("[cảnh báo] Không có .env.example")


def _load_env() -> None:
    sys.path.insert(0, str(UTILS_DIR))
    from env_loader import load_project_env  # noqa: PLC0415

    if load_project_env(PROJECT_ROOT):
        print("[OK] Đã nạp biến từ .env")
    else:
        print("[info] Không có file .env (dùng configs/ hoặc biến môi trường hệ thống)")


def _has_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN", "").strip():
        return True
    kaggle_dir = Path.home() / ".kaggle"
    return (kaggle_dir / "access_token").is_file() or (kaggle_dir / "kaggle.json").is_file()


def _download_dataset(*, skip_download: bool) -> None:
    sys.path.insert(0, str(UTILS_DIR))
    from download_dataset import download_dataset, get_kaggle_slug, load_dataset_config  # noqa: PLC0415

    if skip_download:
        print("\n[skip] Bỏ qua tải dataset (--skip-download)")
        return

    if download_dataset(PROJECT_ROOT, check_only=True):
        print("\n[OK] dataset/ đã sẵn sàng — bỏ qua tải Kaggle.")
        return

    config = load_dataset_config(PROJECT_ROOT)
    slug = get_kaggle_slug(config)
    if not slug:
        print(
            "\n[MISSING] Chưa cấu hình slug Kaggle.\n"
            "  Sửa .env: KAGGLE_DATASET_SLUG=owner/ten-bo-du-lieu\n"
            "  hoặc configs/dataset_config.yaml → kaggle_dataset_slug\n"
        )
        raise SystemExit(1)

    if not _has_kaggle_credentials():
        print(
            "\n[MISSING] Chưa có xác thực Kaggle.\n"
            "  kagglehub login\n"
            "  hoặc KAGGLE_API_TOKEN trong .env\n"
            "  hoặc ~/.kaggle/kaggle.json\n"
        )
        raise SystemExit(1)

    if not download_dataset(PROJECT_ROOT):
        raise SystemExit(1)


def _smoke_tests() -> None:
    for model in ("lrcn", "convlstm"):
        try:
            _run(
                [sys.executable, str(UTILS_DIR / "test_model_forward.py"), "--model", model],
                step=f"Smoke test {model}",
            )
        except RuntimeError:
            print(f"[cảnh báo] Smoke test {model} thất bại — kiểm tra cài đặt torch/torchvision.")

    try:
        _run(
            [sys.executable, str(UTILS_DIR / "test_model_forward.py"), "--model", "movinet"],
            step="Smoke test movinet",
        )
    except RuntimeError:
        print("[cảnh báo] MoViNet bỏ qua — cài pytorchvideo nếu cần: pip install pytorchvideo")


def _print_next_steps() -> None:
    print(
        "\n"
        "=" * 60
        + "\n  Thiết lập xong. Lệnh tiếp theo:\n"
        + "\n  python src/training/train_lrcn.py"
        + "\n  python src/training/train_convlstm.py"
        + "\n  python src/training/train_movinet.py"
        + "\n  python src/utils/evaluate.py --model lrcn"
        + "\n" + "=" * 60 + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thiết lập project sau clone")
    parser.add_argument("--skip-install", action="store_true", help="Bỏ qua pip install")
    parser.add_argument("--skip-download", action="store_true", help="Bỏ qua tải Kaggle")
    parser.add_argument("--skip-smoke", action="store_true", help="Bỏ qua smoke test model")
    return parser.parse_args()


def main() -> None:
    _configure_stdout()
    os.chdir(PROJECT_ROOT)
    args = parse_args()

    print("=== Student Concentration HAR — Setup ===\n")
    print(f"Project: {PROJECT_ROOT}")

    _check_python()
    _ensure_env_file()
    _load_env()

    if not args.skip_install:
        _pip_install()
    else:
        print("\n[skip] Bỏ qua pip install")

    _download_dataset(skip_download=args.skip_download)

    if not args.skip_smoke:
        _smoke_tests()
    else:
        print("\n[skip] Bỏ qua smoke test")

    _print_next_steps()


if __name__ == "__main__":
    main()
