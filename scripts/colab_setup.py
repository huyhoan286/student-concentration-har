"""
Bootstrap project trên Google Colab.

Chạy từ thư mục gốc project (sau clone hoặc %cd):
    python scripts/colab_setup.py

Colab Secrets (🔑): KAGGLE_API_TOKEN, KAGGLE_DATASET_SLUG
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = PROJECT_ROOT / "src" / "utils"


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401, PLC0415

        return True
    except ImportError:
        return bool(os.environ.get("COLAB_RELEASE_TAG"))


def load_colab_secrets() -> dict[str, str]:
    """Đọc Kaggle config từ Colab userdata secrets."""
    loaded: dict[str, str] = {}
    if not is_colab():
        return loaded

    try:
        from google.colab import userdata  # noqa: PLC0415

        for key in ("KAGGLE_API_TOKEN", "KAGGLE_DATASET_SLUG"):
            try:
                value = userdata.get(key).strip()
                if value:
                    os.environ[key] = value
                    loaded[key] = value
            except userdata.SecretNotFoundError:
                pass
    except Exception as exc:
        print(f"[cảnh báo] Không đọc được Colab secrets: {exc}")

    return loaded


def check_gpu() -> None:
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU] {name} ({mem_gb:.1f} GB)")
    else:
        print(
            "[cảnh báo] Không có GPU. Vào Runtime → Change runtime type → T4 GPU."
        )


def pip_install(*, skip: bool = False) -> None:
    if skip:
        print("[skip] pip install")
        return
    req = PROJECT_ROOT / "requirements.txt"
    print(f">>> pip install -r {req}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
        cwd=PROJECT_ROOT,
    )


def download_data(*, skip: bool = False) -> None:
    if skip:
        print("[skip] download dataset")
        return

    sys.path.insert(0, str(UTILS_DIR))
    from env_loader import load_project_env  # noqa: PLC0415
    from download_dataset import download_dataset  # noqa: PLC0415

    load_project_env(PROJECT_ROOT)
    if download_dataset(PROJECT_ROOT, check_only=True):
        print("[OK] dataset/ đã sẵn sàng")
        return
    if not os.environ.get("KAGGLE_DATASET_SLUG", "").strip():
        raise RuntimeError(
            "Thiếu dataset và chưa có KAGGLE_DATASET_SLUG.\n"
            "Thêm Colab Secret hoặc sửa configs/dataset_config.yaml"
        )
    download_dataset(PROJECT_ROOT)


def smoke_test() -> None:
    script = UTILS_DIR / "test_model_forward.py"
    for model in ("lrcn", "convlstm"):
        subprocess.check_call(
            [sys.executable, str(script), "--model", model],
            cwd=PROJECT_ROOT,
        )
    try:
        subprocess.check_call(
            [sys.executable, str(script), "--model", "movinet"],
            cwd=PROJECT_ROOT,
        )
    except subprocess.CalledProcessError:
        print("[cảnh báo] MoViNet smoke test thất bại — bỏ qua nếu chưa cần.")


def bootstrap(
    *,
    skip_install: bool = False,
    skip_download: bool = False,
    skip_smoke: bool = False,
) -> Path:
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(UTILS_DIR))
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

    print("=== Colab bootstrap ===")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Colab:   {is_colab()}")

    secrets = load_colab_secrets()
    if secrets:
        print(f"Secrets: {', '.join(secrets.keys())}")

    check_gpu()
    pip_install(skip=skip_install)
    download_data(skip=skip_download)

    if not skip_smoke:
        smoke_test()

    print("\n[OK] Bootstrap xong. Tiếp theo: train / evaluate.")
    return PROJECT_ROOT


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Bootstrap Google Colab")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    bootstrap(
        skip_install=args.skip_install,
        skip_download=args.skip_download,
        skip_smoke=args.skip_smoke,
    )


if __name__ == "__main__":
    main()
