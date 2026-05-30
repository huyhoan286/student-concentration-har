"""
Kiểm tra forward pass LRCN / ConvLSTM / MoViNet (chưa train).

Chạy từ thư mục gốc project:
    python src/utils/test_model_forward.py
    python src/utils/test_model_forward.py --model lrcn
    python src/utils/test_model_forward.py --model all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _PROJECT_ROOT / "src" / "models"

if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from model_base import ACTIVE_MODELS, build_model, default_num_frames  # noqa: E402

BATCH_SIZE = 2
NUM_CLASSES = 5
HEIGHT = WIDTH = 224


def test_one(model_name: str, device: torch.device) -> None:
    num_frames = default_num_frames(model_name)
    x = torch.randn(BATCH_SIZE, 3, num_frames, HEIGHT, WIDTH, device=device)

    model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=True).to(device)
    model.eval()

    with torch.no_grad():
        logits = model(x)

    expected = (BATCH_SIZE, NUM_CLASSES)
    print(f"=== {model_name.upper()} ===")
    print(f"Device: {device}")
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")

    if tuple(logits.shape) != expected:
        raise AssertionError(f"{model_name}: mong đợi {expected}, nhận {tuple(logits.shape)}")

    print(f"[OK] {model_name.upper()} forward pass thành công.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test forward pass HAR models")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help=f"lrcn | convlstm | movinet | all (mặc định: all). Hỗ trợ: {', '.join(ACTIVE_MODELS)}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model.lower() == "all":
        names = list(ACTIVE_MODELS)
    else:
        names = [args.model.lower().strip()]

    for name in names:
        if name not in ACTIVE_MODELS:
            raise ValueError(f"Model không hỗ trợ: {name}. Chọn: {', '.join(ACTIVE_MODELS)}")
        test_one(name, device)


if __name__ == "__main__":
    main()
