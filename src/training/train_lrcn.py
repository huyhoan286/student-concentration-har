"""
Huấn luyện LRCN.

Chạy từ thư mục gốc project:
    python src/training/train_lrcn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UTILS_DIR = _PROJECT_ROOT / "src" / "utils"
_MODELS_DIR = _PROJECT_ROOT / "src" / "models"

for path in (_UTILS_DIR, _MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lrcn import LRCN  # noqa: E402
from training_common import TrainConfig, run_training  # noqa: E402

DEBUG = False

CONFIG = TrainConfig(
    model_name="lrcn",
    num_frames=16,
    epochs=5,
    debug=DEBUG,
    # batch_size, num_workers, AMP: tự tối ưu khi có CUDA (gpu_runtime.py)
)


def main() -> None:
    run_training(CONFIG, lambda: LRCN(num_classes=CONFIG.num_classes, pretrained=True))


if __name__ == "__main__":
    main()
