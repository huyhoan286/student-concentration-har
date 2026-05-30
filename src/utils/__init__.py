"""
utils/ — Tiện ích dùng chung.

- constants.py          : hằng số class, model
- dataset.py            : PyTorch Dataset đọc video
- download_dataset.py   : tải dataset từ Kaggle
- training_common.py    : loop train/val
- test_model_forward.py : kiểm tra forward model
- evaluate.py           : đánh giá trên test set
"""

from constants import ACTIVE_MODELS, CLASSES, DEFAULT_NUM_FRAMES, NUM_CLASSES

__all__ = [
    "ACTIVE_MODELS",
    "CLASSES",
    "NUM_CLASSES",
    "DEFAULT_NUM_FRAMES",
]
