"""
Lớp cơ sở cho các mô hình HAR (LRCN, ConvLSTM, MoViNet).

Input chuẩn:  [B, C, T, H, W]
Output chuẩn: [B, num_classes]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch
import torch.nn as nn

ACTIVE_MODELS: tuple[str, ...] = ("lrcn", "convlstm", "movinet")
DEFAULT_NUM_FRAMES: dict[str, int] = {
    "lrcn": 16,
    "convlstm": 16,
    "movinet": 16,
}


class BaseHARModel(nn.Module, ABC):
    """
    Base class cho mọi model nhận diện hành vi video trong project.

    Các model con phải đặt `model_name` và triển khai `forward`.
    """

    model_name: ClassVar[str] = "base"
    default_num_frames: ClassVar[int] = 16

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.num_classes = num_classes

    @staticmethod
    def validate_input(x: torch.Tensor) -> tuple[int, int, int, int, int]:
        """Kiểm tra tensor video [B, C, T, H, W]; trả về shape."""
        if x.dim() != 5:
            raise ValueError(f"Input phải 5D [B,C,T,H,W], nhận shape {tuple(x.shape)}")
        batch_size, channels, num_frames, height, width = x.shape
        return batch_size, channels, num_frames, height, width

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về logits [B, num_classes]."""


def default_num_frames(model_name: str) -> int:
    return DEFAULT_NUM_FRAMES.get(model_name.lower().strip(), 16)


def build_model(
    model_name: str,
    num_classes: int = 5,
    pretrained: bool = True,
) -> BaseHARModel:
    """
    Factory tạo model theo tên.

    Args:
        model_name: lrcn | convlstm | movinet
        num_classes: số lớp hành vi
        pretrained: backbone pretrained (nếu hỗ trợ)
    """
    name = model_name.lower().strip()

    if name == "lrcn":
        from lrcn import LRCN

        return LRCN(num_classes=num_classes, pretrained=pretrained)
    if name == "convlstm":
        from convlstm import ConvLSTM

        return ConvLSTM(num_classes=num_classes, pretrained=pretrained)
    if name == "movinet":
        from movinet import MoViNet

        return MoViNet(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(
        f"Model '{model_name}' không hỗ trợ. Chọn một trong: {', '.join(ACTIVE_MODELS)}"
    )
