"""
MoViNet-A0 (PyTorchVideo) cho nhận diện hành vi video.

Yêu cầu: pip install pytorchvideo
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_base import BaseHARModel


def _load_movinet_backbone(pretrained: bool) -> nn.Module:
    try:
        from pytorchvideo.models.hub import movinet_a0
    except ImportError as exc:
        raise ImportError("Thiếu pytorchvideo. Cài: pip install pytorchvideo") from exc
    return movinet_a0(pretrained=pretrained)


def _replace_classifier_head(backbone: nn.Module, num_classes: int) -> None:
    head = backbone.head
    if not hasattr(head, "proj"):
        raise RuntimeError(
            "MoViNet head không có 'proj' — kiểm tra phiên bản pytorchvideo."
        )
    in_features = head.proj.in_features
    head.proj = nn.Linear(in_features, num_classes)


class MoViNet(BaseHARModel):
    """MoViNet-A0 với classifier 5 lớp hành vi."""

    model_name = "movinet"
    default_num_frames = 16

    def __init__(self, num_classes: int = 5, pretrained: bool = True) -> None:
        super().__init__(num_classes=num_classes)
        self.backbone = _load_movinet_backbone(pretrained=pretrained)
        _replace_classifier_head(self.backbone, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        return self.backbone(x)
