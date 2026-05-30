"""
MoViNet-A0 cho nhận diện hành vi video.

Dùng gói MoViNet-pytorch (movinets). pytorchvideo không còn export movinet_a0.

Yêu cầu:
    pip install git+https://github.com/Atze00/MoViNet-pytorch.git
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_base import BaseHARModel


def _replace_movinets_classifier(
    model: nn.Module,
    num_classes: int,
    *,
    hidden_dim: int,
    tf_like: bool,
) -> None:
    """Thay head Kinetics-600 (600 lớp) bằng head phân loại hành vi."""
    from movinets.models import ConvBlock3D

    model.classifier[-1] = ConvBlock3D(
        hidden_dim,
        num_classes,
        kernel_size=(1, 1, 1),
        padding=(0, 0, 0),
        tf_like=tf_like,
        causal=False,
        conv_type="3d",
        bias=True,
    )


def _load_movinet_model(num_classes: int, pretrained: bool) -> nn.Module:
    try:
        from movinets import MoViNet as MoViNetImpl
        from movinets.config import _C
    except ImportError as exc:
        raise ImportError(
            "Thiếu MoViNet-pytorch. Cài:\n"
            "  pip install git+https://github.com/Atze00/MoViNet-pytorch.git"
        ) from exc

    cfg = _C.MODEL.MoViNetA0
    model = MoViNetImpl(
        cfg,
        causal=False,
        pretrained=pretrained,
        num_classes=num_classes,
        conv_type="3d",
        tf_like=pretrained,
    )
    if pretrained and num_classes != 600:
        _replace_movinets_classifier(
            model,
            num_classes,
            hidden_dim=cfg.dense9.hidden_dim,
            tf_like=True,
        )
    return model


class MoViNet(BaseHARModel):
    """MoViNet-A0 với classifier 5 lớp hành vi."""

    model_name = "movinet"
    default_num_frames = 16

    def __init__(self, num_classes: int = 5, pretrained: bool = True) -> None:
        super().__init__(num_classes=num_classes)
        self.backbone = _load_movinet_model(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        return self.backbone(x)
