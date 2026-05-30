"""
LRCN: ResNet18 (frame) + LSTM + classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from model_base import BaseHARModel


class LRCN(BaseHARModel):
    """Long-term Recurrent Convolutional Network."""

    model_name = "lrcn"
    default_num_frames = 16

    def __init__(
        self,
        num_classes: int = 5,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self.hidden_size = hidden_size

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.cnn = backbone

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_frames, height, width = self.validate_input(x)

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size * num_frames, channels, height, width)
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)

        frame_features = self.cnn(x)
        frame_features = frame_features.view(batch_size, num_frames, self.feature_dim)

        lstm_out, _ = self.lstm(frame_features)
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(self.dropout(last_hidden))
