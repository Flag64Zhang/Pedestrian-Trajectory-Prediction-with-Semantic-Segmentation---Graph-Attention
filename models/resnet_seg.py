from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class SceneEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 128,
        backbone: str = "resnet18",
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        try:
            from torchvision import models as tv_models
        except ImportError as exc:
            raise ImportError("torchvision is required for the ResNet scene encoder.") from exc

        # 使用 ResNet 主干提取场景语义特征。
        if backbone == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = tv_models.resnet18(weights=weights)
        elif backbone == "resnet34":
            weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = tv_models.resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 仅保留卷积主干，后接全局池化与线性投影。
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        pooled = self.pool(feat).flatten(1)
        return self.fc(pooled)
