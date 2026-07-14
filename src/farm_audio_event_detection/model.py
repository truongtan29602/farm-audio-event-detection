"""CNN model for farm audio event classification.

Architecture
------------
Input  : (B, 3, 128, 250)   -- 3-channel Log-Mel spectrogram
Output : (B, 6)             -- logits for 6 classes

Design choices
--------------
* Four convolutional blocks with BatchNorm and ReLU, each followed by 2x2
  max-pooling.  The number of filters doubles each block (32 → 64 → 128 → 256)
  to capture increasingly abstract patterns.
* Global Average Pooling replaces a large fully-connected layer, keeping the
  parameter count low and reducing overfitting on this ~200-sample dataset.
* A two-layer classifier head (256 → 128 → 6) with dropout(0.4) gives the
  model enough capacity while discouraging co-adaptation of features.

Classes (label_index order)
----------------------------
    0  dog
    1  cat
    2  cow
    3  rooster
    4  sheep
    5  others
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU → MaxPool2d."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FarmAudioCNN(nn.Module):
    """Small CNN for 3-channel Log-Mel spectrogram classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 6).
    dropout : float
        Dropout probability in the classifier head (default 0.4).
    """

    def __init__(self, num_classes: int = 6, dropout: float = 0.4) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),    # (B, 32, 64, 125)
            ConvBlock(32, 64),   # (B, 64, 32,  62)
            ConvBlock(64, 128),  # (B, 128,16,  31)
            ConvBlock(128, 256), # (B, 256, 8,  15)
        )

        # Global Average Pooling: (B, 256, H, W) → (B, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


def build_model(num_classes: int = 6, dropout: float = 0.4) -> FarmAudioCNN:
    """Construct and return a new FarmAudioCNN instance."""
    return FarmAudioCNN(num_classes=num_classes, dropout=dropout)
