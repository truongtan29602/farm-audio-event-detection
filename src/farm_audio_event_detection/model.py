import torch
import torch.nn as nn

class ConvBlock(nn.Module):
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
    return FarmAudioCNN(num_classes=num_classes, dropout=dropout)
