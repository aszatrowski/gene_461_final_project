import torch
import torch.nn as nn
from typing import Sequence


SUPERPOP_NAMES = {0: "AFR", 1: "AMR", 2: "EAS", 3: "EUR", 4: "SAS"}
N_CLASSES = len(SUPERPOP_NAMES)


class AncestryClassifier(nn.Module):
    def __init__(
        self,
        window_size: int = 500,
        n_classes: int = N_CLASSES,
        conv_channels: Sequence[int] = (32, 64),
        kernel_sizes: Sequence[int] = (7, 5),
        dilation_rates: Sequence[int] | None = None,  # None → all 1s (no dilation)
        dropout: float = 0.3,
        global_pool: bool = False,  # True → AdaptiveAvgPool1d(1), decouples from window_size
    ):
        super().__init__()
        dils = list(dilation_rates) if dilation_rates is not None else [1] * len(conv_channels)
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, k, d in zip(conv_channels, kernel_sizes, dils):
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=d * (k // 2), dilation=d),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(4),
            ]
            in_ch = out_ch

        self.conv_blocks = nn.Sequential(*layers)

        if global_pool:
            self.pool = nn.AdaptiveAvgPool1d(1)
            flat_dim = conv_channels[-1]
        else:
            self.pool = None
            flat_dim = conv_channels[-1] * (window_size // (4 ** len(conv_channels)))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        if self.pool is not None:
            x = self.pool(x)
        return self.classifier(x)
