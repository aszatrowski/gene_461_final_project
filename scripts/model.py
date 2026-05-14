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
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, k in zip(conv_channels, kernel_sizes):
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(4),
            ]
            in_ch = out_ch

        self.conv_blocks = nn.Sequential(*layers)
        flat_dim = conv_channels[-1] * (window_size // (4 ** len(conv_channels)))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_blocks(x))
