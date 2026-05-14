import torch
import torch.nn as nn


SUPERPOP_NAMES = {0: "AFR", 1: "AMR", 2: "EAS", 3: "EUR", 4: "SAS"}
N_CLASSES = len(SUPERPOP_NAMES)


class AncestryClassifier(nn.Module):
    def __init__(self, window_size: int = 500, n_classes: int = N_CLASSES):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Block 1 — short-range LD patterns (~7 SNP receptive field)
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 500 → 125

            # Block 2 — broader haplotype structure
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 125 → 31
        )
        flat_dim = 64 * (window_size // 16)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_blocks(x))
