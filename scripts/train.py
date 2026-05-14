"""
train.py — Train the AncestryClassifier CNN on sliding SNP windows.

Designed to run as a standalone script on Google Colab (GPU) or locally.
Data is loaded once into RAM from an HDF5 file produced by prepare_data.py.

Usage:
    python train.py --data data/dataset.h5 --output models/best_model.pt

On Colab:
    !python scripts/train.py \\
        --data /content/drive/MyDrive/gene461/dataset.h5 \\
        --output /content/drive/MyDrive/gene461/checkpoints/best_model.pt
"""

import argparse
import os
import sys
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

# Allow importing model.py from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from model import AncestryClassifier, SUPERPOP_NAMES


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GenomicWindowDataset(Dataset):
    """
    Generates non-overlapping windows on-the-fly from a pre-loaded genotype matrix.

    The full genotype data is loaded into RAM in __init__. On Linux (Colab),
    forked DataLoader workers share this memory copy-on-write, so RAM usage
    does not multiply with num_workers.
    """

    def __init__(self, h5_path: str, split: str, window_size: int = 500):
        with h5py.File(h5_path, "r") as f:
            splits = f["splits"][:].astype(str)
            ind_mask = splits == split
            self.labels = f["superpop_labels"][ind_mask].astype(np.int64)
            self.chr_list = sorted(k for k in f.keys() if k.startswith("chr"))
            self.genotypes = {
                c: f[f"{c}/genotypes"][ind_mask]  # (n_ind, n_snps), int8
                for c in self.chr_list
            }

        self.window_size = window_size
        self.n_ind = len(self.labels)

        # Pre-compute flat index arrays (very compact: ~28 MB for 4 M windows)
        ind_list, chr_list, start_list = [], [], []
        for c_idx, chr_name in enumerate(self.chr_list):
            n_snps    = self.genotypes[chr_name].shape[1]
            n_windows = n_snps // window_size
            for i in range(self.n_ind):
                for w in range(n_windows):
                    ind_list.append(i)
                    chr_list.append(c_idx)
                    start_list.append(w * window_size)

        self._ind   = np.array(ind_list,   dtype=np.int16)
        self._chr   = np.array(chr_list,   dtype=np.int8)
        self._start = np.array(start_list, dtype=np.int32)

    def __len__(self) -> int:
        return len(self._ind)

    def __getitem__(self, idx):
        i   = int(self._ind[idx])
        c   = int(self._chr[idx])
        s   = int(self._start[idx])
        chr_name = self.chr_list[c]
        x = (
            self.genotypes[chr_name][i, s : s + self.window_size]
            .astype(np.float32) / 2.0          # map 0/1/2 → 0.0/0.5/1.0
        )
        return (
            torch.from_numpy(x).unsqueeze(0),  # (1, window_size)
            torch.tensor(self.labels[i], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def make_weighted_sampler(dataset: GenomicWindowDataset) -> WeightedRandomSampler:
    window_labels = dataset.labels[dataset._ind.astype(np.int32)]
    class_counts  = np.bincount(window_labels, minlength=len(SUPERPOP_NAMES)).astype(float)
    class_weights = 1.0 / np.where(class_counts > 0, class_counts, 1.0)
    sample_weights = class_weights[window_labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(dataset),
        replacement=True,
    )


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train AncestryClassifier CNN")
    parser.add_argument("--data",        required=True,       help="Path to dataset.h5")
    parser.add_argument("--output",      default="models/best_model.pt")
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--epochs",      type=int, default=25)
    parser.add_argument("--batch-size",  type=int, default=512)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Datasets
    print("Loading training data ...", flush=True)
    train_ds = GenomicWindowDataset(args.data, "train", args.window_size)
    val_ds   = GenomicWindowDataset(args.data, "val",   args.window_size)
    print(
        f"  train windows: {len(train_ds):,}  |  val windows: {len(val_ds):,}",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=make_weighted_sampler(train_ds),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Model
    model = AncestryClassifier(window_size=args.window_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = correct = total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            correct      += (logits.argmax(1) == y).sum().item()
            total        += y.size(0)

        scheduler.step()
        train_acc = correct / total
        val_acc   = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={running_loss/total:.4f}  "
            f"train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch":       epoch,
                    "state_dict":  model.state_dict(),
                    "val_acc":     val_acc,
                    "window_size": args.window_size,
                },
                args.output,
            )
            print(f"  -> saved checkpoint (val_acc={val_acc:.4f})", flush=True)

    print(f"\nBest val accuracy: {best_val_acc:.4f}", flush=True)
    print(f"Checkpoint: {args.output}", flush=True)


if __name__ == "__main__":
    main()
