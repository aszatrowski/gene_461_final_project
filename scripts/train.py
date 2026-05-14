"""
train.py — Train the AncestryClassifier CNN on sliding SNP windows.

Designed to run as a standalone script on Google Colab (GPU) or locally,
and as an importable module for W&B hyperparameter sweeps.

Standalone usage:
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

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

sys.path.insert(0, os.path.dirname(__file__))
from model import AncestryClassifier, SUPERPOP_NAMES


# ---------------------------------------------------------------------------
# Architecture encoding
# ---------------------------------------------------------------------------

def parse_conv_arch(arch_str: str) -> tuple[list[int], list[int]]:
    """
    Parse a compact architecture string into channel and kernel lists.

    Format: "c1,c2,..._k1,k2,..."
    Example: "32,64_7,5" -> ([32, 64], [7, 5])
    """
    channels_str, kernels_str = arch_str.split("_")
    channels = [int(x) for x in channels_str.split(",")]
    kernels  = [int(x) for x in kernels_str.split(",")]
    if len(channels) != len(kernels):
        raise ValueError(
            f"conv_arch '{arch_str}': channels ({len(channels)}) and "
            f"kernels ({len(kernels)}) must have the same length."
        )
    return channels, kernels


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
                c: f[f"{c}/genotypes"][ind_mask]
                for c in self.chr_list
            }

        self.window_size = window_size
        self.n_ind = len(self.labels)

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
        i        = int(self._ind[idx])
        c        = int(self._chr[idx])
        s        = int(self._start[idx])
        chr_name = self.chr_list[c]
        x = (
            self.genotypes[chr_name][i, s : s + self.window_size]
            .astype(np.float32) / 2.0
        )
        return (
            torch.from_numpy(x).unsqueeze(0),
            torch.tensor(self.labels[i], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Helpers
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


def evaluate(model, loader, device) -> float:
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
# Core training function (importable by sweep notebook)
# ---------------------------------------------------------------------------

def run_training(cfg: dict, data_h5: str, output_path: str | None, device) -> float:
    """
    Train one model configuration and return the best val accuracy.

    cfg keys:
        window_size, lr, epochs, batch_size, num_workers,
        conv_channels (list[int]), kernel_sizes (list[int]),
        dropout (float), use_wandb (bool)

    output_path=None skips saving the checkpoint (used during sweep trials).
    """
    use_wandb = cfg.get("use_wandb", False) and _WANDB_AVAILABLE

    train_ds = GenomicWindowDataset(data_h5, "train", cfg["window_size"])
    val_ds   = GenomicWindowDataset(data_h5, "val",   cfg["window_size"])
    print(
        f"  train windows: {len(train_ds):,}  |  val windows: {len(val_ds):,}",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        sampler=make_weighted_sampler(train_ds),
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = AncestryClassifier(
        window_size=cfg["window_size"],
        conv_channels=cfg["conv_channels"],
        kernel_sizes=cfg["kernel_sizes"],
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, cfg["epochs"] + 1):
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
            f"Epoch {epoch:3d}/{cfg['epochs']}  "
            f"loss={running_loss/total:.4f}  "
            f"train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}",
            flush=True,
        )

        if use_wandb:
            wandb.log({
                "epoch":     epoch,
                "loss":      running_loss / total,
                "train_acc": train_acc,
                "val_acc":   val_acc,
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if output_path is not None:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                torch.save(
                    {
                        "epoch":         epoch,
                        "state_dict":    model.state_dict(),
                        "val_acc":       val_acc,
                        "window_size":   cfg["window_size"],
                        "conv_channels": cfg["conv_channels"],
                        "kernel_sizes":  cfg["kernel_sizes"],
                        "dropout":       cfg.get("dropout", 0.3),
                    },
                    output_path,
                )
                print(f"  -> saved checkpoint (val_acc={val_acc:.4f})", flush=True)

    if use_wandb:
        wandb.summary["best_val_acc"] = best_val_acc

    print(f"\nBest val accuracy: {best_val_acc:.4f}", flush=True)
    return best_val_acc


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train AncestryClassifier CNN")
    parser.add_argument("--data",        required=True,  help="Path to dataset.h5")
    parser.add_argument("--output",      default="models/best_model.pt")
    parser.add_argument("--conv-arch",   default="32,64_7,5",
                        help='Architecture string, e.g. "32,64_7,5" or "64,128,256_7,5,3"')
    parser.add_argument("--window-size", type=int,   default=500)
    parser.add_argument("--epochs",      type=int,   default=25)
    parser.add_argument("--batch-size",  type=int,   default=512)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--num-workers", type=int,   default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    channels, kernels = parse_conv_arch(args.conv_arch)
    cfg = {
        "window_size":   args.window_size,
        "lr":            args.lr,
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "num_workers":   args.num_workers,
        "conv_channels": channels,
        "kernel_sizes":  kernels,
        "dropout":       args.dropout,
        "use_wandb":     False,
    }

    print("Loading training data ...", flush=True)
    run_training(cfg, args.data, args.output, device)


if __name__ == "__main__":
    main()
