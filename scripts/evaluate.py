"""
evaluate.py — Evaluate a trained AncestryClassifier.

Produces:
  1. Confusion matrix on held-out test-split windows.
  2. LAI karyogram: per-window ancestry probabilities on simulated admixed
     individuals, with ground-truth tract boundaries overlaid.

Can be run via Snakemake (uses snakemake namespace) or standalone:
    python scripts/evaluate.py \\
        --data data/dataset.h5 \\
        --admixed data/admixed_test.h5 \\
        --checkpoint models/best_model.pt \\
        --confusion outputs/confusion_matrix.png \\
        --karyogram outputs/lai_karyogram.png
"""

import argparse
import os
import sys
import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from model import AncestryClassifier, SUPERPOP_NAMES, N_CLASSES

SUPERPOP_COLORS = {
    0: "#E41A1C",  # AFR — red
    1: "#FF7F00",  # AMR — orange
    2: "#4DAF4A",  # EAS — green
    3: "#377EB8",  # EUR — blue
    4: "#984EA3",  # SAS — purple
}

# ---------------------------------------------------------------------------
# Argument / Snakemake interface
# ---------------------------------------------------------------------------
try:
    data_path      = snakemake.input.dataset
    admixed_path   = snakemake.input.admixed
    ckpt_path      = snakemake.input.checkpoint
    confusion_out  = snakemake.output.confusion
    karyogram_out  = snakemake.output.karyogram
    f1_bar_out     = snakemake.output.f1_bar
    window_size    = snakemake.config.get("window_size", 500)
except NameError:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True)
    parser.add_argument("--admixed",    required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--confusion",  default="outputs/confusion_matrix.png")
    parser.add_argument("--karyogram",  default="outputs/lai_karyogram.png")
    parser.add_argument("--f1-bar",     default="outputs/f1_bar.png")
    parser.add_argument("--window-size", type=int, default=500)
    args = parser.parse_args()
    data_path     = args.data
    admixed_path  = args.admixed
    ckpt_path     = args.checkpoint
    confusion_out = args.confusion
    karyogram_out = args.karyogram
    f1_bar_out    = args.f1_bar
    window_size   = args.window_size

for out_path in (confusion_out, karyogram_out, f1_bar_out):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt   = torch.load(ckpt_path, map_location=device)
model  = AncestryClassifier(
    window_size    = ckpt.get("window_size",    window_size),
    conv_channels  = ckpt.get("conv_channels",  (32, 64)),
    kernel_sizes   = ckpt.get("kernel_sizes",   (7, 5)),
    dilation_rates = ckpt.get("dilation_rates", None),
    dropout        = ckpt.get("dropout",        0.3),
    global_pool    = ckpt.get("global_pool",    False),
).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

# ---------------------------------------------------------------------------
# Helper: run CNN on a 1-D genotype array in sliding-window mode
# ---------------------------------------------------------------------------
@torch.no_grad()
def sliding_window_probs(geno: np.ndarray, stride: int) -> np.ndarray:
    """
    geno   : (n_snps,) int8 dosage array
    stride : step between windows (use window_size//4 for smooth LAI)
    returns: (n_windows, n_classes) softmax probabilities
    """
    n_snps   = len(geno)
    starts   = list(range(0, n_snps - window_size + 1, stride))
    windows  = np.stack([
        geno[s : s + window_size].astype(np.float32) / 2.0
        for s in starts
    ])                               # (n_windows, window_size)
    x = torch.from_numpy(windows).unsqueeze(1).to(device)  # (N, 1, W)
    logits = model(x)
    return torch.softmax(logits, dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# 1.  Confusion matrix on test-split windows
# ---------------------------------------------------------------------------
print("Evaluating on test-split windows ...", flush=True)

with h5py.File(data_path, "r") as f:
    splits      = f["splits"][:].astype(str)
    labels_all  = f["superpop_labels"][:]
    test_mask   = splits == "test"
    test_labels = labels_all[test_mask]
    chrs        = sorted(k for k in f.keys() if k.startswith("chr"))
    test_genos  = {c: f[f"{c}/genotypes"][test_mask] for c in chrs}

all_preds, all_true = [], []

for chr_name in chrs:
    geno_mat = test_genos[chr_name]   # (n_test_ind, n_snps)
    n_ind, n_snps = geno_mat.shape
    n_windows = n_snps // window_size

    for i in range(n_ind):
        for w in range(n_windows):
            s = w * window_size
            x = (
                torch.from_numpy(
                    geno_mat[i, s : s + window_size].astype(np.float32) / 2.0
                )
                .unsqueeze(0).unsqueeze(0)   # (1, 1, W)
                .to(device)
            )
            with torch.no_grad():
                pred = model(x).argmax(dim=1).item()
            all_preds.append(pred)
            all_true.append(test_labels[i])

all_preds = np.array(all_preds)
all_true  = np.array(all_true)
acc       = (all_preds == all_true).mean()
print(f"Test window accuracy: {acc:.4f}", flush=True)
report = classification_report(
    all_true, all_preds,
    target_names=[SUPERPOP_NAMES[i] for i in range(N_CLASSES)],
    output_dict=True,
)
print(classification_report(
    all_true, all_preds,
    target_names=[SUPERPOP_NAMES[i] for i in range(N_CLASSES)],
))

cm = confusion_matrix(all_true, all_preds, labels=list(range(N_CLASSES)))
pop_names = [SUPERPOP_NAMES[i] for i in range(N_CLASSES)]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(pop_names)
ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(pop_names)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Confusion matrix — test windows (acc={acc:.3f})")
plt.colorbar(im, ax=ax)
thresh = cm.max() / 2.0
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=9)
fig.tight_layout()
fig.savefig(confusion_out, dpi=150)
plt.close(fig)
print(f"Saved confusion matrix → {confusion_out}", flush=True)

pop_names  = [SUPERPOP_NAMES[i] for i in range(N_CLASSES)]
f1_scores  = [report[name]["f1-score"] for name in pop_names]
colors     = [SUPERPOP_COLORS[i] for i in range(N_CLASSES)]

fig, ax = plt.subplots(figsize=(4, 10))
ax.barh(pop_names, f1_scores, color=colors)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("F1 score")
ax.set_title(f"Per-population F1\n(acc={acc:.3f})")
for i, v in enumerate(f1_scores):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
fig.tight_layout()
fig.savefig(f1_bar_out, dpi=150)
plt.close(fig)
print(f"Saved F1 bar plot → {f1_bar_out}", flush=True)

# ---------------------------------------------------------------------------
# 2.  LAI karyogram on simulated admixed individuals
# ---------------------------------------------------------------------------
print("Generating LAI karyogram ...", flush=True)

with h5py.File(data_path, "r") as f:
    positions = {c: f[f"{c}/positions"][:] for c in chrs}

with h5py.File(admixed_path, "r") as f:
    parent_pops = f["parent_pops"][:]
    admixed_chrs = sorted(k for k in f.keys() if k.startswith("chr"))
    admixed_genos  = {c: f[f"{c}/genotypes"][:]   for c in admixed_chrs}
    admixed_tracts = {c: f[f"{c}/tract_labels"][:] for c in admixed_chrs}

n_admixed = parent_pops.shape[0]
n_show    = min(6, n_admixed)           # plot at most 6 examples
stride    = window_size // 4            # overlapping for smooth predictions

fig, axes = plt.subplots(n_show, len(admixed_chrs), figsize=(5 * len(admixed_chrs), 3 * n_show),
                         squeeze=False)

for row, ind_idx in enumerate(range(n_show)):
    pop_a, pop_b = int(parent_pops[ind_idx, 0]), int(parent_pops[ind_idx, 1])
    title = f"Simulated {SUPERPOP_NAMES[pop_a]}+{SUPERPOP_NAMES[pop_b]}"

    for col, chr_name in enumerate(admixed_chrs):
        ax = axes[row][col]
        geno  = admixed_genos[chr_name][ind_idx]    # (n_snps,)
        tract = admixed_tracts[chr_name][ind_idx]   # (n_snps,)
        pos   = positions[chr_name]                 # (n_snps,) in bp

        probs = sliding_window_probs(geno, stride)  # (n_windows, 5)
        n_snps = len(geno)
        starts = list(range(0, n_snps - window_size + 1, stride))
        window_mid_pos = pos[np.array(starts) + window_size // 2] / 1e6  # → Mb

        # Plot per-class probability curves
        for cls_idx in range(N_CLASSES):
            ax.plot(
                window_mid_pos, probs[:, cls_idx],
                color=SUPERPOP_COLORS[cls_idx],
                lw=1.2, alpha=0.85,
                label=SUPERPOP_NAMES[cls_idx],
            )

        # Shade ground-truth ancestry tracts
        crossover_snp = np.searchsorted(tract, pop_b, side="left")  # first SNP of second tract
        if 0 < crossover_snp < n_snps:
            crossover_mb = pos[crossover_snp] / 1e6
            ax.axvspan(pos[0] / 1e6, crossover_mb,
                       color=SUPERPOP_COLORS[pop_a], alpha=0.08)
            ax.axvspan(crossover_mb, pos[-1] / 1e6,
                       color=SUPERPOP_COLORS[pop_b], alpha=0.08)
            ax.axvline(crossover_mb, color="black", lw=1.0, ls="--", label="crossover")

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Position (Mb)")
        ax.set_ylabel("P(ancestry)")
        if col == 0:
            ax.set_title(f"{title}\n{chr_name}", fontsize=9)
        else:
            ax.set_title(chr_name, fontsize=9)

        if row == 0 and col == len(admixed_chrs) - 1:
            handles = [
                mpatches.Patch(color=SUPERPOP_COLORS[i], label=SUPERPOP_NAMES[i])
                for i in range(N_CLASSES)
            ]
            ax.legend(handles=handles, fontsize=7, loc="upper right")

fig.suptitle("Local ancestry inference — simulated admixed individuals", y=1.01)
fig.tight_layout()
fig.savefig(karyogram_out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved LAI karyogram → {karyogram_out}", flush=True)
