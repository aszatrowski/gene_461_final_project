# Deep Learning in Genomics Final Project: Ancestry classifier on 1kG
*Austin Szatrowski*


**Task.** Given a phased haplotype (or genotype) window, predict the source
population. Two natural framings:
1. *Global ancestry*: one label per individual (super-population or population).
2. *Local ancestry inference (LAI)*: per-window labels along a chromosome,
   useful for admixed individuals. More interesting, more standard ML setup
   (sliding windows → labels), and lets the model demonstrate something PCA
   cannot do trivially.

**Data.**
- **1000 Genomes (1kG).** Public FTP at `ftp.1000genomes.ebi.ac.uk`. Phase 3
  release: ~2,504 individuals, 26 populations, 5 super-populations
  (AFR/AMR/EAS/EUR/SAS). Phased VCFs per chromosome. 

**Model sketch.**
- Encode each SNP as a 0/1 (haploid) or 0/1/2 dosage; alternatively one-hot
  on {ref-hom, het, alt-hom, missing}. For LAI, work on phased haplotypes.
- 1D CNN over windows of ~500–10,000 SNPs. Stack a few conv blocks + global
  pool + linear head. Per-chromosome models, then ensemble / majority-vote
  across chromosomes for global ancestry; or a single model that sees
  chromosome-stratified windows.
- Baselines: (a) PCA + linear classifier, (b) ADMIXTURE for global,
  (c) RFMix for LAI. These are essential — without them the result is
  uninterpretable for the class.
- Stretch: compare with a small Transformer on the same windows; ablate
  window size; measure calibration on held-out populations.

**Train/test design.**
- For global ancestry, hold out *individuals* (never windows from the same
  individual on both sides). Stratify by population.
- For LAI, the principled split is by chromosome — train on chr1–20, validate
  on chr21, test on chr22 — so the model cannot memorize specific haplotype
  blocks. This also keeps things tractable on a single GPU.
- Simulate admixed test individuals from unmixed parents (standard in the LAI
  literature, e.g. via `admix-simu` or by hand) so there is a clean LAI
  evaluation set with ground-truth tracts.

**Pros.**
- Data is clean, well-documented, and trivially reproducible.
- Easy to write up: clear baselines, clear metrics (accuracy, F1, tract
  boundary error for LAI).
- CNN architecture choices (window size, dilations, receptive field) map
  naturally to a biological story about LD block length.
- Compute is modest — fits on a laptop GPU per chromosome.

**Cons / risks.**
- Global ancestry on 1kG super-populations is essentially solved; a CNN will
  hit ~100% and there is no story. **Mitigation:** target HGDP fine-grained
  populations or LAI on simulated admixed individuals — both are genuinely
  harder.
- Class imbalance in HGDP (some pops have <15 samples). **Mitigation:** group
  rare populations to regional labels, or use balanced sampling + report
  per-class metrics.
- Big windows = big input tensors; needs care with batching and possibly
  variant pruning (LD-prune or MAF filter) before training.

## Project Notes
* `prepare_data.py`: 00:18:10 core-walltime on chr21, 23.85 GB used

## Training log
First pass:
```
Loaded checkpoint from epoch 25 (val_acc=0.5696)
Evaluating on test-split windows ...
Test window accuracy: 0.5681
              precision    recall  f1-score   support

         AFR       0.83      0.87      0.85     33660
         AMR       0.31      0.19      0.24     17680
         EAS       0.60      0.69      0.64     25840
         EUR       0.46      0.54      0.50     25840
         SAS       0.38      0.32      0.35     24820

    accuracy                           0.57    127840
   macro avg       0.52      0.52      0.52    127840
weighted avg       0.55      0.57      0.56    127840

Saved confusion matrix → /content/drive/MyDrive/gene461/confusion_matrix.png
Generating LAI karyogram ...
Saved LAI karyogram → /content/drive/MyDrive/gene461/lai_karyogram.png
```

best from `wandb` runs:
`balmy-sweep-16`:
```
{
  "lr": {
    "value": 0.001301064979142684
  },
  "_wandb": {
    "value": {
      "m": [],
      "t": {
        "1": [
          1
        ],
        "2": [
          1
        ],
        "3": [
          2,
          14,
          62
        ],
        "4": "3.12.13",
        "5": "0.26.1",
        "8": [
          1,
          12
        ],
        "12": "0.26.1",
        "13": "linux-x86_64"
      },
      "cli_version": "0.26.1",
      "python_version": "3.12.13"
    }
  },
  "dropout": {
    "value": 0.1378284030110427
  },
  "conv_arch": {
    "value": "32,64_7,7_1,4"
  },
  "global_pool": {
    "value": false
  },
  "window_size": {
    "value": 2000
  }
}
```

`lively-sweep-17` was better on accuracy but doesn't have dilations that will surely be useful going forward

new best, 23:25 (usual-sweep):
```
{
  "lr": {
    "value": 0.0001798367708675064
  },
  "_wandb": {
    "value": {
      "m": [],
      "t": {
        "1": [
          1
        ],
        "2": [
          1
        ],
        "3": [
          2,
          14,
          62
        ],
        "4": "3.12.13",
        "5": "0.26.1",
        "8": [
          1,
          12
        ],
        "12": "0.26.1",
        "13": "linux-x86_64"
      },
      "cli_version": "0.26.1",
      "python_version": "3.12.13"
    }
  },
  "dropout": {
    "value": 0.4734405376290487
  },
  "conv_arch": {
    "value": "32,64,128_7,7,7_1,4,16"
  },
  "global_pool": {
    "value": false
  },
  "window_size": {
    "value": 5000
  }
}
```

New best (`val_acc = 0.81`):
```
{
  "lr": {
    "value": 0.0002963535898395063
  },
  "_wandb": {
    "value": {
      "m": [],
      "t": {
        "1": [
          1
        ],
        "2": [
          1
        ],
        "3": [
          2,
          14,
          62
        ],
        "4": "3.12.13",
        "5": "0.26.1",
        "8": [
          1,
          12
        ],
        "12": "0.26.1",
        "13": "linux-x86_64"
      },
      "cli_version": "0.26.1",
      "python_version": "3.12.13"
    }
  },
  "dropout": {
    "value": 0.10330932751211558
  },
  "conv_arch": {
    "value": "64,128,256_7,7,7_1,8,32"
  },
  "global_pool": {
    "value": false
  },
  "window_size": {
    "value": 5000
  }
}
```

* For LAI test, might want to try both 5kb, 10kb, and 20kb models, since 20 might be too long for local ancestry.
* Also of note: 32-64 with no dilations is stubbornly good (though only with window of 20k). These are much more lightweight and train in about half the time. 
  * here, the final linear layer is doing the spatial integration, which has an ERF of the whole window (all the regular convolutions!)
* For repro:
  * save the checkpoints to the repo
  * snakemake: download data, preprocess, to HDF5, load checkpoints, forward pass, produce outputs

## Remaining Todos:
- [ ] figure out LAI test and whether it makes sense to present it
- [ ] upload best model to project dir on Midway
- [ ] integrate best model forward pass with snakemake