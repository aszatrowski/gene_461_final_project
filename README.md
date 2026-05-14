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
  (AFR/AMR/EAS/EUR/SAS). Phased VCFs per chromosome. The NYGC 30x recall is
  also public and gives ~3,202 samples with related individuals.
- **HGDP.** Public — Stanford/CEPH originally; the cleanest modern source is
  the gnomAD HGDP+1kG harmonized callset (gs://gcp-public-data--gnomad and
  via the gnomAD downloads page). ~929 individuals across ~54 populations,
  much finer geographic resolution than 1kG, but small per-population sample
  sizes (often <30) which makes per-population classification brittle.
- Recommended starting point: **HGDP+1kG harmonized callset** (gnomAD v3.1)
  — it solves the joint-calling and label-harmonization problem for free.

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