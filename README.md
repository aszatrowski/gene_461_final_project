# Deep Learning in Genomics Final Project
*Austin Szatrowski*

## Candidate projects

Two main directions under consideration. Notes below capture the data landscape, a
sketch of the modeling approach, and the main risks for each.

---

### Option 1 — Ancestry classifier on 1kG / HGDP

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

---

### Option 2 — TCR-based antigen recognition prediction

**Clarifying the task.** "Antigen presentation" in the literature usually
means **peptide–MHC binding** (NetMHCpan, MHCflurry) and does not involve
TCRs. The TCR-relevant task is **TCR–pMHC specificity prediction**: given a
TCR sequence (typically the CDR3β, sometimes paired α/β + V/J genes) and a
peptide (often with MHC context), predict whether the TCR recognizes that
pMHC. Assuming this is the intended task.

**Data.**
- **VDJdb** — curated TCR–pMHC pairs, ~100k entries, mostly human, heavily
  skewed toward a few epitopes (flu M1 GILGFVFTL, EBV BMLF1, CMV pp65,
  several HIV/SARS-CoV-2 epitopes).
- **McPAS-TCR** — similar curation, overlapping but not identical entries.
- **IEDB** — large but noisier; needs filtering.
- **10x Genomics dextramer datasets** — paired α/β chains + binding readout
  per cell across a panel of pMHC multimers. Good for paired-chain models.
- **Adaptive ImmuneCODE** (COVID-19) — huge, but binding labels come from
  MIRA assay which is noisy.
- **NetMHCpan training data** — only relevant if the project pivots to
  peptide–MHC binding instead.

**Model sketch.**
- Inputs: CDR3β amino acid sequence (and optionally CDR3α, V/J genes,
  peptide, MHC pseudo-sequence).
- Tokenize at amino acid level; encode with a small Transformer or a
  CNN-over-residues. Recent papers (ERGO-II, TITAN, NetTCR-2.x, pMTnet,
  TCR-BERT, MixTCRpred) are all reasonable reference points for architecture.
- Pretrain options: protein language model embeddings (ESM-2 small) for the
  TCR / peptide; this is a clean ablation to include.
- Loss: binary cross-entropy on (TCR, peptide) pairs with hard negatives
  drawn from other epitopes' TCRs (negative sampling design is the most
  consequential modeling choice in this area).

**Train/test design — this is the trap.**
- Random splits give wildly optimistic numbers because nearly identical TCRs
  appear on both sides.
- The honest split is **by epitope**: hold out entire peptides at test time
  and evaluate on unseen epitopes. State of the art on this split is poor
  (AUC often 0.55–0.65). A project that reports the seen-epitope number
  alone is misleading; one that reports both, with a clear-eyed discussion,
  is publishable-quality framing.
- Cluster TCRs (e.g. by Levenshtein or TCRdist) and split by cluster to
  prevent leakage even within the seen-epitope setting.

**Pros.**
- Open research problem — more "interesting" than ancestry.
- Sequence-modeling task; good vehicle for Transformers / protein LMs.
- Strong alignment with current immunology / immuno-oncology interest.

**Cons / risks.**
- Data is fragmented across sources with overlapping records and
  inconsistent formats. Non-trivial week of just data cleaning.
- Strong epitope imbalance — a handful of epitopes dominate, so the model
  learns "is this a flu-MP TCR" rather than general TCR–peptide binding.
- The honest evaluation (unseen epitopes) is hard, and a negative result is
  the most likely outcome. That can still be a strong project if framed as
  "characterize the generalization gap and what helps close it" rather than
  "build a SOTA predictor."
- Less biological background on your side — adds risk to the timeline.

---

## Recommendation

If the goal is a clean, defensible course project that demonstrates CNN
mechanics and shows up well against principled baselines: **Option 1, scoped
to LAI on simulated admixed individuals using the HGDP+1kG callset.** Lower
risk, controllable scope, easy to write up, and the per-chromosome /
window-size axis gives a real ablation story.

If the goal is to engage with an open problem and you are willing to absorb
extra biology + data-wrangling cost: **Option 2, framed around the
unseen-epitope generalization gap** rather than chasing AUC. This is
genuinely interesting but the most likely "result" is a careful negative
finding — be sure that is acceptable for the class deliverable before
committing.

Open questions to resolve before picking:
- Is the deliverable a paper-style writeup, a poster, or working code?
- Is there a compute budget (single GPU? cluster?)?
- Does the class reward novelty or careful methodology?

## Other ideas (parked)
- Variant effect prediction in a specific cellular context (needs a concrete
  context + readout — too vague as stated).
- CNN for immune activation from pathogen + TCR — same data problems as
  Option 2; not a separate project.
