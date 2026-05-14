"""
prepare_data.py — Convert exported plink genotype matrices to a training HDF5.

Called by Snakemake; expects the following in the snakemake namespace:
    snakemake.input.raw     — list of .raw dosage files (one per chromosome)
    snakemake.input.pvar    — list of filtered .pvar files (same SNP order as .raw)
    snakemake.input.samples — 1kG phase3 sample/population sheet
    snakemake.output[0]     — path for the output HDF5
    snakemake.config        — dict with val_frac, test_frac
"""

import sys
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import StratifiedShuffleSplit

SUPERPOP_LABELS = {"AFR": 0, "AMR": 1, "EAS": 2, "EUR": 3, "SAS": 4}

raw_files    = list(snakemake.input.raw)
pvar_files   = list(snakemake.input.pvar)
samples_file = snakemake.input.samples
out_h5       = snakemake.output[0]
chrs         = snakemake.config["chrs"]
val_frac     = snakemake.config.get("val_frac",  0.15)
test_frac    = snakemake.config.get("test_frac", 0.15)

# ---------------------------------------------------------------------------
# Load population labels
# ---------------------------------------------------------------------------
samples_df = pd.read_csv(samples_file, sep="\t", usecols=["sample", "super_pop"])
sample_to_superpop = dict(zip(samples_df["sample"], samples_df["super_pop"]))

# ---------------------------------------------------------------------------
# Read each chromosome
# ---------------------------------------------------------------------------
chr_data = {}  # chr_name -> {"iids": ndarray, "geno": ndarray (int8), "positions": ndarray}

for chr_name, raw_path, pvar_path in zip(chrs, raw_files, pvar_files):
    print(f"[{chr_name}] reading {raw_path} ...", flush=True)

    # SNP positions from the MAF-filtered pvar (comment='#' skips the #CHROM header)
    pvar = pd.read_csv(
        pvar_path, sep="\t", comment="#",
        header=None, usecols=[1], names=["pos"],
        dtype={"pos": np.int64},
    )
    positions = pvar["pos"].values
    n_snps_pvar = len(positions)

    # Dosage matrix: rows = individuals, cols = SNPs (first 6 columns are metadata)
    raw = pd.read_csv(raw_path, sep="\t", na_values="NA")
    iids = raw["IID"].values

    geno_raw = raw.iloc[:, 6:].values   # float64 with NaN for missing
    n_ind, n_snps_raw = geno_raw.shape

    if n_snps_raw != n_snps_pvar:
        raise ValueError(
            f"{chr_name}: .raw has {n_snps_raw} SNPs but .pvar has {n_snps_pvar}. "
            "Ensure --make-just-pvar and --export A use the same --maf filter."
        )

    # Fill missing dosages with 1 (het ≈ population mean for a biallelic locus)
    geno_raw = np.where(np.isnan(geno_raw), 1.0, geno_raw)
    geno = geno_raw.astype(np.int8)

    chr_data[chr_name] = {"iids": iids, "geno": geno, "positions": positions}
    print(f"[{chr_name}] {n_ind} individuals, {n_snps_raw} SNPs", flush=True)

# ---------------------------------------------------------------------------
# Verify individual order is consistent across chromosomes
# ---------------------------------------------------------------------------
ref_iids = chr_data[chrs[0]]["iids"]
for chr_name in chrs[1:]:
    if not np.array_equal(ref_iids, chr_data[chr_name]["iids"]):
        raise ValueError(f"Individual order differs between {chrs[0]} and {chr_name}.")
iids = ref_iids

# ---------------------------------------------------------------------------
# Assign super-population labels; drop individuals without a label
# ---------------------------------------------------------------------------
superpop_labels = np.array(
    [SUPERPOP_LABELS.get(sample_to_superpop.get(iid, ""), -1) for iid in iids],
    dtype=np.int8,
)
valid = superpop_labels >= 0
if not valid.all():
    n_dropped = (~valid).sum()
    print(f"Dropping {n_dropped} individuals with no population label.", flush=True)
    iids = iids[valid]
    superpop_labels = superpop_labels[valid]
    for chr_name in chrs:
        chr_data[chr_name]["geno"] = chr_data[chr_name]["geno"][valid]

n_ind = len(iids)
print(f"Total: {n_ind} labelled individuals", flush=True)

# ---------------------------------------------------------------------------
# Individual-stratified train / val / test split
# ---------------------------------------------------------------------------
split_labels = np.full(n_ind, "train", dtype="U5")

sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=42)
trainval_idx, test_idx = next(sss_test.split(np.zeros(n_ind), superpop_labels))
split_labels[test_idx] = "test"

val_frac_adj = val_frac / (1.0 - test_frac)
sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_adj, random_state=42)
train_rel, val_rel = next(
    sss_val.split(np.zeros(len(trainval_idx)), superpop_labels[trainval_idx])
)
split_labels[trainval_idx[val_rel]] = "val"

for name in ("train", "val", "test"):
    print(f"  {name}: {(split_labels == name).sum()} individuals", flush=True)

# ---------------------------------------------------------------------------
# Save to HDF5
# ---------------------------------------------------------------------------
with h5py.File(out_h5, "w") as f:
    f.create_dataset("sample_ids",      data=iids.astype("S"),          compression="gzip")
    f.create_dataset("superpop_labels", data=superpop_labels,           compression="gzip")
    f.create_dataset("splits",          data=split_labels.astype("S"),  compression="gzip")

    for chr_name in chrs:
        grp = f.create_group(chr_name)
        geno = chr_data[chr_name]["geno"]
        pos  = chr_data[chr_name]["positions"]
        n_rows, n_cols = geno.shape
        chunk_rows = min(50, n_rows)
        chunk_cols = min(2000, n_cols)
        grp.create_dataset(
            "genotypes", data=geno,
            compression="gzip", compression_opts=4,
            chunks=(chunk_rows, chunk_cols),
        )
        grp.create_dataset("positions", data=pos, compression="gzip")
        print(f"[{chr_name}] saved genotypes {geno.shape}", flush=True)

print(f"Dataset written to {out_h5}", flush=True)
