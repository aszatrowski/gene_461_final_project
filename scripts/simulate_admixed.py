"""
simulate_admixed.py — Create synthetic admixed individuals for LAI evaluation.

For each of several ancestry-pair combinations, N individuals are simulated by
splicing together chromosomal tracts from two test-set reference individuals.
The crossover point is chosen uniformly between the 20th and 80th percentile of
SNP positions, so both parent tracts are well represented.

Called by Snakemake; expects:
    snakemake.input[0]  — dataset.h5 produced by prepare_data.py
    snakemake.output[0] — admixed_test.h5
    snakemake.config    — dict with n_simulated_admixed
"""

import numpy as np
import h5py

SUPERPOP_NAMES = {0: "AFR", 1: "AMR", 2: "EAS", 3: "EUR", 4: "SAS"}

# Ancestry pairs to simulate (pop_a_idx, pop_b_idx)
# Chosen to cover major 2-way admixture scenarios
ANCESTRY_PAIRS = [
    (3, 0),  # EUR + AFR  (like African-American)
    (3, 2),  # EUR + EAS  (like East Asian - European)
    (0, 2),  # AFR + EAS
    (3, 4),  # EUR + SAS  (like South Asian - European)
    (0, 4),  # AFR + SAS
]

input_h5  = snakemake.input[0]
output_h5 = snakemake.output[0]
n_total   = snakemake.config.get("n_simulated_admixed", 200)
n_per_pair = max(1, n_total // len(ANCESTRY_PAIRS))

# ---------------------------------------------------------------------------
# Load test-split individuals from the reference dataset
# ---------------------------------------------------------------------------
with h5py.File(input_h5, "r") as f:
    splits = f["splits"][:].astype(str)
    labels = f["superpop_labels"][:]
    test_mask = splits == "test"

    chrs = sorted(k for k in f.keys() if k.startswith("chr"))
    chr_genos = {c: f[f"{c}/genotypes"][test_mask] for c in chrs}

test_labels = labels[test_mask]

# ---------------------------------------------------------------------------
# Simulate 2-way admixed individuals
# ---------------------------------------------------------------------------
records = []  # list of dicts: {geno_per_chr, tract_per_chr, crossover_per_chr, pops}

for pop_a, pop_b in ANCESTRY_PAIRS:
    idx_a = np.where(test_labels == pop_a)[0]
    idx_b = np.where(test_labels == pop_b)[0]
    if len(idx_a) == 0 or len(idx_b) == 0:
        print(
            f"Skipping {SUPERPOP_NAMES[pop_a]}+{SUPERPOP_NAMES[pop_b]}: "
            f"insufficient test individuals ({len(idx_a)}, {len(idx_b)})",
            flush=True,
        )
        continue

    rng = np.random.default_rng(seed=pop_a * pop_b)
    sel_a = rng.choice(idx_a, size=n_per_pair, replace=len(idx_a) < n_per_pair)
    sel_b = rng.choice(idx_b, size=n_per_pair, replace=len(idx_b) < n_per_pair)

    for i_a, i_b in zip(sel_a, sel_b):
        geno_per_chr     = {}
        tract_per_chr    = {}
        crossover_per_chr = {}

        for chr_name in chrs:
            g_a    = chr_genos[chr_name][i_a]
            g_b    = chr_genos[chr_name][i_b]
            n_snps = g_a.shape[0]

            # Crossover uniformly between 20% and 80% of SNP indices
            k = rng.integers(int(0.2 * n_snps), int(0.8 * n_snps))

            geno_per_chr[chr_name]     = np.concatenate([g_a[:k], g_b[k:]])
            tract_per_chr[chr_name]    = np.concatenate([
                np.full(k,          pop_a, dtype=np.int8),
                np.full(n_snps - k, pop_b, dtype=np.int8),
            ])
            crossover_per_chr[chr_name] = k

        records.append({
            "geno":      geno_per_chr,
            "tract":     tract_per_chr,
            "crossover": crossover_per_chr,
            "pops":      (pop_a, pop_b),
        })

    print(
        f"Simulated {n_per_pair} "
        f"{SUPERPOP_NAMES[pop_a]}+{SUPERPOP_NAMES[pop_b]} individuals",
        flush=True,
    )

n_admixed = len(records)
print(f"Total simulated individuals: {n_admixed}", flush=True)

# ---------------------------------------------------------------------------
# Save to HDF5
# ---------------------------------------------------------------------------
with h5py.File(output_h5, "w") as f:
    for chr_name in chrs:
        n_snps = chr_genos[chr_name].shape[1]
        geno_arr     = np.stack([r["geno"][chr_name]  for r in records])  # (N, n_snps)
        tract_arr    = np.stack([r["tract"][chr_name] for r in records])
        crossover_arr = np.array([r["crossover"][chr_name] for r in records], dtype=np.int32)

        grp = f.create_group(chr_name)
        grp.create_dataset("genotypes",       data=geno_arr,     compression="gzip")
        grp.create_dataset("tract_labels",    data=tract_arr,    compression="gzip")
        grp.create_dataset("crossover_points", data=crossover_arr)

    parent_pops = np.array([r["pops"] for r in records], dtype=np.int8)
    f.create_dataset("parent_pops", data=parent_pops)
    f.attrs["ancestry_pairs"] = str(ANCESTRY_PAIRS)

print(f"Admixed test set written to {output_h5}", flush=True)
