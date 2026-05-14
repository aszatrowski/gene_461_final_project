configfile: "config.yaml"
CHRS = config["chrs"]

wildcard_constraints:
        chr     = r"chr\d+",  # chromosome names like chr19, chr21
rule all:
    input: 
        expand(
            "outputs/pca_{chr}.png",
            chr=CHRS
        ),

rule download_1kg:
    """
    Downloads chromosome-specific VCF files with wget from the 1000Genomes public FTP server.
    """
    output:
        vcf="data/1kg_{chr}.vcf.gz",
        tbi="data/1kg_{chr}.vcf.gz.tbi"
    params:
        url=lambda wc: (
            "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
            f"ALL.{wc.chr}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
        )
    localrule: True
    shell:
        """
        mkdir -p data
        wget -O {output.vcf} {params.url}
        wget -O {output.tbi} {params.url}.tbi
        """

rule download_sample_pop_sheet:
    """
    Downloads the 1000Genomes population assignment sheet (to be used as classifier labels) from the FTP server.
    """
    output:
        "data/1kg_phase3_samples.tsv"
    params:
        url = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel"
    shell:
        """
        mkdir -p data
        wget -O {output} {params.url}
        """

rule extract_sample_list:
    input:
        "data/1kg_phase3_samples.tsv"
    output:
        "data/sample_lists/{pop_1kg}.txt"
    params:
        pops=lambda wc: " ".join(POPULATIONS[wc.pop_1kg])
    localrule: True
    shell:
        """
        mkdir -p $(dirname {output})
        awk -v pops="{params.pops}" \
            'BEGIN{{n=split(pops,a," "); for(i=1;i<=n;i++) ok[a[i]]=1}} \
             NR>1 && ok[$2] {{print $1}}' {input} > {output}
        """

rule extract_biallelic_test_SNPs:
    """
    Filters downloaded VCFs to just biallelic segregating SNPs; in effect this filters the data to only the most informative loci, which saves storage, memory, compute, training time, etc.
    """
    input:
        vcf="data/1kg_{chr}.vcf.gz",
        tbi="data/1kg_{chr}.vcf.gz.tbi"
    output:
        vcf="data/1kg_{chr}_biallelic_segregating.vcf.gz",
        tbi="data/1kg_{chr}_biallelic_segregating.vcf.gz.tbi"
    log: "logs/extract_biallelic_test_SNPs.{chr}.log"
    conda: "workflow/envs/preprocess.yaml"
    threads: 8
    resources:
        runtime = 30,
        mem_mb = 500
    shell:
        """
        exec 2> {log}
        bcftools view -m2 -M2 -v snps --threads {threads} {input.vcf} | \
        bcftools filter -Oz --include 'INFO/AC > 0 && INFO/AC < INFO/AN' --threads {threads} > {output.vcf}
        bcftools index --tbi {output.vcf}
        """

rule make_plink_fileset:
    """
    Converts chr-specific VCFs to plink format for downstream plink-based processing.
    """
    input: 
        vcf = "data/1kg_{chr}_biallelic_segregating.vcf.gz",
        index = "data/1kg_{chr}_biallelic_segregating.vcf.gz.tbi"
    output: 
        multiext("data/{chr}/plink", ".pgen", ".pvar", ".psam")
    params:
        out_prefix = "data/{chr}/plink"
    resources:
        mem = "2G",
        runtime = 5
    conda: "workflow/envs/preprocess.yaml"
    shell: 
        """
        mkdir -p data/plink
        plink2 --vcf {input.vcf} --make-pgen --out {params.out_prefix}
        """

rule compute_pca:
    input: 
        multiext("data/{chr}/plink", ".pgen", ".pvar", ".psam")
    output: 
        proj = "data/{chr}/plink.eigenvec",
        loadings = "data/{chr}/plink.eigenval"
    params:
        prefix = "data/{chr}/plink"
    resources:
        mem = "2G",
        runtime = 45
    conda: "workflow/envs/preprocess.yaml"
    shell: 
        """
        plink2 --pfile {params.prefix} --pca --out {params.prefix}
        """

rule plot_pca:
    input:
        proj = "data/{chr}/plink.eigenvec",
        variance = "data/{chr}/plink.eigenval",
        sample_pops = "data/1kg_phase3_samples.tsv"
    output:
        pca_plot = "outputs/pca_{chr}.png"
    params:
        outlier_threshold_sd = 5
    conda: "workflow/envs/preprocess.yaml"
    script: "scripts/plot_pca.R"


rule export_genotypes:
    """
    Exports plink binary genotypes to a dosage text matrix (.raw) after MAF filtering.
    Also writes a filtered variant info file (.pvar) whose rows match the .raw columns.
    """
    input:
        multiext("data/{chr}/plink", ".pgen", ".pvar", ".psam")
    output:
        raw  = "data/{chr}/genotypes.raw",
        pvar = "data/{chr}/genotypes.pvar"
    params:
        in_prefix  = "data/{chr}/plink",
        out_prefix = "data/{chr}/genotypes",
        maf        = config["maf_threshold"]
    conda: "workflow/envs/preprocess.yaml"
    resources:
        mem_mb  = 4000,
        runtime = 30
    shell:
        """
        plink2 --pfile {params.in_prefix} \
            --maf {params.maf} \
            --max-alleles 2 \
            --export A \
            --make-just-pvar \
            --out {params.out_prefix}
        """


rule prepare_training_data:
    """
    Reads per-chromosome dosage matrices and population labels, performs an
    individual-stratified train/val/test split, and writes a single HDF5 dataset.
    """
    input:
        raw     = expand("data/{chr}/genotypes.raw",  chr=config["chrs"]),
        pvar    = expand("data/{chr}/genotypes.pvar", chr=config["chrs"]),
        samples = "data/1kg_phase3_samples.tsv"
    output:
        "data/dataset.h5"
    conda: "workflow/envs/ml.yaml"
    resources:
        mem_mb  = 16000,
        runtime = 60
    script: "scripts/prepare_data.py"


rule simulate_admixed:
    """
    Creates synthetic 2-way admixed individuals from test-split reference
    individuals for local ancestry inference evaluation.
    """
    input:
        "data/dataset.h5"
    output:
        "data/admixed_test.h5"
    conda: "workflow/envs/ml.yaml"
    resources:
        mem_mb  = 8000,
        runtime = 20
    script: "scripts/simulate_admixed.py"


rule evaluate_model:
    """
    Evaluates a trained CNN checkpoint: confusion matrix on held-out test windows
    and LAI karyogram on simulated admixed individuals.
    """
    input:
        dataset    = "data/dataset.h5",
        admixed    = "data/admixed_test.h5",
        checkpoint = "models/best_model.pt"
    output:
        confusion = "outputs/confusion_matrix.png",
        karyogram = "outputs/lai_karyogram.png"
    conda: "workflow/envs/ml.yaml"
    resources:
        mem_mb  = 8000,
        runtime = 30
    script: "scripts/evaluate.py"