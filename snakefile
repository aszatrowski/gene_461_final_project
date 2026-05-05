configfile: "config.yaml"
CHRS = config["chrs"]

wildcard_constraints:
        chr     = r"chr\d+",  # chromosome names like chr19, chr21
rule all:
    input: 
        expand(
            "data/1kg_{chr}_biallelic_segregating.vcf.gz{ext}",
            chr=CHRS,
            ext=['', '.tbi']
        ),
        expand(
            "outputs/{chr}/pca.png",
            chr=CHRS
        ),
        "data/1kg_phase3_samples.tsv",

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
        pca_plot = "outputs/{chr}/pca.png"
    params:
        outlier_threshold_sd = 5
    conda: "workflow/envs/preprocess.yaml"
    script: "scripts/plot_pca.R"