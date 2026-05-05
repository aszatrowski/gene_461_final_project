configfile: "config.yaml"
CHRS = config["chrs"]

wildcard_constraints:
        chr     = r"chr\d+",  # chromosome names like chr2, chr6
rule all:
    input: 
        expand(
            "data/1kg_{chr}_biallelic_segregating.vcf.gz{ext}",
            chr=CHRS,
            ext=['', '.tbi']
        ),
        "data/1kg_phase3_samples.tsv"

rule download_1kg:
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