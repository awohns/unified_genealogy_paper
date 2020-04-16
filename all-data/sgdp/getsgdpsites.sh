#!/bin/bash

# get snp list from sgdp genomes file
SGDP_GENOTYPES_BASE=https://sharehost.hms.harvard.edu/genetics/reich_lab/sgdp/phased_data/PS2_multisample_public
for i in {1..22}
do
    file="cteam_extended.v4.PS2_phase.public.chr${i}.vcf.gz"
    output="sgdp_chr${i}.vcf.gz"
    wget -nc ${SGDP_GENOTYPES_BASE}/${file} -O $output
    zgrep -v "^##" $output | cut -f1-2,4-5 > "sgdp.chr${i}.snps.pos.alleles.txt"
    # rm $output
done
