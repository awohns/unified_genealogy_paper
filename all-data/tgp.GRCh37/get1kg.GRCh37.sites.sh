#!/bin/bash

# get snp list from 1000 genomes file
for i in {1..22} 
do
    url="ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
    file="ALL.chr${i}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
    if [ ! -f $file ]; then
        wget -nc ${url}${file} -O $file
    fi
    zgrep -v "^##" $file | cut -f1-2,4-5 > "1kg.GRCh37.chr${i}.snps.txt" 
    #rm $file
done

wget -nc "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chrX.phase3_shapeit2_mvncall_integrated_v1b.20130502.genotypes.vcf.gz"
zgrep -v "^##" "ALL.chrX.phase3_shapeit2_mvncall_integrated_v1b.20130502.genotypes.vcf.gz" | cut -f1-2,4-5 > "1kg.GRCh37.chrX.snps.txt"

wget -nc "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chrY.phase3_integrated_v2a.20130502.genotypes.vcf.gz"
zgrep -v "^##" "ALL.chrY.phase3_integrated_v2a.20130502.genotypes.vcf.gz" | cut -f1-2,4-5 > "1kg.GRCh37.chrY.snps.txt"
