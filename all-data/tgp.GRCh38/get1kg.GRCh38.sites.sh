#!/bin/bash

# get snp list from 1000 genomes file
for i in {1..22}
do
    url="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/GRCh38_positions/"
    file="ALL.chr${i}_GRCh38.genotypes.20170504.vcf.gz"
    if [ ! -f $file ]; then
        curl ${url}${file} -o $file
    fi
    zgrep -v "^##" $file | cut -f1-2,4-5 > "1kg.GRCh38.chr${i}.snps.pos.alleles.txt" 
    #rm $file
done
