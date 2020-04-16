#!/bin/bash

# get snp list from hgdp genomes file
for i in {1..22} "X" "Y"
do
    url="ftp://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/"
    file="hgdp_wgs.20190516.full.chr${i}.vcf.gz"
    if [ ! -f $file ]; then
        wget -nc ${url}${file} -O $file
    fi
    zgrep -v "^##" $file | cut -f1-5 > hgdp_wgs.20190516.chr${i}.GRCh38.snps.bed
    # zgrep -v "^##" $file | cut -f1-2,4-5 > "hgdp_chr${i}.snps.txt"
    awk '{print $1, $2, $2 + 1, $3"_"$4"_"$5}' hgdp_wgs.20190516.chr${i}.GRCh38.snps.bed > hgdp_wgs.20190516.chr${i}.GRCh38.snps.mod.bed
    ./liftover hgdp_wgs.20190516.chr${i}.GRCh38.snps.mod.bed hg38toHg19.over.chain.gz hgdp_wgs.20190516.chr${i}.GRCh37.snps.bed hgdp_wgs.20190516.chr${i}.unmappable 
    awk '{split($4,a,"_"); print substr($1,4),$2,a[2]" "a[3]}' hgdp_wgs.20190516.chr${i}.GRCh37.snps.bed > hgdp_wgs.20190516.chr${i}.GRCh37.snps.txt
done

# Some snps will have moved to other chromosomes in the liftover
# Now combine all the files and send snps to correct file
cat hgdp_wgs.20190516.chr*.GRCh37.snps.txt > hgdp.allchrs.GRCh37.snps.txt
awk '{print>"hgdp_wgs.20190516.chr"$1".GRCh37.snps.txt"}' hgdp.allchrs.GRCh37.snps.txt
