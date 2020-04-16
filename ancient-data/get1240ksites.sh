#!/bin/bash

# get snp list from 1240k array

url="https://reichdata.hms.harvard.edu/pub/datasets/amh_repo/curated_releases/V42/V42.4/SHARE/public.dir/v42.4.1240K.snp"

curl $url -o "v42.4.1240K.snp"

cat v42.4.1240K.snp | tr -s ' ' | cut -d ' ' -f3,5,6,7 > "reich1240karray.snps.pos.alleles.txt"
