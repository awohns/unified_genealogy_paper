#!/bin/bash

# Replace all 0 values for recombination rate by 0.001
#cat genetic_map_GRCh38_merged.tab | tr ' ' '\t' > genetic_map_GRCh38_merged_modified.tab
awk -F"\t" 'BEGIN{OFS="\t";} $3 ==0 {$3=0.001}1' genetic_map_GRCh38_merged.tab > genetic_map_GRCh38_merged_modified.tab

# Split file based on value of column 1
awk 'NR>1 {print>"genetic_map_GRCh38_"$1".txt"}' genetic_map_GRCh38_merged_modified.tab 

# Last recombination rate value should be 0 
for chrom in {1..22}
do
    sed -i '1s/^/chrom\tpos\trecomb_rate\tpos_cm\n/' genetic_map_GRCh38_chr${chrom}.txt
    sed -i "$(( $(wc -l < genetic_map_GRCh38_chr${chrom}.txt) - 1 + 1)),\$s/[[:space:]]0.001[[:space:]]/\t0\t/g" genetic_map_GRCh38_chr${chrom}.txt
done

sed -i '1s/^/chrom\tpos\trecomb_rate\tpos_cm\n/' genetic_map_GRCh38_chrX.txt
sed -i "$(( $(wc -l < genetic_map_GRCh38_chrX.txt) - 1 + 1)),\$s/[[:space:]]0.001[[:space:]]/\t0\t/g" genetic_map_GRCh38_chrX.txt
