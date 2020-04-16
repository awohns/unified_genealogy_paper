#!/bin/bash

# get snp list from ukbb snp array
url="http://www.affymetrix.com/analysis/downloads/na34/genotyping/"
file="Axiom_UKB_WCSG.na34.annot.csv.zip"

#curl ${url}${file} -o ${file}
#unzip ${file}
grep -v "^#" "Axiom_UKB_WCSG.na34.annot.csv" | awk -F '[,]' '{print $5, $6, $14, $15}' | tr -d '"' > "ukbb.snps.pos.alleles.txt"
#rm Axiom_UKB_WCSG.na34.annot.csv
#rm Axiom_UKB_WCSG.na34.annot.csv.zip
#rm AFFX_README-NetAffx-CSV-Files_Axiom.txt
