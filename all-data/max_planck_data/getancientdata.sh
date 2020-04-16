#!/bin/bash

# get snps from four sequenced ancient genomes

for i in {1..22} "X" "Y"
do
    make denisovan.chr${i}
    make vindija.chr${i}
    make altai.chr${i}
    make ust_ishim.chr${i}
    make chagyrskaya.chr${i}
    make lbk.chr${i}
    make loshbour.chr${i}
done
