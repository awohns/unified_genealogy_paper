import pandas as pd
import numpy as np
import sys
import logging
from functools import reduce
from matplotlib_venn import venn3, venn3_circles

logging.basicConfig(filename='finding_union.log', filemode='w',
                    level=logging.DEBUG)

maxplanck = "max_planck_data/"
column_names = ["Chromosome", "Position", "Ref", "Alt"]
datasets = {}

for chromosome in list(range(1, 23)) + ["X", "Y"]:
    logging.debug(
            "Loading variant lists for chromosome {}".format(str(chromosome)))
    # Read in all the variant lists as pandas dataframes
    tgp = pd.read_csv("tgp.GRCh37/1kg.GRCh37.chr" + str(chromosome) + ".snps.txt",
                      sep="\t", names=column_names, skiprows=1)
    try:
        datasets['sgdp'] = pd.read_csv(
                "sgdp/sgdp.chr" + str(chromosome) + ".snps.pos.alleles.txt",
                sep="\t", names=column_names, skiprows=1)
    except:
        logging.error("Could not load SGDP sex chromosomes")

    try:
        datasets['ukbb'] = pd.read_csv(
                "ukbb/ukbb.chr" + str(chromosome) + ".snps.txt", sep=" ")
    except:
        logging.info("UKB phased sex chromosomes not available")
    reich_array = pd.read_csv(
            "1240k/reich1240karray.snps.pos.alleles.txt", sep=" ", header=None,
            names=column_names)

    if chromosome == "X":
        reich_chr = 23
    elif chromosome == "Y":
        reich_chr = 24
    else:
        reich_chr = chromosome
    datasets['reich_1240k'] = reich_array[
            reich_array["Chromosome"] == reich_chr]
    datasets['hgdp'] = pd.read_csv(
            "hgdp/hgdp_wgs.20190516.chr" + str(chromosome) + ".GRCh37.snps.txt",
            sep=" ", header=None, names=column_names)
    datasets['denisovan'] = pd.read_csv(
            maxplanck + "denisovan/denisovan.chr" + str(chromosome) + ".GRCh37.snps.txt",
            delimiter=" ", names=column_names, skiprows=1)
    datasets['altai'] = pd.read_csv(
            maxplanck + "altai/altai.chr" + str(chromosome) + ".GRCh37.snps.txt",
            delimiter=" ", names=column_names, skiprows=1)
    datasets['vindija'] = pd.read_csv(
            maxplanck + "vindija/vindija.chr" + str(chromosome) + ".GRCh37.snps.txt",
            delimiter=" ", names=column_names, skiprows=1)
    datasets['ust_ishim'] = pd.read_csv(
            maxplanck + "ust_ishim/ust_ishim.chr" + str(chromosome) + ".GRCh37.snps.txt",
            delimiter=" ", names=column_names, skiprows=1)
    datasets['chagyrskaya'] = pd.read_csv(
            maxplanck + "chagyrskaya/chagyrskaya.chr" + str(chromosome) + ".GRCh37.snps.txt",
            delimiter=" ", names=column_names, skiprows=1)
    datasets['lbk'] = pd.read_csv(
            maxplanck + "lbk/lbk.chr" + str(chromosome) + ".GRCh37.snps.txt",
            delimiter=" ", names=column_names, skiprows=1)
    datasets['loshbour'] = pd.read_csv(
            maxplanck + "loshbour/loshbour.chr" + str(chromosome) + ".GRCh37.snps.txt",
            delimiter=" ", names=column_names, skiprows=1)

    # Start merging with thousand genomes array
    all_vars = tgp
    # Mask out non-biallelic sites to keep a record of only biallelic SNPs
    mask = (tgp['Ref'].str.len() == 1) & (tgp['Alt'].str.len() == 1)
    all_snps = tgp.loc[mask]

    for name, dataset in datasets.items():
        logging.debug("Merging {}".format(name))
        # Merge next dataset into all_variants and all_snps dfs with outer join
        all_vars = all_vars.merge(
                dataset.set_index('Position'), on="Position", how='outer',
                suffixes=('', '_' + name))
        logging.debug(
                "Variants found in {} but not previous datasets: {}".format(
                    name, np.sum((
                        all_vars['Ref'].isna()) | (all_vars['Alt'].isna()))))
        # If variant not found in the left dataset, copy info from right dataset
        # over to the ref and alt columns
        all_vars['Ref'] = all_vars['Ref'].fillna(all_vars['Ref_' + name])
        all_vars['Alt'] = all_vars['Alt'].fillna(all_vars['Alt_' + name])
        all_vars['Chromosome'] = all_vars['Chromosome'].fillna(
                all_vars['Chromosome_' + name])

        # Duplicated sites may mean overlap is bigger than smaller merging df
        # For instance, 1kg often has insertions and SNPs at the same position
        overlap = all_vars[
                np.logical_and(
                    all_vars['Ref'].notnull(),
                    all_vars['Ref_' + name].notnull())]
        logging.debug(
                "Variants recorded in both a previous dataset and {}: ".format(
                    name, overlap.shape[0]))
        # Check how many ref and alt alleles disagree between merging dfs
        logging.debug("Number of ref alleles which disagree {}".format(
                      np.sum(overlap['Ref'] != overlap['Ref_' + name])))
        logging.debug("Number of alt alleles which disagree {}".format(
                      np.sum(overlap['Alt'] != overlap['Alt_' + name])))

        # Now only consider biallelic SNPs
        mask = (dataset['Ref'].str.len() == 1) & (dataset['Alt'].str.len() == 1)
        dataset = dataset.loc[mask]
        all_snps = all_snps.merge(
                dataset.set_index('Position'), on="Position", how='outer',
                suffixes=('', '_' + name))
        logging.debug("SNPs found in {} but not previous datasets: {}".format(
                name,
                np.sum((all_snps['Ref'].isna()) | (all_snps['Alt'].isna()))))
        # If variant not found in the left dataset, copy info from right dataset
        # over to the ref and alt columns
        all_snps['Ref'] = all_snps['Ref'].fillna(all_snps['Ref_' + name])
        all_snps['Alt'] = all_snps['Alt'].fillna(all_snps['Alt_' + name])
        all_snps['Chromosome'] = all_snps['Chromosome'].fillna(
                all_snps['Chromosome_' + name])
        # Check that all ref and alt alleles are the same
        # Duplicated sites may mean overlap is bigger than smaller merging df
        overlap = all_snps[np.logical_and(all_snps['Ref'].notnull(),
                           all_snps['Ref_' + name].notnull())]
        logging.debug(
                "SNPs recorded in both a previous dataset and {}: {}".format(
                    name, overlap.shape[0]))
        logging.debug("Number of ref alleles which disagree {}".format(
            np.sum(overlap['Ref'] != overlap['Ref_' + name])))
        logging.debug("Number of alt alleles which disagree {}".format(
            np.sum(overlap['Alt'] != overlap['Alt_' + name])))

    # Save all_vars and all_snps to csv files
    if isinstance(chromosome, int):
        all_vars['Chromosome'] = all_vars['Chromosome'].astype('int32')
    all_vars[column_names].to_csv(
            'union_variants/all_variants.chr' + str(chromosome) + '.txt',
            sep=' ', index=False)
    if isinstance(chromosome, int):
        all_snps['Chromosome'] = all_snps['Chromosome'].astype('int32')
    all_snps[column_names].to_csv(
            'union_snps/all_snps.chr' + str(chromosome) + '.txt',
            sep=' ', index=False)
