import pandas as pd
import numpy as np
import sys
import logging
from functools import reduce
from matplotlib_venn import venn3, venn3_circles
import pysam

logging.basicConfig(filename='finding_union.log', filemode='w',
                    level=logging.DEBUG)

def mask_biallelic(snp_df):
    mask = (snp_df['Ref'].str.len() == 1) & (snp_df['Alt'].str.len() == 1)
    snp_df_biallelic = snp_df.loc[mask].reset_index()
    return snp_df_biallelic 

def check_ref(snp_df, dataset_name, chromosome, reference_states):
    assert np.all(snp_df["Chromosome"] == str(chromosome)), dataset_name
    non_matching_ref = 0
    dropped_indices = list()
    flipped_indices = list()
    for index, (position, ref, alt) in enumerate(zip(snp_df["Position"], snp_df["Ref"], snp_df["Alt"])):
        if ref != reference_states[position]:
            non_matching_ref += 1
            if alt == reference_states[position]:
                flipped_indices.append(index)
            else:
                dropped_indices.append(index)
    snp_df = snp_df.drop(dropped_indices)
    if len(flipped_indices) > 0:
        flipped_indices = np.array(flipped_indices)
        flipped_alleles = snp_df.loc[flipped_indices, ["Alt", "Ref"]] 
        snp_df.loc[flipped_indices, ["Ref", "Alt"]] = flipped_alleles.values
    logging.debug(
            "Total number of sites in {} is {}".format(dataset_name, str(snp_df.shape[0])))
    logging.debug(
            "Sites with conflicting ref alleles {}".format(str(non_matching_ref)))
    logging.debug("Sites where ref/alt alleles were able to be flipped {}".format(str(len(flipped_indices))))
    return snp_df 

fasta = pysam.FastaFile("hs37d5.fa")
maxplanck = "max_planck_data/"
column_names = ["Chromosome", "Position", "Ref", "Alt"]
dtypes = {"Chromosome": str, "Position": int, "Ref": str, "Alt": str}
reich_array = pd.read_csv(
        "1240k/reich1240karray.snps.pos.alleles.txt", sep=" ", header=None,
        names=column_names, dtype=dtypes)
reich_array.loc[reich_array["Chromosome"] == "23", "Chromosome"] = "X"
reich_array.loc[reich_array["Chromosome"] == "24", "Chromosome"] = "Y"

for chromosome in list(range(1, 23)) + ["X", "Y"]:
    datasets = {}
    if chromosome not in ["X", "Y"]:
        reference_states = "X" + fasta.fetch(reference=fasta.references[chromosome - 1])
    elif chromosome == "X":
        reference_states = "X" + fasta.fetch(reference=fasta.references[22])
    elif chromosome == "Y":
        reference_states = "X" + fasta.fetch(reference=fasta.references[23])

    logging.debug(
            "Loading variant lists for chromosome {}".format(str(chromosome)))
    # Read in all the variant lists as pandas dataframes
    tgp = pd.read_csv("tgp.GRCh37/1kg.GRCh37.chr" + str(chromosome) + ".snps.txt",
                      sep="\t", names=column_names, dtype=dtypes, header=0) 
    try:
        datasets['sgdp'] = pd.read_csv(
                "sgdp/sgdp.chr" + str(chromosome) + ".snps.pos.alleles.txt",
                sep="\t", names=column_names, dtype=dtypes, skiprows=1)
    except:
        logging.error("Could not load SGDP sex chromosomes")

    try:
        datasets['ukbb'] = pd.read_csv(
                "ukbb/ukbb.chr" + str(chromosome) + ".snps.txt", sep=" ", dtype=dtypes)
    except:
        logging.info("UKB phased sex chromosomes not available")

    datasets['reich_1240k'] = reich_array[
            reich_array["Chromosome"] == chromosome]
    datasets['hgdp'] = pd.read_csv(
            "hgdp/hgdp_wgs.20190516.chr" + str(chromosome) + ".GRCh37.snps.txt",
            sep=" ", header=None, names=column_names, dtype=dtypes)
    datasets['denisovan'] = pd.read_csv(
            maxplanck + "denisovan/denisovan.chr" + str(chromosome) + ".GRCh37.snps.txt",
            delimiter=" ", names=column_names, skiprows=1, dtype=dtypes)
    datasets['altai'] = pd.read_csv(
            maxplanck + "altai/altai.chr" + str(chromosome) + ".GRCh37.snps.txt",
            delimiter=" ", names=column_names, skiprows=1, dtype=dtypes)
    datasets['vindija'] = pd.read_csv(
            maxplanck + "vindija/vindija.chr" + str(chromosome) + ".GRCh37.snps.txt",
            delimiter=" ", names=column_names, skiprows=1, dtype=dtypes)
    datasets['ust_ishim'] = pd.read_csv(
            maxplanck + "ust_ishim/ust_ishim.chr" + str(chromosome) + ".GRCh37.snps.txt",
            delimiter=" ", names=column_names, skiprows=1, dtype=dtypes)
    datasets['chagyrskaya'] = pd.read_csv(
            maxplanck + "chagyrskaya/chagyrskaya.chr" + str(chromosome) + ".GRCh37.snps.txt",
            delimiter=" ", names=column_names, skiprows=1, dtype=dtypes)
    datasets['lbk'] = pd.read_csv(
            maxplanck + "lbk/lbk.chr" + str(chromosome) + ".GRCh37.snps.txt",
            delimiter=" ", names=column_names, skiprows=1, dtype=dtypes)
    datasets['loshbour'] = pd.read_csv(
            maxplanck + "loshbour/loshbour.chr" + str(chromosome) + ".GRCh37.snps.txt",
            delimiter=" ", names=column_names, skiprows=1, dtype=dtypes)

    # Start merging with thousand genomes array
    all_vars = tgp
    # Mask out non-biallelic sites to keep a record of only biallelic SNPs
    all_snps = mask_biallelic(tgp)
    all_snps = check_ref(all_snps, "tgp", chromosome, reference_states)

    for name, dataset in datasets.items():
        logging.debug("Merging {}".format(name))
        # Merge next dataset into all_variants and all_snps dfs with outer join
        dataset_new = dataset[~np.isin(dataset["Position"], all_vars["Position"])]
        all_vars = pd.concat([all_vars, dataset_new], sort=True).drop_duplicates()
       # all_vars = all_vars.merge(
       #         dataset.set_index('Position'), on="Position", how='outer',
       #         suffixes=('', '_' + name))
        logging.debug(
                "Variants at positions found in {} but not previous datasets: {}".format(
                    name, dataset_new.shape[0]))
        # If variant not found in the left dataset, copy info from right dataset
        # over to the ref and alt columns
       # all_vars['Ref'] = all_vars['Ref'].fillna(all_vars['Ref_' + name])
       # all_vars['Alt'] = all_vars['Alt'].fillna(all_vars['Alt_' + name])
       # all_vars['Chromosome'] = all_vars['Chromosome'].fillna(
       #         all_vars['Chromosome_' + name])

        # Duplicated sites may mean overlap is bigger than smaller merging df
#        # For instance, 1kg often has insertions and SNPs at the same position
#        overlap = all_vars[
#                np.logical_and(
#                    all_vars['Ref'].notnull(),
#                    all_vars['Ref_' + name].notnull())]
#        logging.debug(
#                "Variants recorded in both a previous dataset and {}: ".format(
#                    name, overlap.shape[0]))
        # Check how many ref and alt alleles disagree between merging dfs
#        logging.debug("Number of ref alleles which disagree {}".format(
#                      np.sum(overlap['Ref'] != overlap['Ref_' + name])))
#        logging.debug("Number of alt alleles which disagree {}".format(
#                      np.sum(overlap['Alt'] != overlap['Alt_' + name])))

        # Now only consider biallelic SNPs
        dataset = mask_biallelic(dataset) 
        dataset = check_ref(dataset, name, chromosome, reference_states)
        dataset_new = dataset[~np.isin(dataset["Position"], all_snps["Position"])]
        all_snps = pd.concat([all_snps, dataset_new], sort=True).drop_duplicates()
        logging.debug(
                "SNPs at positions found in {} but not previous datasets: {}".format(
                    name, dataset_new.shape[0]))

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
