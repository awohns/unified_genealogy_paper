"""
Script to find oldest ancient sample carrying derived alleles.
"""
import argparse
import pandas as pd
import numpy as np
import allel
import pickle
import os.path
from os import path
import tsinfer
import tskit
import sys
from tqdm import tqdm

sys.path.insert(0, "/home/wilderwohns/tsdate_paper/tsdate/")
import tsdate
from tsdate import posterior_mean_var


# class AncientMutAgesTgp:
#    """
#    Class for associating 1000G mutations with ancient sample ages and locations
#    """
#
#    def __init__(self, chromosome, samples_fn, tgp_tree_sequence, tsdate_tgp_fn, relate_tgp_fn, geva_tgp_fn):
#
#        self.anno_file = pd.read_csv(anno_fn, delimiter="\t", index_col="Index")
#        self.anno_file.columns = ['Instance_ID', 'Master_ID', 'Skeletal_Code',
#                                  'Skeletal_element', 'LibraryID(s)',
#                                  'No._Libraries', 'Data_type', 'Publication',
#                                   'Data_type', 'Publication', 'Average_age',
#                                   'Date_2formats', 'Group_ID', 'Location',
#                                   'Country', 'Lat.', 'Long.', 'Sex', 'mtDNA',
#                                   'Ychrom', 'perc_endogenous', 'Coverage',
#                                   'Autosome_SNPs_hit', 'UDG_treatment',
#                                   'Damage_restrict', 'Xcontam_point_est',
#                                   'Xcontam_z_score', 'endogenous_by_lib',
#                                   'damage_by_lib', 'mthaplogroup_by_lib',
#                                   'mt_match_rate_to_consensus_by_lib',
#                                   'Year_published']
#        self.snp_file = pd.read_csv(snp_fn, header=None, delim_whitespace=True)
#        self.snp_file = self.snp_file.loc[np.where(self.snp_file[1] == self.chrom)[0]]
#        self.tgp_sites = pd.read_csv(tgp_fn)
# self.vcf = allel.read_vcf(vcf_fn, delimiter=' ', index_col=0)
# self.vcf = allel.read_vcf(vcf_fn)


#    def check_polarisation(self, position, snp=None):
#        if snp:
#            if snp in self.snp_file[0]:
#                if position in self.snp_file[3]:
#                    return self.snp_file[np.logical_and([self.snp_file[0] == snp],
#                                                        [self.snp_file[3] == position])[0]]
#        else:
#            if position in self.snp_file[3]:
#                return self.snp_file[self.snp_file[3] == postiion]
#
#    def get_age(self):
#        """
#        For each variant, get the ages of ancient samples with derived allele
#        """
#        for index, row in self.tgp_sites.iterrows():
#            snp_info = self.check_polarisation(row[0])
#            vcf_pos_var = np.where(self.vcf['variants/POS'] == row[0])[0]
#            # Check the polarisation is the same between ancient and 1kg
#            snp_data = self.check_polarisation(row[0])
#            if snp_data[4] == row[2] and snp_data[5] == row[3]:
#                # Check which samples have the derived variant
#                anc_derv_samples = np.where(self.vcf['calldata/GT'][vcf_pos_var] == 1)
#                self.anno_file.loc[anc_derv_samples[0]]['Average_age']
#
#    def derived_in_any(self):
#        """
#        Find list of sites that are derived in 1240k, Afanasievo, Archaics or other ancient samples.
#        """
#
#        # Altai derived alleles
#        np.where(vcf['calldata/GT'] == 1)[0]

def get_relate_tgp_age_df():
    if path.exists("all-data/1kg_chr20_relate_mutation_ages.csv"):
        relate_ages = pd.read_csv("all-data/1kg_chr20_relate_mutation_ages.csv", index_col=0)
    else:
    #relate_ts = tskit.load("/home/wilderwohns/tsdate/relate_chr20.trees")
        relate_ts = tskit.load("/home/jk/large_files/relate/relate_chr20_metdata.trees")
        relate_mut_ages, relate_mut_upper_bound = get_mut_ages(relate_ts, relate_ts.tables.nodes.time[:])

        relate_ages = pd.DataFrame(
            {
                "position": relate_ts.tables.sites.position[
                    relate_ts.tables.mutations.site
                ],
                "relate_age": relate_mut_ages,
                "relate_upper_bound": relate_mut_upper_bound,
                "relate_ancestral_allele": np.array(tskit.unpack_strings(relate_ts.tables.sites.ancestral_state,
                    relate_ts.tables.sites.ancestral_state_offset))[relate_ts.tables.mutations.site],
                "relate_derived_allele": tskit.unpack_strings(relate_ts.tables.mutations.derived_state,
                    relate_ts.tables.mutations.derived_state_offset)
            }
        )
        relate_ages.to_csv("all-data/1kg_chr20_relate_mutation_ages.csv")
    return relate_ages

def get_geva_tgp_age_df():
    if path.exists("all-data/1kg_chr20_geva_mutation_ages.csv"):
        geva_ages = pd.read_csv("all-data/1kg_chr20_geva_mutation_ages.csv", index_col=0)
    else:
        geva = pd.read_csv(
            "/home/wilderwohns/tsinfer_geva/atlas.chr20.csv.gz",
            delimiter=",",
            skipinitialspace=True,
            skiprows=3,
        )
        geva_tgp = geva[geva["DataSource"] == "TGP"]
        geva_tgp_consistent = geva_tgp[geva_tgp["AlleleAnc"] == geva_tgp["AlleleRef"]]
        geva_ages = geva_tgp_consistent[["Position", "AgeMean_Jnt", "AgeCI95Upper_Jnt", "AlleleRef", "AlleleAlt"]]
        geva_ages.to_csv("all-data/1kg_chr20_geva_mutation_ages.csv")
    return geva_ages

def get_tsdate_tgp_age_df():
#    if path.exists("all-data/tsdate_ages_tgp_chr20_ma0.1_ms0.01_p13.csv"):
#        tsdate_ages = pd.read_csv("all-data/tsdate_ages_tgp_chr20_ma0.1_ms0.01_p13.csv", index_col=0)
#        # Only choose the oldest mutation at each position
#        sort_upper_bound = tsdate_ages.sort_values(by=["tsdate_upper_bound"], ascending=False, kind='mergesort')
#        tsdate_ages = sort_upper_bound.groupby('position', as_index=False).first() 

#    else:
    tgp_chr20 = tskit.load("all-data/1kg_chr20_allsites_ma0.1_ms0.01_p13.simplify.snipped.trees")
    tgp_chr20 = tgp_chr20.keep_intervals(
        [[tgp_chr20.tables.sites.position[0], tgp_chr20.tables.sites.position[-1]]]
    )
        #    combined_pos = np.intersect1d(geva_tgp_consistent['Position'],
        #            tgroep_chr20.tables.sites.position[tgp_chr20.tables.mutations.site])
        #    combined_pos = np.intersect1d(
        #            relate_ts.tables.sites.position[relate_ts.tables.mutations.site],
        #            combined_pos)
        #    relate_overlap = np.isin(
        #            relate_ts.tables.sites.position[relate_ts.tables.mutations.site],
        #            combined_pos)
        #    geva_overlap = np.isin(geva_tgp_consistent['Position'], combined_pos)
        #    tsdate_overlap = np.isin(
        #            tgp_chr20.tables.sites.position[tgp_chr20.tables.mutations.site],
        #            combined_pos)

    #    prior_remove_oldest = pickle.load(
    #        open("/home/wilderwohns/tsdate/timepoints_50_prior_snipped_notrim.p", "rb")
    #    )
    #    posterior_remove_oldest = pickle.load(open("tsdate/posterior.p", "rb"))
        #prior = pickle.load(open("prior.p", "rb"))
        #posterior = pickle.load(open("posterior.p", "rb"))
        #mn, vr= posterior_mean_var(
        #    tgp_chr20,
        #    prior.timepoints,
        #    posterior,
        #    fixed_node_set=set(tgp_chr20.samples()),
        #)
        #mn = pickle.load(open("all-data/1kg_chr20_ma0.1_ms0.01_p13.simplify.snipped.dates.p", "rb"))
    mn = pickle.load(open("all-data/1kg_chr20_allsites_ma0.1_ms0.01_p13.simplify.snipped.50timepoints.dates.p", "rb"))
    posterior_mut_ages, posterior_upper_bound = get_mut_ages(
        tgp_chr20, mn, ignore_sample_muts=True
    )
    mut_frequencies = get_mut_frequencies(tgp_chr20)
    tsdate_ages = pd.DataFrame(
        {
            "position": tgp_chr20.tables.sites.position[
                tgp_chr20.tables.mutations.site
            ],
            "tsdate_age": 20000 * posterior_mut_ages,
            "tsdate_upper_bound": 20000 * posterior_upper_bound,
            "frequency": mut_frequencies,
            "tsdate_ancestral_allele": np.array(tskit.unpack_strings(tgp_chr20.tables.sites.ancestral_state,
                tgp_chr20.tables.sites.ancestral_state_offset))[tgp_chr20.tables.mutations.site],
            "tsdate_derived_allele": tskit.unpack_strings(tgp_chr20.tables.mutations.derived_state,
                tgp_chr20.tables.mutations.derived_state_offset)
        }
    )
    tsdate_ages.to_csv("all-data/tsdate_ages_tgp_chr20_ma0.1_ms0.01_p13.csv")
    # Only choose the oldest mutation at each position
    sort_upper_bound = tsdate_ages.sort_values(by=["tsdate_upper_bound"], ascending=False, kind='mergesort')
    tsdate_ages = sort_upper_bound.groupby('position', as_index=False).first() 
    return tsdate_ages


def tgp_date_estimates():
    """
    Produce comparable set of mutations from tgp
    """
    #tgp_chr20 = tskit.load("all-data/1kg_chr20_allsites.snipped.trees")
    #tgp_chr20 = tskit.load("all-data/1kg_chr20_ma0.1_ms0.01_p13.simplify.snipped.trees")
    tsdate_ages = get_tsdate_tgp_age_df()
    geva = get_geva_tgp_age_df()
    merged = pd.merge(tsdate_ages, geva,
            left_on=["position", "tsdate_ancestral_allele", "tsdate_derived_allele"], right_on=["Position", "AlleleRef", "AlleleAlt"])
    #merged.to_csv("intermediate_merge")
    relate_ages = get_relate_tgp_age_df()
    #relate_ages = pd.read_csv("relateages")
    merged = pd.merge(merged, relate_ages,
            left_on=["Position", "tsdate_ancestral_allele", "tsdate_derived_allele"],
            right_on=["position", "relate_ancestral_allele", "relate_derived_allele"])
    merged = merged.drop(columns=["position_x", "position_y"])
    merged.to_csv("all-data/tgp_mutations_allmethods_new_snipped_allsites.csv")

def get_mut_frequencies(ts):
    """
    Calculate frequency of each site and return numpy 1d array of len num_mutations 
    with frequency as values. This assumes that if there are multiple mutations at a 
    site they are recurrent.
    """
    mut_freq = np.zeros((ts.num_mutations)) 
    for var in tqdm(ts.variants(), total=ts.num_sites, desc="Get mutation frequencies"):
        frequency = np.sum(var.genotypes) / ts.num_samples
        for mut in var.site.mutations:
            mut_freq[mut.id] = frequency
    return mut_freq

def get_mut_ages(ts, dates, ignore_sample_muts=False):
    mut_ages = list()
    mut_upper_bounds = list()
    if ignore_sample_muts:
        mutations_table = ts.tables.mutations
        unique_sites = np.unique(ts.tables.mutations.site, return_counts=True)
        unique_sites = unique_sites[0][unique_sites[1] > 1]
        no_samp_muts = ~np.logical_and(np.isin(mutations_table.site, unique_sites),
                np.isin(mutations_table.node, ts.samples()))
    for tree in tqdm(ts.trees(), total=ts.num_trees, desc="Finding mutation ages"):
        for site in tree.sites():
            for mut in site.mutations:
                parent_age = dates[tree.parent(mut.node)]
                mut_upper_bounds.append(parent_age)
                mut_ages.append((dates[mut.node] + parent_age) / 2)
    return np.array(mut_ages), np.array(mut_upper_bounds)


def get_ancient_constraints_tgp():
    ancient_samples = tsinfer.load("all-data/combined_archaics.samples")
    ages = np.array(
        [
            [metadata["age"], metadata["age"]]
            for metadata in ancient_samples.individuals_metadata[:]
        ]
    )
    ages = ages.reshape(13374).astype(int)
    genotypes = ancient_samples.sites_genotypes[:][:, ages != 0]
    ancient_deriv_bool = np.any(genotypes == 1, axis=1)
    genotypes = genotypes[ancient_deriv_bool]
    positions = ancient_samples.sites_position[:][ancient_deriv_bool]
    alleles = ancient_samples.sites_alleles[:][ancient_deriv_bool]
    ages = ages[ages != 0]
    lower_bound = [
        (pos, allele[0], allele[1], int(np.max(ages[np.where(geno == 1)[0]])), np.sum(geno[geno==1]))
        for pos, allele, geno in zip(positions, alleles, genotypes)
    ]
    constraint_df = pd.DataFrame(lower_bound, columns=["Position", "Reference Allele",
        "Alternative Allele", "Ancient Bound", "Number of Ancients"])
    constraint_df = constraint_df.astype({"Position": "int64", "Ancient Bound": "float64", "Number of Ancients": "int32"})
    constraint_df = constraint_df[constraint_df["Ancient Bound"] != 0]
    #constraint_df = constraint_df.set_index("Position")
    constraint_df.to_csv("all-data/ancient_constraints.csv")
    try:
        tgp_mut_ests = pd.read_csv("all-data/tgp_mutations_allmethods_new_snipped_allsites.csv", index_col=0)
    except:
        raise ValueError("tgp_mutations.csv does not exist. Must run tgp_dates first")
    tgp_muts_constraints = pd.merge(
        tgp_mut_ests, constraint_df, left_on=["Position", "tsdate_ancestral_allele", "tsdate_derived_allele"], right_on=["Position", "Reference Allele", "Alternative Allele"])
    tgp_muts_constraints.to_csv("all-data/tgp_muts_constraints.csv")


def main():
    name_map = {
        "tgp_dates": tgp_date_estimates,
        "ancient_constraints": get_ancient_constraints_tgp,
    }

    parser = argparse.ArgumentParser(
        description="Process the human data and make data files for plotting."
    )
    parser.add_argument(
        "name", type=str, help="figure name", choices=list(name_map.keys())
    )

    args = parser.parse_args()
    name_map[args.name]()


if __name__ == "__main__":
    main()
