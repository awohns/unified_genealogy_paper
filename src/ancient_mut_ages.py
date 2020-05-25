"""
Script to find oldest ancient sample carrying derived alleles.
"""
import argparse
import pandas as pd
import numpy as np
import allel
import pickle
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


def tgp_date_estimates():
    """
    Produce comparable set of mutations from tgp
    """
    tgp_chr20 = tskit.load("all-data/1kg_chr20_allsites.snipped.trees")
    relate_ts = tskit.load("/home/wilderwohns/tsdate/relate_chr20.trees")
    relate_node_times = relate_ts.tables.nodes.time[:]
    relate_mut_ages, relate_mut_upper_bound = get_mut_ages(relate_ts, relate_node_times)

    geva = pd.read_csv(
        "/home/wilderwohns/tsinfer_geva/atlas.chr20.csv",
        delimiter=",",
        skipinitialspace=True,
        skiprows=4,
    )
    geva_tgp = geva[geva["DataSource"] == "TGP"]
    geva_tgp_consistent = geva_tgp[geva_tgp["AlleleAnc"] == geva_tgp["AlleleRef"]]
    geva = geva_tgp_consistent[["Position", "EstAgeJnt"]]
    geva = geva.set_index("Position")
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
    prior_remove_oldest = pickle.load(
        open("/home/wilderwohns/tsdate/timepoints_50_prior_snipped_notrim.p", "rb")
    )
    posterior_remove_oldest = pickle.load(open("tsdate/posterior.p", "rb"))
    mn_remove_oldest, vr_remove_oldest = posterior_mean_var(
        tgp_chr20,
        prior_remove_oldest.timepoints,
        posterior_remove_oldest,
        fixed_node_set=set(tgp_chr20.samples()),
    )
    posterior_mut_ages, posterior_upper_bound = get_mut_ages(
        tgp_chr20, mn_remove_oldest
    )
    tsdate_ages = pd.DataFrame(
        {
            "position": tgp_chr20.tables.sites.position[
                tgp_chr20.tables.mutations.site
            ],
            "tsdate_age": 20000 * posterior_mut_ages,
            "tsdate_upper_bound": 20000 * posterior_upper_bound,
            "frequency": tgp_chr20.tables.nodes.time[tgp_chr20.tables.mutations.node]
            / 5008,
        }
    )
    tsdate_ages = tsdate_ages.set_index("position")
    tsdate_ages.to_csv("all-data/tsdate_ages_tgp_chr20.csv")

    merged = pd.merge(tsdate_ages, geva, left_index=True, right_index=True)
    relate_ages = pd.DataFrame(
        {
            "position": relate_ts.tables.sites.position[
                relate_ts.tables.mutations.site
            ],
            "relate_age": relate_mut_ages,
            "relate_upper_bound": relate_mut_upper_bound,
        }
    )
    relate_ages = relate_ages.set_index("position")
    merged = pd.merge(merged, relate_ages, left_index=True, right_index=True)
    merged.to_csv("all-data/tgp_mutations.csv")


def get_mut_ages(ts, dates):
    mut_ages = list()
    mut_upper_bounds = list()
    for tree in tqdm(ts.trees(), total=ts.num_trees):
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
    positions = ancient_samples.sites_position[:]
    genotypes = ancient_samples.sites_genotypes[:]
    lower_bound = [
        (pos, np.max(ages[np.where(geno == 1)[0]]))
        for pos, geno in zip(positions, genotypes)
    ]
    constraint_df = pd.DataFrame(lower_bound, columns=["Position", "Ancient Bound"])
    constraint_df = constraint_df.set_index("Position")
    try:
        tgp_mut_ests = pd.read_csv("all-data/tgp_mutations.csv", index_col=0)
    except:
        raise ValueError("tgp_mutations.csv does not exist. Must run tgp_dates first")
    tgp_muts_constraints = pd.merge(
        tgp_mut_ests, constraint_df, left_index=True, right_index=True
    )
    tgp_muts_constraints.to_csv("all-data/tgp_muts_constraints")


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
