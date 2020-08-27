"""
Useful functions used in multiple scripts.
"""

import numpy as np
import pandas as pd

import tskit


def get_mut_ages(ts, dates):
    mut_ages = list()
    mut_upper_bounds = list()
    for tree in ts.trees():
        for site in tree.sites():
            for mut in site.mutations:
                parent_age = dates[tree.parent(mut.node)]
                mut_upper_bounds.append(parent_age)
                mut_ages.append((dates[mut.node] + parent_age) / 2)
    return np.array(mut_ages), np.array(mut_upper_bounds)


def get_mut_ages_dict(ts, dates, exclude_root=False):
    site_ages = dict()
    for tree in ts.trees():
        for site in tree.sites():
            site_ages[site.position] = (-np.inf, -np.inf)
            for mut in site.mutations:
                parent_node = tree.parent(mut.node)
                if exclude_root and parent_node == ts.num_nodes - 1:
                    continue
                else:
                    parent_age = dates[parent_node]
                    mut_age = [(np.sqrt(dates[mut.node] * parent_age)), mut.node]
                    if site_ages[site.position][0] < mut_age[0]:
                        site_ages[site.position] = mut_age
    return site_ages 

def get_mut_pos_df(ts, name, node_dates, exclude_root=False):
    mut_dict = get_mut_ages_dict(ts, node_dates, exclude_root=exclude_root) 
    mut_df = pd.DataFrame.from_dict(mut_dict, orient="index", columns=[name, "Node"])
    mut_df.index = (np.round(mut_df.index)).astype(int)
    sort_dates = mut_df.sort_values(by=[name], ascending=False, kind="mergesort")
    mut_df = sort_dates.groupby(sort_dates.index).first()
#    mut_df = mut_df.loc[~mut_df.index.duplicated()]
    return mut_df

