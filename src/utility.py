"""
Useful functions used in multiple scripts.
"""

import numpy as np

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
