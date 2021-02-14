"""
Useful functions used in multiple scripts.
"""

import numpy as np
import pandas as pd

import tskit
import tsdate


def sites_time_from_ts(
    tree_sequence, *, node_selection="child", exclude_root=True
):
    if tree_sequence.num_sites < 1:
        raise ValueError("Invalid tree sequence: no sites present")
    if node_selection not in ["arithmetic", "geometric", "child", "parent", "msprime"]:
        raise ValueError(
            "The node_selection parameter must be "
            "'child', 'parent', 'arithmetic', 'geometric', or 'msprime'"
        )
    nodes_time = tree_sequence.tables.nodes.time
    sites_time = np.full(tree_sequence.num_sites, np.nan)

    for tree in tree_sequence.trees():
        for site in tree.sites():
            for mutation in site.mutations:
                parent_node = tree.parent(mutation.node)
                if node_selection == "child" or parent_node == tskit.NULL:
                    age = nodes_time[mutation.node]
                else:
                    parent_age = nodes_time[parent_node]
                    if node_selection == "parent":
                        age = parent_age
                    elif node_selection == "arithmetic":
                        age = (nodes_time[mutation.node] + parent_age) / 2
                    elif node_selection == "geometric":
                        age = np.sqrt(nodes_time[mutation.node] * parent_age)
                    elif node_selection == "msprime":
                        age = mutation.time
                if np.isnan(sites_time[site.id]) or sites_time[site.id] < age:
                    sites_time[site.id] = age
                    if exclude_root and parent_node == tree_sequence.num_nodes - 1:
                        sites_time[site.id] = np.nan
    return sites_time


def get_mut_pos_df(ts, name, node_dates, node_selection="arithmetic", exclude_root=False):
    if node_selection == "msprime":
        sites_time = sites_time_from_ts(ts, node_selection="msprime", exclude_root=False)
    else:
        if exclude_root:
            sites_time = sites_time_from_ts(
                ts, node_selection=node_selection, exclude_root=True
            )
        else:
            sites_time = tsdate.sites_time_from_ts(
                ts, node_selection=node_selection, unconstrained=False, min_time=0
            )
    positions = ts.tables.sites.position
    mut_dict = dict(zip(positions, sites_time))
    mut_df = pd.DataFrame.from_dict(mut_dict, orient="index", columns=[name])
    mut_df.index = (np.round(mut_df.index)).astype(int)
    # Remove duplicates
    mut_df = mut_df[~mut_df.index.duplicated(keep="first")]
    return mut_df


def weighted_geographic_center(lat_list, long_list, weights):
    x = list()
    y = list()
    z = list()
    if len(lat_list) == 1 and len(long_list) == 1:
        return (lat_list[0], long_list[0])
    lat_radians = np.radians(lat_list)
    long_radians = np.radians(long_list)
    x = np.cos(lat_radians) * np.cos(long_radians)
    y = np.cos(lat_radians) * np.sin(long_radians)
    z = np.sin(lat_radians)
    weights = np.array(weights)
    central_latitude, central_longitude = radians_center_weighted(x, y, z, weights)
    return (np.degrees(central_latitude), np.degrees(central_longitude))


def radians_center_weighted(x, y, z, weights):
    total_weight = np.sum(weights)
    weighted_avg_x = np.sum(weights * np.array(x)) / total_weight
    weighted_avg_y = np.sum(weights * np.array(y)) / total_weight
    weighted_avg_z = np.sum(weights * np.array(z)) / total_weight
    central_longitude = np.arctan2(weighted_avg_y, weighted_avg_x)
    central_square_root = np.sqrt(
        weighted_avg_x * weighted_avg_x + weighted_avg_y * weighted_avg_y
    )
    central_latitude = np.arctan2(weighted_avg_z, central_square_root)
    return central_latitude, central_longitude


def vectorized_weighted_geographic_center(lat_arr, long_arr, weights):
    lat_arr = np.radians(lat_arr)
    long_arr = np.radians(long_arr)
    x = np.cos(lat_arr) * np.cos(long_arr)
    y = np.cos(lat_arr) * np.sin(long_arr)
    z = np.sin(lat_arr)
    if len(weights.shape) > 1:
        total_weights = np.sum(weights, axis=1)
    else:
        total_weights = np.sum(weights)
    weighted_avg_x = np.sum(weights * x, axis=1) / total_weights
    weighted_avg_y = np.sum(weights * y, axis=1) / total_weights
    weighted_avg_z = np.sum(weights * z, axis=1) / total_weights
    central_longitude = np.arctan2(weighted_avg_y, weighted_avg_x)
    central_sqrt = np.sqrt((weighted_avg_x ** 2) + (weighted_avg_y ** 2))
    central_latitude = np.arctan2(weighted_avg_z, central_sqrt)
    return np.degrees(central_latitude), np.degrees(central_longitude)


def add_grand_mrca(ts):
    """
    Function to add a grand mrca node to a tree sequence
    """
    grand_mrca = ts.max_root_time + 1
    tables = ts.dump_tables()
    new_node_number = tables.nodes.add_row(time=grand_mrca)
    for tree in ts.trees():
        tables.edges.add_row(
            tree.interval[0], tree.interval[1], new_node_number, tree.root
        )
    tables.sort()
    return tables.tree_sequence()

