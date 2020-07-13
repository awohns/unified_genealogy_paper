import argparse
import pandas as pd
import numpy as np
import tskit
import tsinfer

import constants


def snip_root_node(ts):
    """
    Remove the oldest root node of the tree sequence and any edges 
    where it is the parent.
    """
    #    if not (0 < left < right < ts.sequence_length):
    #        raise ValueError("Invalid centromere coordinates")
    tables = ts.dump_tables()
    root_node = ts.num_nodes - 1
    #    if len(tables.sites) > 0:
    #        position = tables.sites.position
    #        left_index = np.searchsorted(position, left)
    #        right_index = np.searchsorted(position, right)
    #        if right_index != left_index:
    #            raise ValueError("Cannot have sites defined within the centromere")

    edges = tables.edges.copy()
    # Get all edges that do not lead to parent and add them
    # index = np.logical_or(right <= edges.left, left >= edges.right)
    index = np.where(edges.parent != root_node)[0]
    tables.edges.set_columns(
        left=edges.left[index],
        right=edges.right[index],
        parent=edges.parent[index],
        child=edges.child[index],
    )
    # # Get all edges that intersect and add two edges for each.
    # index = np.logical_not(index)
    # i_parent = edges.parent[index]
    # i_child = edges.child[index]
    # i_left = edges.left[index]
    # i_right = edges.right[index]

    #    # Only insert valid edges (remove any entirely lost topology)
    #    index = i_left < left
    #    num_intersecting = np.sum(index)
    #    tables.edges.append_columns(
    #        left=i_left[index],
    #        right=np.full(num_intersecting, left, dtype=np.float64),
    #        parent=i_parent[index],
    #        child=i_child[index],
    #    )
    #
    #    # Only insert valid edges (remove any entirely lost topology)
    #    index = right < i_right
    #    num_intersecting = np.sum(index)
    #    tables.edges.append_columns(
    #        left=np.full(num_intersecting, right, dtype=np.float64),
    #        right=i_right[index],
    #        parent=i_parent[index],
    #        child=i_child[index],
    #    )
    tables.sort()
    # record = provenance.get_provenance_dict(
    #    command="remove_oldest_root",
    # )
    # tables.provenances.add_row(record=json.dumps(record))
    return tables.tree_sequence()


def reinfer_1kg(input_fn, output_fn):
    muts_constraints = pd.read_csv("all-data/tgp_muts_constraints.csv", index_col=0)
    # Find the maximum of the estimated tsdate age and the ancient constraints for each mutation
    muts_constraints["constrained_tsdate"] = np.maximum(
        muts_constraints["tsdate_age"] * constants.GENERATION_TIME,
        muts_constraints["Ancient Bound"],
    )
    muts_constraints = muts_constraints["constrained_tsdate"]
    muts = pd.read_csv("all-data/tsdate_ages_tgp_chr20.csv")
    muts = muts.set_index("position")
    dates_all = pd.merge(
        muts, muts_constraints, how="left", left_index=True, right_index=True
    )
    # Use tsdate estimated time wherever constraint does not have a value
    dates_all_constrained = dates_all["constrained_tsdate"].fillna(
        dates_all["tsdate_age"] * constants.GENERATION_TIME
    )
    chr20_samples = tsinfer.load(input_fn)
    chr20_samples_copy = chr20_samples.copy(output_fn)
    samples_index_bool = np.isin(dates_all.index, chr20_samples.sites_position[:])
    # Round the age to the nearest 10. This will speed up inference with minimal cost to precision
    chr20_samples_copy.sites_time[:] = np.round(
        dates_all_constrained[samples_index_bool].values, -1
    )
    chr20_samples_copy.finalise()


def main():
    name_map = {"reinfer_1kg": reinfer_1kg}

    parser = argparse.ArgumentParser(
        description="Process the human data and make data files for plotting."
    )
    parser.add_argument(
        "name", type=str, help="figure name", choices=list(name_map.keys())
    )
    parser.add_argument("input_fn", type=str, help="input sampledata file")
    parser.add_argument("output_fn", type=str, help="output filename")

    args = parser.parse_args()
    name_map[args.name](args.input_fn, args.output_fn)


if __name__ == "__main__":
    main()
