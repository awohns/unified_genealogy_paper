#!/usr/bin/env python3
"""
Generates all the actual figures. Run like
 python3 src/plot.py PLOT_NAME
"""
import argparse
import os
import pickle
from operator import attrgetter

import scipy
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_log_error
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, mark_inset,
                                                   zoomed_inset_axes)
import matplotlib.colors as mplc
import colorcet as cc

import constants


class Figure(object):
    """
    Superclass for creating figures. Each figure is a subclass
    """

    name = None
    data_path = None
    filename = None

    def __init__(self):
        self.data = list()
        for fn in self.filename:
            datafile_name = os.path.join(self.data_path, fn + ".csv")
            self.data.append(pd.read_csv(datafile_name))

    def save(self, figure_name=None, bbox_inches="tight"):
        if figure_name is None:
            figure_name = self.name
        print("Saving figure '{}'".format(figure_name))
        plt.savefig("figures/{}.pdf".format(figure_name), bbox_inches="tight", dpi=400)
        plt.savefig("figures/{}.png".format(figure_name), bbox_inches="tight", dpi=400)
        plt.close()

    def error_label(self, error, label_for_no_error="No genotyping error"):
        """
        Make a nice label for an error parameter
        """
        try:
            error = float(error)
            return "Error rate = {}".format(error) if error else label_for_no_error
        except (ValueError, TypeError):
            try:  # make a simplified label
                if "Empirical" in error:
                    error = "With genotyping"
            except:
                pass
            return "{} error".format(error) if error else label_for_no_error

    def mutation_accuracy(self, ax, x, y, label, cmap="Blues", kc_distance_0=None, kc_distance_1=None):
        hb = ax.hexbin(x, y, xscale="log", yscale="log", bins="log",
                  cmap=cmap, mincnt=1)
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        if label is not None:
            ax.set_title(label, fontsize=24, color=cmap[:-1])
        assert len(x) == len(y)
        ax.text(0.05, 0.9, str(len(x)) + " mutations",
            transform=ax.transAxes, size=14)
        ax.text(0.05, 0.85, "RMSLE: " + "{0:.2f}".format(
            np.sqrt(mean_squared_log_error(x, y))),
            transform=ax.transAxes, size=14)
        ax.text(0.05, 0.8, "Pearson's r: " + "{0:.2f}".format(
            scipy.stats.pearsonr(np.log(x), np.log(y))[0]),
            transform=ax.transAxes, size=14)
        ax.text(0.05, 0.75, "Spearman's $\\rho$: " + "{0:.2f}".format(
            scipy.stats.spearmanr(x, y)[0]),
            transform=ax.transAxes, size=14)
        ax.text(0.05, 0.7, "Bias:" + "{0:.2f}".format(
            np.mean(y - x)),
            transform=ax.transAxes, size=14)
        if kc_distance_0 is not None:
            ax.text(0.3, 0.11, "KC Dist. ($\lambda$=0):" +
                                "{:.2E}".format(kc_distance_0),
                                transform=ax.transAxes, size=14)
        if kc_distance_1 is not None:
            ax.text(0.3, 0.03, "KC Dist. ($\lambda$=1):" +
                                "{:.2E}".format(kc_distance_1),
                                transform=ax.transAxes, size=14)
        return hb


class TsdateSimulatedAccuracyNeutral(Figure):
    """
    For Figure 1: accuracy of tsdate on simulated data under a neutral model.
    Compares age of mutations: simulated time vs. tsdate estimation using
    simulated topology and tsdate using tsinfer inferred topologies.
    """

    name = "tsdate_simulated_accuracy"
    data_path = "simulated-data"
    filename = ["tsdate_neutral_simulated_mutation_accuracy_mutations"]

    def plot(self):
        df = self.data[0]
        fig, ax = plt.subplots(
            nrows=1, ncols=2, figsize=(12, 6), sharex=True, sharey=True
        )
        df = df[df["simulated_ts"] > 0]
        true_vals = df["simulated_ts"]
        tsdate = df["tsdate"]
        tsdate_inferred = df["tsdate_inferred"]

        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_xlim(1, 2e5)
        ax[0].set_ylim(1, 2e5)

        # tsdate on true tree
        self.mutation_accuracy(ax[0], true_vals, tsdate, None, cmap=None)
        ax[0].set_title("tsdate (using true topology)", fontsize=24)

        # tsdate on inferred tree
        hb = self.mutation_accuracy(ax[1], true_vals, tsdate_inferred, None, cmap=None)
        ax[1].set_title("tsdate + tsdate", fontsize=24)
        fig.subplots_adjust(right=0.9)
        colorbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
        cb = fig.colorbar(hb, cax=colorbar_ax)
        cb.set_label('log10(Number of Mutations)')
        fig.text(0.5, 0.03, "True Mutation Ages (Generations)", size=20, ha="center")
        fig.text(0.03, 0.5, "Estimated Mutation \n Ages (Generations)", size=20, 
                va="center", rotation="vertical")
        self.save(self.name)


class IterateNoAncients(Figure):
    """
    Figure to show accuracy of iterative approach.
    """

    name = "iterative_accuracy_noancients"
    data_path = "simulated-data"
    filename = ["iterate_no_ancients_mutations"]

    def plot(self):
        df = self.data[0]
        tsdate = df["tsdateTime"]
        iteration_times = df["IterationTime"]
        keep_times = df["tsinfer_keep_time"]
        topo_times = df["SimulatedTopoTime"]

        plt.errorbar(
            [
                "Modern Dated",
                "With Iteration",
                "with tsinfer real times",
                "With real ages",
            ],
            [
                np.mean(tsdate),
                np.mean(iteration_times),
                np.mean(keep_times),
                np.mean(topo_times),
            ],
            yerr=[
                np.std(tsdate),
                np.std(iteration_times),
                np.std(keep_times),
                np.std(topo_times),
            ],
        )
        plt.xticks(rotation=45)
        plt.ylabel("Mean Squared Log Error")
        self.save(self.filename[0])

    def plot_full(self):
        df = self.data[0][0:10000]

        with sns.axes_style("white"):
            fig, ax = plt.subplots(
                nrows=2, ncols=2, figsize=(12, 6), sharex=True, sharey=True
            )
            true_vals = df["SimulatedTime"]
            tsdate = df["SimulatedTopoTime"]
            tsdate_inferred = df["tsdateTime"]
            keep_times = df["tsinfer_keep_time"]
            iteration_times = df["IterationTime"]

            ax[0, 0].set_xscale("log")
            ax[0, 0].set_yscale("log")
            ax[0, 0].set_xlim(1, 2e5)
            ax[0, 0].set_ylim(1, 2e5)

            # tsdate on true tree
            x = true_vals
            y = tsdate
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            z = z / np.max(z)
            ax[0, 0].scatter(x, y, c=z, s=5, cmap=plt.cm.plasma)
            ax[0, 0].plot(ax[0, 0].get_xlim(), ax[0, 0].get_ylim(), ls="--", c=".3")
            ax[0, 0].set_title("tsdate (using true topology)", fontsize=24)
            ax[0, 0].text(
                0.36,
                0.16,
                "RMSLE:" + "{0:.2f}".format(np.sqrt(mean_squared_log_error(x, y))),
                fontsize=19,
                transform=ax[0, 0].transAxes,
            )
            ax[0, 0].text(
                0.36,
                0.1,
                "Spearman's "
                + "$\\rho$: "
                + "{0:.2f}".format(scipy.stats.spearmanr(x, y)[0]),
                fontsize=19,
                transform=ax[0, 0].transAxes,
            )
            ax[0, 0].text(
                0.36,
                0.04,
                "Pearson's r: " + "{0:.2f}".format(scipy.stats.pearsonr(x, y)[0]),
                fontsize=19,
                transform=ax[0, 0].transAxes,
            )

            # tsdate on inferred tree
            y = tsdate_inferred
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            z = z / np.max(z)
            scatter = ax[0, 1].scatter(x, y, c=z, s=5, cmap=plt.cm.plasma)
            ax[0, 1].plot(ax[0, 1].get_xlim(), ax[0, 1].get_ylim(), ls="--", c=".3")
            ax[0, 1].set_title("tsinfer + tsdate", fontsize=24)
            ax[0, 1].text(
                0.36,
                0.16,
                "RMSLE:" + "{0:.2f}".format(np.sqrt(mean_squared_log_error(x, y))),
                fontsize=19,
                transform=ax[0, 1].transAxes,
            )
            ax[0, 1].text(
                0.36,
                0.1,
                "Spearman's "
                + "$\\rho$: "
                + "{0:.2f}".format(scipy.stats.spearmanr(x, y)[0]),
                fontsize=19,
                transform=ax[0, 1].transAxes,
            )
            ax[0, 1].text(
                0.36,
                0.04,
                "Pearson's r: " + "{0:.2f}".format(scipy.stats.pearsonr(x, y)[0]),
                fontsize=19,
                transform=ax[0, 1].transAxes,
            )

            # tsdate on inferred tree
            y = keep_times
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            z = z / np.max(z)
            scatter = ax[1, 0].scatter(x, y, c=z, s=5, cmap=plt.cm.plasma)
            ax[1, 0].plot(ax[1, 0].get_xlim(), ax[1, 0].get_ylim(), ls="--", c=".3")
            ax[1, 0].set_title("tsinfer + tsdate", fontsize=24)
            ax[1, 0].text(
                0.36,
                0.16,
                "RMSLE:" + "{0:.2f}".format(np.sqrt(mean_squared_log_error(x, y))),
                fontsize=19,
                transform=ax[1, 0].transAxes,
            )
            ax[1, 0].text(
                0.36,
                0.1,
                "Spearman's "
                + "$\\rho$: "
                + "{0:.2f}".format(scipy.stats.spearmanr(x, y)[0]),
                fontsize=19,
                transform=ax[1, 0].transAxes,
            )
            ax[1, 0].text(
                0.36,
                0.04,
                "Pearson's r: " + "{0:.2f}".format(scipy.stats.pearsonr(x, y)[0]),
                fontsize=19,
                transform=ax[1, 0].transAxes,
            )

            # tsdate iterative
            y = iteration_times
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            z = z / np.max(z)
            scatter = ax[1, 1].scatter(x, y, c=z, s=5, cmap=plt.cm.plasma)
            ax[1, 1].plot(ax[1, 1].get_xlim(), ax[1, 1].get_ylim(), ls="--", c=".3")
            ax[1, 1].set_title("tsinfer + tsdate", fontsize=24)
            ax[1, 1].text(
                0.36,
                0.16,
                "RMSLE:" + "{0:.2f}".format(np.sqrt(mean_squared_log_error(x, y))),
                fontsize=19,
                transform=ax[1, 1].transAxes,
            )
            ax[1, 1].text(
                0.36,
                0.1,
                "Spearman's "
                + "$\\rho$: "
                + "{0:.2f}".format(scipy.stats.spearmanr(x, y)[0]),
                fontsize=19,
                transform=ax[1, 1].transAxes,
            )
            ax[1, 1].text(
                0.36,
                0.04,
                "Pearson's r: " + "{0:.2f}".format(scipy.stats.pearsonr(x, y)[0]),
                fontsize=19,
                transform=ax[1, 1].transAxes,
            )
            fig.text(
                0.5, 0.03, "True Mutation Ages (Generations)", size=23, ha="center"
            )
            fig.text(
                0.03,
                0.5,
                "Estimated Mutation \n Ages (Generations)",
                size=23,
                va="center",
                rotation="vertical",
            )

            cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
            cbar = fig.colorbar(scatter, cax=cbar_ax)
            cbar.set_label("Density", fontsize=15, rotation=270, labelpad=20)

            self.save("iterative_accuracy_noancients")


class Figure2Ancients(Figure):
    """
    Main text figure 2. Accuracy of increasing number of ancient samples.
    """

    name = "iteration_ancients"
    data_path = "simulated-data"
    filename = ["chr20_ancient_iteration_msle", "chr20_ancient_iteration_spearman",
                "chr20_ancient_iteration_kc", "chr20_ancient_iteration_ooa_msle",
                "chr20_ancient_iteration_ooa_spearman", "chr20_ancient_iteration_ooa_kc",
                "chr20_ancient_iteration_amh_msle", "chr20_ancient_iteration_amh_spearman",
                "chr20_ancient_iteration_amh_kc"]
    plt_title = "iteration_ancients"

    def __init__(self):
        super().__init__()

    def plot(self):
        msle = self.data[0]
        spearman = self.data[1]
        kc = self.data[2]
        kc = kc.set_index(kc.columns[0])
        msle_ooa = self.data[3]
        spearman_ooa = self.data[4]
        kc_ooa = self.data[5]
        kc_ooa = kc_ooa.set_index(kc_ooa.columns[0])
        msle_amh = self.data[6]
        spearman_amh = self.data[7]
        kc_amh = self.data[8]
        kc_amh = kc_amh.set_index(kc_amh.columns[0])
        #muts = pd.merge([self.data[0], self.data[2].add_suffix("_ooa"), self.data[4].add_suffix("_amh"))
        #spearman = pd.merge([self.data[1], self.data[3].add_suffix("_ooa"), self.data[5].add_suffix("_amh")])
        #muts = self.data[0]
        #muts_amh = self.data[2]
        #spearman = self.data[1]
        widths = [0.5, 0.5, 3]
        heights = [3, 3]
        gs_kw = dict(width_ratios=widths, height_ratios=heights)
        gs_kw.update(wspace=0.03)
        fig, ax = plt.subplots(
            ncols=3, nrows=2, constrained_layout=True, gridspec_kw=gs_kw, sharey="row"
        )
        
        df = msle
        comb_df = pd.concat([msle, msle_ooa, msle_amh])
        sns.boxplot(x=comb_df["tsdate_inferred"], orient="v", ax=ax[0,0], color="silver")
        sns.boxplot(x=comb_df["iter_dated_inferred"], orient="v", ax=ax[0,1], color="silver")
        cols = ["Subset " + str(subset) for subset in [1, 5, 10, 20, 40]]
        df_melt = df.melt(value_vars=cols)
        df_melt["variable"] = df_melt["variable"].str.split().str[-1]

        sns.lineplot(
            x="variable",
            y="value",
            data=df_melt,
            sort=False,
            ax=ax[0,2],
            alpha=0.8,
            color="grey",
        )
        groupby = df_melt.groupby("variable").mean()
        ax[0, 2].scatter(groupby.index, groupby["value"], s=80, color="black", zorder=3, alpha=0.8)

        df_melt = msle_ooa.melt(value_vars=cols)
        df_melt["variable"] = df_melt["variable"].str.split().str[-1]
        sns.lineplot(
            x="variable",
            y="value",
            data=df_melt,
            sort=False,
            ax=ax[0, 2],
            alpha=0.7,
            color="grey"
        )
        groupby = df_melt.groupby("variable").mean()
        ax[0, 2].scatter(groupby.index, groupby["value"], s=80, marker="X", color="black", zorder=3, alpha=0.8)
        df_melt = msle_amh.melt(value_vars=cols)
        df_melt["variable"] = df_melt["variable"].str.split().str[-1]
        sns.lineplot(
            x="variable",
            y="value",
            data=df_melt,
            sort=False,
            ax=ax[0, 2],
            alpha=0.7,
            color="grey"
        )
        groupby = df_melt.groupby("variable").mean()
        ax[0, 2].scatter(groupby.index, groupby["value"], s=80, marker="P", color="black", zorder=3, alpha=0.8)

        comb_df = pd.concat([spearman, spearman_ooa, spearman_amh])

        sns.boxplot(x=comb_df["inferred"], orient="v", ax=ax[1,0], color="silver")
        sns.boxplot(x=comb_df["reinferred"], orient="v", ax=ax[1,1], color="silver")

        cols = ["Subset " + str(subset) for subset in [1, 5, 10, 20, 40]]

        df_melt = spearman.melt(value_vars=cols)
        df_melt["variable"] = df_melt["variable"].str.split().str[-1]
        sns.lineplot(
            x="variable",
            y="value",
            data=df_melt,
            sort=False,
            ax=ax[1, 2],
            alpha=0.8,
            color="grey"
        )
        groupby = df_melt.groupby("variable").mean()
        ax[1, 2].scatter(groupby.index, groupby["value"], s=80, color="black", zorder=3, alpha=0.8)
        df_melt = spearman_ooa.melt(value_vars=cols)
        df_melt["variable"] = df_melt["variable"].str.split().str[-1]
        sns.lineplot(
            x="variable",
            y="value",
            data=df_melt,
            sort=False,
            ax=ax[1, 2],
            alpha=0.7,
            color="grey"
        )
        groupby = df_melt.groupby("variable").mean()
        ax[1, 2].scatter(groupby.index, groupby["value"], s=80, marker="X", color="black", zorder=3, alpha=0.8)
        df_melt = spearman_amh.melt(value_vars=cols)
        df_melt["variable"] = df_melt["variable"].str.split().str[-1]
        sns.lineplot(
            x="variable",
            y="value",
            data=df_melt,
            sort=False,
            ax=ax[1, 2],
            alpha=0.7,
            color="grey"
        )
        groupby = df_melt.groupby("variable").mean()
        ax[1, 2].scatter(groupby.index, groupby["value"], s=80, marker="P", color="black", zorder=3, alpha=0.8)

        ax[0, 1].set_ylabel("")
        ax[0, 0].set_ylabel("Mean Squared Log Error")
        ax[0, 2].set_xlabel("")
        ax[0, 1].tick_params(left="off")
        ax[0, 2].tick_params(labelbottom=False)
        ax[1, 1].set_ylabel("")
        ax[1, 0].set_ylabel("Spearman's $\\rho$")
        ax[1, 2].set_xlabel("Ancient Sample Size")
        ax[1, 1].tick_params(left="off")
        ax[1, 2].tick_params(left="off")
        ax[0, 0].set_title("i")
        ax[0, 1].set_title("ii")
        ax[0, 2].set_title("iii")

        #plt.suptitle("Mutation Estimation Accuracy: " + self.plt_title)
        self.save(self.name)


class Figure2(Figure):
    """
    Main text figure 2. Accuracy of increasing number of ancient samples.
    """

    name = "iteration_eval"
    data_path = "simulated-data"
    filename = ["chr20_ancient_iteration_msle", "chr20_ancient_iteration_spearman",
                "chr20_ancient_iteration_amh_msle", "chr20_ancient_iteration_amh_spearman"]
    plt_title = "iteration_eval"

    def __init__(self):
        super().__init__()

    def plot(self):
        muts = pd.merge([self.data[0], self.data[2].add_suffix("_amh")])
        spearman = pd.merge([self.data[1], self.data[3].add_suffix("_amh")])
        #muts = self.data[0]
        #muts_amh = self.data[2]
        #spearman = self.data[1]
        widths = [0.5, 0.5, 3, 0.5]
        heights = [3, 3]
        gs_kw = dict(width_ratios=widths, height_ratios=heights)
        gs_kw.update(wspace=0.03)
        fig, ax = plt.subplots(
            ncols=4, nrows=2, constrained_layout=True, gridspec_kw=gs_kw, sharey="row"
        )
        ax[0, 1].set_ylabel("")
        ax[0, 0].set_ylabel("Mean Squared Log Error")
        ax[0, 2].set_xlabel("Ancient Sample Size")
        ax[0, 3].set_ylabel("")
        ax[0, 1].tick_params(left="off")
        ax[0, 2].tick_params(left="off")
        ax[0, 3].tick_params(left="off")
        ax[0, 0].set_title("i")
        ax[0, 1].set_title("ii")
        ax[0, 2].set_title("iii")
        ax[0, 3].set_title("iv")

        for row, df in enumerate([spearman, muts]):
            if row == 1:
                sns.boxplot(x=df["tsdate_inferred"], orient="v", ax=ax[row, 0])
            else:
                sns.boxplot(x=df["inferred"], orient="v", ax=ax[row, 0])
            if row == 1:
                sns.boxplot(
                    x=df["iter_dated_inferred"],
                    orient="v",
                    ax=ax[row, 1],
                )
            else:
                sns.boxplot(
                    x=df["reinferred"],
                    orient="v",
                    ax=ax[row, 1],
                )

            cols = ["Subset " + str(subset) for subset in [1, 5, 10, 20, 40]]
            df_melt = df.melt(value_vars=cols)
            df_melt["variable"] = df_melt["variable"].str.split().str[-1]
            sns.lineplot(
                x="variable",
                y="value",
                data=df_melt,
                ax=ax[row, 2],
            )
            # ax = sns.violinplot(x="ancient_sample_size", y="tsinfer_keep_time", data=muts)
            #sns.boxplot(x=muts["tsinfer_keep_time"], orient="v", ax=ax[3])
            # ax[0].set_xlabel("Date \nTree Seq")
            # ax[0].set_xticklabels(["Date \nTree Sequence"])
        plt.suptitle("Mutation Estimation Accuracy: " + self.plt_title)
        self.save(self.name)


class IterateAncientsVanillaMsle(Figure):
    """
    Figure to show accuracy of iterative approach with ancient samples
    and vanilla demographic model. Plots MSLE results.
    """

    name = "iterate_ancients_vanilla_msle"
    data_path = "simulated-data"
    filename = ["simulate_vanilla_ancient_mutations.msle"]
    plt_title = "Vanilla Simulations MSLE"

    def __init__(self):
        super().__init__()
        self.data = self.data[0]

    def plot(self):
        muts = self.data
        widths = [0.5, 0.5, 3, 0.5]
        heights = [3]
        gs_kw = dict(width_ratios=widths, height_ratios=heights)
        gs_kw.update(wspace=0.03)
        fig, ax = plt.subplots(
            ncols=4, nrows=1, constrained_layout=True, gridspec_kw=gs_kw, sharey=True
        )
        sns.boxplot(x=muts["tsdateTime"], orient="v", ax=ax[0])

        sns.boxplot(
            x=muts[muts["ancient_sample_size"] == 0]["IterationTime"],
            orient="v",
            ax=ax[1],
        )
        sns.lineplot(
            x="ancient_sample_size",
            y="IterationTime",
            data=muts[muts["ancient_sample_size"] != 0],
            ax=ax[2],
        )
        # ax = sns.violinplot(x="ancient_sample_size", y="tsinfer_keep_time", data=muts)
        sns.boxplot(x=muts["tsinfer_keep_time"], orient="v", ax=ax[3])
        # ax[0].set_xlabel("Date \nTree Seq")
        # ax[0].set_xticklabels(["Date \nTree Sequence"])
        ax[1].set_ylabel("")
        ax[2].set_xlim(0, 100)
        ax[0].set_ylabel("Mean Squared Log Error")
        ax[2].set_xlabel("Ancient Sample Size")
        ax[3].set_ylabel("")
        ax[1].tick_params(left="off")
        ax[2].tick_params(left="off")
        ax[3].tick_params(left="off")
        ax[0].set_title("i")
        ax[1].set_title("ii")
        ax[2].set_title("iii")
        ax[3].set_title("iv")
        plt.suptitle("Mutation Estimation Accuracy: " + self.plt_title)
        self.save(self.name)


class IterateAncientsVanillaPearsonR(IterateAncientsVanillaMsle):
    """
    Figure to show accuracy of iterative approach with ancient samples
    and vanilla demographic model. Plots MSLE results.
    """

    name = "iterate_ancients_vanilla_pearsonr"
    data_path = "simulated-data"
    filename = ["simulate_vanilla_ancient_mutations.pearsonr"]
    plt_title = "Vanilla Simulations Pearson R"

    def __init__(self):
        super().__init__()


class IterateAncientsVanillaSpearmanR(IterateAncientsVanillaMsle):
    """
    Figure to show accuracy of iterative approach with ancient samples
    and vanilla demographic model. Plots MSLE results.
    """

    name = "iterate_ancients_vanilla_spearmanr"
    data_path = "simulated-data"
    filename = ["simulate_vanilla_ancient_mutations.spearmanr"]
    plt_title = "Vanilla Simulations Spearman R"

    def __init__(self):
        super().__init__()


class IterateAncientsVanillaMsleError(IterateAncientsVanillaMsle):
    """
    Figure to show accuracy of iterative approach with ancient samples
    and vanilla demographic model. Plots MSLE results with empirical error.
    """

    name = "iterate_ancients_vanilla_msle_error"
    data_path = "simulated-data"
    filename = ["simulate_vanilla_ancient_mutations.msle.empiricalerror"]
    plt_title = "Vanilla Simulations MSLE Empirical Error"

    def __init__(self):
        super().__init__()


class IterateAncientsVanillaPearsonRError(IterateAncientsVanillaMsle):
    """
    Figure to show accuracy of iterative approach with ancient samples
    and vanilla demographic model. Plots Pearson R results with empirical error.
    """

    name = "iterate_ancients_vanilla_pearsonr_error"
    data_path = "simulated-data"
    filename = ["simulate_vanilla_ancient_mutations.pearsonr.empiricalerror"]
    plt_title = "Vanilla Simulations Pearson R Empirical Error"

    def __init__(self):
        super().__init__()


class IterateAncientsVanillaSpearmanRError(IterateAncientsVanillaMsle):
    """
    Figure to show accuracy of iterative approach with ancient samples
    and vanilla demographic model. Plots Spearman R results with empirical error.
    """

    name = "iterate_ancients_vanilla_spearmanr_error"
    data_path = "simulated-data"
    filename = ["simulate_vanilla_ancient_mutations.spearmanr.empiricalerror"]
    plt_title = "Vanilla Simulations Spearman R Empirical Error"

    def __init__(self):
        super().__init__()


class IterateAncientsOOA(IterateAncientsVanillaMsle):
    """
    Figure to show accuracy of iterative approach on Chromosome 20.
    Using the Out of Africa Model
    """

    name = "iterate_ancients_ooa"
    data_path = "simulated-data"
    filename = ["ooa_chr20_mutations"]
    plt_title = "Chromosome 20 Out of Africa"

    def __init__(self):
        super().__init__()


class IterateAncientsVanillaKC(Figure):
    """
    Figure to show accuracy of iterative approach with ancient samples
    and vanilla demographic model.
    """

    name = "iterate_ancients_vanilla_kc"
    data_path = "simulated-data"
    filename = ["simulate_vanilla_ancient_kc_distances"]
    plt_title = "KC Distances between Simulated and Inferred Tree Sequences"

    def __init__(self):
        super().__init__()

    def plot(self):
        kc_distances = self.data[0]
        widths = [0.5, 0.5, 3, 0.5]
        heights = [3, 3]
        gs_kw = dict(width_ratios=widths, height_ratios=heights)
        gs_kw.update(wspace=0.03)
        fig, ax = plt.subplots(
            ncols=4, nrows=2, constrained_layout=True, gridspec_kw=gs_kw, sharey="row"
        )
        lambda_0 = kc_distances[kc_distances["lambda_param"] == 0]
        lambda_1 = kc_distances[kc_distances["lambda_param"] == 1]
        for ax_index, lambda_results in zip([0, 1], [lambda_0, lambda_1]):
            sns.boxplot(x=lambda_results[
                lambda_results["ancient_sample_size"] == 0]["tsdateTime"],
                orient="v", ax=ax[ax_index, 0])

            sns.boxplot(
                x=lambda_results[
                    lambda_results["ancient_sample_size"] == 0]["IterationTime"],
                orient="v",
                ax=ax[ax_index, 1],
            )
            sns.lineplot(
                x="ancient_sample_size",
                y="IterationTime",
                data=lambda_results[lambda_results["ancient_sample_size"] != 0],
                ax=ax[ax_index, 2],
            )
            sns.boxplot(x=lambda_results["tsinfer_keep_time"], orient="v",
                        ax=ax[ax_index, 3])
            ax[ax_index, 1].set_ylabel("")
            ax[ax_index, 2].set_xlim(0, 100)
            ax[ax_index, 2].set_xlabel("Ancient Sample Size")
            ax[ax_index, 3].set_ylabel("")
            ax[ax_index, 1].tick_params(left="off")
            ax[ax_index, 2].tick_params(left="off")
            ax[ax_index, 3].tick_params(left="off")
            ax[ax_index, 0].set_title("i")
            ax[ax_index, 1].set_title("ii")
            ax[ax_index, 2].set_title("iii")
            ax[ax_index, 3].set_title("iv")

        ax[0, 0].set_ylabel("KC Distance, Lambda=0")
        ax[1, 0].set_ylabel("KC Distance, Lambda=1")
        plt.suptitle(self.plt_title)
        self.save(self.name)


class IterateAncientsVanillaKC(IterateAncientsVanillaKC):
    """
    Figure to show accuracy of iterative approach with ancient samples
    and vanilla demographic model.
    """

    name = "iterate_ancients_vanilla_kc_error"
    data_path = "simulated-data"
    filename = ["simulate_vanilla_ancient_kc_distances.empiricalerror"]
    plt_title = "KC Distances between Simulated and Inferred Tree Sequences"

    def __init__(self):
        super().__init__()

class AverageAgeAncientMuts(Figure):
    """
    Figure 3 part b: average age of mutations carried by each ancient sample
    """
    name = "mutations_per_ancient"
    data_path = "all-data"
    filename = []

    def plot(self):
        samples = tsinfer.load("1kg_ancients_noreich_chr20")
        min_site_times = samples.min_site_times(individuals_only=True)

class AncientConstraints(Figure):
    """
    Figure 3: Ancient Constraints on Age of Mutations from 1000 Genomes Project
    """
    name = "ancient_constraints_1000g"
    data_path = "all-data"
    filename = ["tgp_muts_constraints"]
    plt_title = "ancient_constraint_1kg"

    def jitter(self, array, log=True):
        max_min = np.max(array) - np.min(array)
        if log:
            return np.exp(
                np.log(array) + np.random.randn(len(array)) * (max_min * 0.0000003))
        else:
            return array + np.random.randn(len(array))

    def plot(self):
        df = self.data[0]
        fig = plt.figure(figsize=(15, 5), constrained_layout=False)
        widths = [3, 3, 3, 0.1]
        spec5 = fig.add_gridspec(ncols=4, nrows=1, width_ratios=widths)
        for a in range(3):
            inner_spec = spec5[a].subgridspec(ncols=2, nrows=1, wspace=0, hspace=0, width_ratios = [1, 10])
            if a==0:
                contemp = fig.add_subplot(inner_spec[0])
                contemp.set_ylim([200, 9e6])
                contemp.set_xlim([-5, 5])
                contemp.set_yscale("log")
                contemp.set_xscale("linear")
                contemp.set_xticks([0])
                contemp.set_xticklabels(["0"])
                ancient = fig.add_subplot(inner_spec[1], sharey=contemp)
                ancient.set_xscale("log")
                ancient.set_xlim([200, 2e5])
                ancient.spines['left'].set_visible(False)
                ancient.yaxis.set_visible(False)
                ax_main = [[contemp, ancient], ]
            else:
                ax_main.append([
                    fig.add_subplot(inner_spec[0], sharey=contemp, sharex=contemp),
                    fig.add_subplot(inner_spec[1], sharey=contemp, sharex=ancient),
                ])
                ax_main[-1][0].set_xticks([0])
                ax_main[-1][0].set_xticklabels(["0"])
                ax_main[-1][1].spines['left'].set_visible(False)
                ax_main[-1][1].yaxis.set_visible(False)
        ax_scale = fig.add_subplot(spec5[3])
        ax_scale.set_yscale("linear")
        
        df_old = df[df["Ancient Bound"] > 0].set_index("Ancient Bound").sort_index()
        df_new = df[np.logical_not(df["Ancient Bound"] > 0)]
        df_old["Ancient Bound Bins"] = pd.cut(df_old.index, 30)
        smoothed_mean = df_old.groupby("Ancient Bound Bins").mean()
        smoothed_mean["bin_right"] = smoothed_mean.index.map(attrgetter('right'))
        smoothed_mean = smoothed_mean.dropna()

        scatter_size = 0.2
        scatter_alpha = 0.2
        shading_alpha = 0.2
        for i, method in enumerate([
            # Hack the titles with extra spaces to centre properly, as it's too tricky
            # to centre over a pair or subplots
            ("tsdate      ",  ["tsdate_upper_bound", "tsdate_age"]),
            ("Relate      ", ["relate_upper_bound", "relate_age"]),
            ("GEVA      ", ["AgeCI95Upper_Jnt", "AgeMean_Jnt"]),
        ]):
            ax = ax_main[i][0]
            ax.scatter(
                self.jitter(np.zeros(len(df_new.index)), log=False),
                constants.GENERATION_TIME * df_new[method[1][1]],
                c=df_new["tsdate_frequency"],
                s=scatter_size,
                alpha=scatter_alpha/6, cmap="plasma_r",
                norm=mplc.LogNorm(vmin=np.min(df_new["tsdate_frequency"]), vmax=1)                
            )
            ax = ax_main[i][1]
            ax.set_title(method[0])
            ax.text(0.1, 0.09, 'Ancient Derived Variant Lower Bound', rotation=36.51,
                    transform=ax.transAxes)
            diag = [ax.get_xlim(), ax.get_xlim()]
            upper_lim = ax.get_ylim()
            ax.plot(diag[0], diag[1], "--", c="black")
            ax.fill_between(diag[0], diag[1], (diag[1][0], diag[1][0]), color="grey",
                             alpha=shading_alpha)
            ax.text(
                0.20,
                0.08,
                "{0:.2f}% variants' estimated upper bound $>=$ ancient lower bound"
                .format(100 / df_old.shape[0] * np.sum((
                    constants.GENERATION_TIME * df_old[method[1][0]]) > df_old.index)),
                fontsize=10,
                transform=ax.transAxes,
            )
            ax.text(
                0.20,
                0.04,
                "{0:.2f}% variants' estimated age $>=$ ancient lower bound"
                .format(100 / df_old.shape[0] * np.sum((
                    constants.GENERATION_TIME * df_old[method[1][1]]) > df_old.index)),
                fontsize=10,
                transform=ax.transAxes,
            )
            scatter = ax.scatter(
                self.jitter(df_old.index),
                constants.GENERATION_TIME * df_old[method[1][1]],
                c=df_old["tsdate_frequency"],
                s=scatter_size,
                alpha=scatter_alpha, cmap="plasma_r",
                norm=mplc.LogNorm(vmin=np.min(df_old["tsdate_frequency"]), vmax=1),
            )
            ax.plot(smoothed_mean["bin_right"].astype(int).values,
                    constants.GENERATION_TIME * smoothed_mean[method[1][1]].values,
                    alpha=0.7, marker="P", color="black")
        fig.text(0.5, 0.01, 'Age of oldest sample with derived allele (years)',
                 ha='center', size=15)
        fig.text(0.08, 0.5, 'Estimated age (years)',
                 va='center', rotation='vertical',
                 size=15)

        cbar = plt.colorbar(
            scatter, format="%.3f", cax=ax_scale, ticks=[0.001, 0.01, 0.1, 0.5, 1])
        cbar.set_alpha(1)
        cbar.draw_all()
        cbar.set_label("Variant Frequency", rotation=270, labelpad=12)
        plt.show()
        self.save(self.name)


class TsdateTooYoung(Figure):
    """
    Examine the mutations that we estimate to be too young from TGP.
    """
    name = "too_young_mutations"
    data_path = "all-data"
    filename = ["tgp_muts_constraints"]
    plt_title = "tsdate_too_young"


class OldestAncientMutations(Figure):
    """
    Examine the mutations observed in the oldest ancient sample.
    """
    name = "oldest_ancient_mutations"
    data_path = "all-data"
    filename = ["tgp_muts_constraints"]
    plt_title = "ancient_constraint_1kg"


class ScalingFigure(Figure):
    """
    Figure showing CPU and memory scaling of tsdate, tsinfer, Relate and GEVA.
    With both samples and length of sequence.
    """
    name = "scaling"
    data_path = "simulated-data"
    filename = ["cpu_scaling_samplesize", "cpu_scaling_length"]
    plt_title = "scaling_fig"

    def plot_subplot(
        self, ax, index, means_arr, time=False, memory=False, samplesize=False, length=False,
        xlabel=False, ylabel=False
    ):
        if memory:
            means_arr = [1e-9 * means for means in means_arr]
            if ylabel:
                ax.set_ylabel("Memory Requirements (Gb)", fontsize=12)
        elif time:
            means_arr = [means * (1 / 3600) for means in means_arr]
            if ylabel:
                ax.set_ylabel("CPU Runtime (hours)", fontsize=12)
        if samplesize and xlabel:
            ax.set_xlabel("Sample Size", fontsize=12)
        elif length and xlabel:
            ax.set_xlabel("Length (Kb)", fontsize=12)
        ax.plot(
            index, means_arr[0], label="tsdate", color=constants.colors["tsdate"]
        )
        ax.plot(
            index, means_arr[1], label="tsinfer", color=constants.colors["tsinfer"]
        )
        ax.plot(
            index,
            means_arr[0] + means_arr[1],
            label="tsinfer +\n tsdate",
            color=constants.colors["tsinfer + tsdate"],
        )
        ax.plot(
            index, means_arr[2], label="relate", color=constants.colors["relate"]
        )
        #ax.plot(index, means_arr[3], label="geva", color=constants.colors["geva"])

    def plot_inset_ax(self, index, ax, means_arr, time=False, memory=False):
        axins1 = inset_axes(ax, width="40%", height="40%", loc=2, borderpad=1)
        if memory:
            means_arr = [1e-9 * means for means in means_arr]
        elif time:
            means_arr = [means * (1 / 3600) for means in means_arr]
        axins1.plot(
            index, means_arr[0], label="tsdate", color=constants.colors["tsdate"]
        )
        axins1.plot(
            index, means_arr[1], label="tsinfer", color=constants.colors["tsinfer"]
        )
        axins1.plot(
            index,
            means_arr[0] + means_arr[1],
            label="tsinfer + tsdate",
            color=constants.colors["tsinfer + tsdate"],
        )
        axins1.plot(
            index, means_arr[2], label="relate", color=constants.colors["relate"]
        )
        #axins1.yaxis.tick_right()

    def plot(self):
        samples_scaling = self.data[0]
        length_scaling = self.data[1]
        samples_means = samples_scaling.groupby("sample_size").mean()
        length_means = length_scaling.groupby("length").mean()
        self.samples_index = samples_means.index
        self.length_index = length_means.index / 1000

        fig, ax = plt.subplots(
            nrows=2, ncols=2, figsize=(18, 6), sharex=False, sharey=False
        )
        self.plot_subplot(
            ax[0, 0],
            self.samples_index,
            [
                samples_means["tsdate_cpu"],
                samples_means["tsinfer_cpu"],
                samples_means["relate_cpu"],
#                    samples_means["geva_cpu"],
            ],
            time=True,
            samplesize=True,
            ylabel=True
        )
#            self.plot_inset_ax(
#                ax[0, 0],
#                [
#                    samples_means["tsdate_cpu"],
#                    samples_means["tsinfer_cpu"],
#                    samples_means["relate_cpu"],
#                ],
#                time=True,
#            )
        self.plot_subplot(
            ax[1, 0],
            self.samples_index,
            [
                samples_means["tsdate_memory"],
                samples_means["tsinfer_memory"],
                samples_means["relate_memory"],
#                    samples_means["geva_memory"],
            ],
            memory=True, samplesize=True, xlabel=True, ylabel=True
        )
#            self.plot_inset_ax(
#                ax[1, 0],
#                [
#                    samples_means["tsdate_memory"],
#                    samples_means["tsinfer_memory"],
#                    samples_means["relate_memory"],
#                ],
#                memory=True,
#            )
        self.plot_subplot(
            ax[0, 1],
            self.length_index,
            [
                length_means["tsdate_cpu"],
                length_means["tsinfer_cpu"],
                length_means["relate_cpu"],
#                    length_means["geva_cpu"],
            ],
            time=True,
            length=True,
        )
        ax[0, 1].get_xaxis().get_major_formatter().set_scientific(False)
#            self.plot_inset_ax(
#                ax[0, 1],
#                [
#                    length_means["tsdate_cpu"],
#                    length_means["tsinfer_cpu"],
#                    length_means["relate_cpu"],
#                ],
#                time=True,
#            )
        self.plot_subplot(
            ax[1, 1],
            self.length_index,
            [
                length_means["tsdate_memory"],
                length_means["tsinfer_memory"],
                length_means["relate_memory"],
#                    length_means["geva_memory"],
            ],
            memory=True, length=True, xlabel=True
        )
        ax[1, 1].get_xaxis().get_major_formatter().set_scientific(False)
#            self.plot_inset_ax(
#                ax[1, 1],
#                [
#                    length_means["tsdate_memory"],
#                    length_means["tsinfer_memory"],
#                    length_means["relate_memory"],
#                ],
#                memory=True,
#            )
        for cur_ax in ax.reshape(-1):
            cur_ax.set_yscale("log")
        ax[0, 0].set_title("Length fixed at 5Mb")
        ax[0, 1].set_title("Sample Size fixed at 1000")
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc=7, fontsize=14, ncol=1)
        self.save(self.name)


class TgpMutEstsFrequency(Figure):
    """
    Figure showing TGP mutation age estimates from tsdate, Relate, GEVA vs. frequency.
    """

    name = "tgp_muts_frequency"
    data_path = "all-data"
    filename = ["tgp_mutations"]
    plt_title = "TGP Mutation Age vs Frequency"

    def plot(self):
        comparable_mutations = self.data[0][
            ["tsdate_age", "relate_age", "AgeMean_Jnt", "tsdate_frequency"]
        ]
        comparable_mutations = comparable_mutations[comparable_mutations["tsdate_age"] > 0]
        frequency = comparable_mutations["tsdate_frequency"]
        fig, ax = plt.subplots(
            nrows=1, ncols=3, figsize=(15, 5), sharey=True, sharex=True)
        ax[0].hexbin(
            frequency,
            comparable_mutations["tsdate_age"], xscale="log", yscale="log",
            bins="log", cmap="Blues", mincnt=1)
        ax[1].hexbin(
            frequency,
            comparable_mutations["relate_age"], xscale="log", yscale="log",
            bins="log", cmap="Greens", mincnt=1)
        ax[2].hexbin(
            frequency,
            comparable_mutations["AgeMean_Jnt"], xscale="log", yscale="log",
            bins="log", cmap="Reds", mincnt=1)
        plt.xlim(3e-3, 1.05)
        plt.ylim(10, 2.4e5)
        ax[0].set_title("Frequency vs. GEVA Estimated Variant Age")
        ax[1].set_title("Frequency vs. Relate Estimated Variant Age")
        ax[2].set_title("Frequency vs. GEVA Estimated Variant Age")
        ax[0].set_xlabel("TGP Frequency")
        ax[1].set_xlabel("TGP Frequency")
        ax[2].set_xlabel("TGP Frequency")
        ax[0].set_ylabel("Estimated Age by tsdate (generations)")
        ax[1].set_ylabel("Estimated Age by Relate (generations)")
        ax[2].set_ylabel("Estimated Age by GEVA (generations)")
        ax[0].plot([0.1, 3e5], [0.1, 3e5], c="black")
        ax[1].plot([0.1, 3e5], [0.1, 3e5], c="black")
        ax[2].plot([0.1, 3e5], [0.1, 3e5], c="black")
        plt.tight_layout()

        self.save(self.name)


class TgpMutationAgeComparisons(Figure):
    """
    Figure comparing TGP mutation age estimates from tsdate, Relate, and GEVA.
    """

    name = "tgp_dates_comparison"
    data_path = "all-data"
    filename = ["tgp_mutations_unconstrained"]
    plt_title = "Compare Mutation Age Estimates"

    def plot(self):
        comparable_mutations = self.data[0][
            ["tsdate_age", "relate_age", "AgeMean_Jnt", "tsdate_frequency"]
        ]
        comparable_mutations = comparable_mutations[comparable_mutations["tsdate_age"] > 0]
        frequency = comparable_mutations["tsdate_frequency"]
        fig, ax = plt.subplots(
            nrows=1, ncols=3, figsize=(15, 5), sharey=True, sharex=True
        )
        ax[0].hexbin(
            comparable_mutations["tsdate_age"],
            comparable_mutations["AgeMean_Jnt"],
            xscale="log", yscale="log",
            bins="log", mincnt=1)

#            c=frequency,
#            norm=mplc.LogNorm(),
#            cmap="plasma_r",
#            s=0.03,
#            alpha=0.03,
#        )
        ax[1].hexbin(
            comparable_mutations["tsdate_age"],
            comparable_mutations["relate_age"],
            xscale="log", yscale="log",
            bins="log", mincnt=1)

#            c=frequency,
#            norm=mplc.LogNorm(),
#            cmap="plasma_r",
#            s=0.03,
#            alpha=0.03,
        ax[2].hexbin(
            comparable_mutations["relate_age"],
            comparable_mutations["AgeMean_Jnt"],
            xscale="log", yscale="log",
            bins="log", mincnt=1)

#            c=frequency,
#            norm=mplc.LogNorm(),
#            cmap="plasma_r",
#            s=0.03,
#            alpha=0.03,
        plt.xlim(1, 2e5)
        plt.ylim(1, 2e5)
        ax[0].set_title("tsdate vs. GEVA Estimated Variant Age")
        ax[1].set_title("tsdate vs. Relate Estimated Variant Age")
        ax[2].set_title("Relate vs. GEVA Estimated Variant Age")
        ax[0].set_xlabel("Estimated Age by tsdate (generations)")
        ax[0].set_ylabel("Estimated Age by GEVA (generations)")
        ax[1].set_xlabel("Estimated Age by tsdate (generations)")
        ax[1].set_ylabel("Estimated Age by Relate (generations)")
        ax[2].set_xlabel("Estimated Age by Relate (generations)")
        ax[2].set_ylabel("Estimated Age by GEVA (generations)")
        ax[0].plot([0.1, 3e5], [0.1, 3e5], c="black")
        ax[1].plot([0.1, 3e5], [0.1, 3e5], c="black")
        ax[2].plot([0.1, 3e5], [0.1, 3e5], c="black")
#        cm = plt.cm.ScalarMappable(
#            cmap="plasma_r",
#            norm=plt.Normalize(vmin=np.min(frequency), vmax=np.max(frequency) + 0.1),
#        )
#        cbar = plt.colorbar(cm, format="%.1f")
#        cbar.set_alpha(1)
#        cbar.draw_all()
#        cbar.set_label("Variant Frequency", rotation=270, labelpad=12)
        plt.tight_layout()

        self.save(self.name)


class TgpMutationAverageAge(Figure):
    """
    Compare mutation age estimates from tsdate, Relate, and GEVA for tgp chromosome 20.
    """

    name = "mutation_average_age"
    data_path = "all-data"
    filename = ["tgp_mutations"]
    plt_title = "Average TGP Mutation Age"

    def plot(self):
        comparable_mutations = self.data[0][["tsdate_age", "relate_age", "AgeMean_Jnt"]]
        comparable_mutations = comparable_mutations[comparable_mutations["tsdate_age"] > 0]
        ax = sns.boxplot(
            data=comparable_mutations.rename(
                columns={
                    "tsdate_age": "tsdate",
                    "relate_age": "relate",
                    "AgeMean_Jnt": "GEVA",
                }
            ),
        )
        ax.set_yscale("log")
        ax.artists[0].set_facecolor("blue")
        ax.artists[1].set_facecolor("green")
        ax.artists[2].set_facecolor("red")
        plt.ylabel("Estimated Mutation Age (generations)")
        plt.title(
            "Average Estimated Mutation Age from TGP \n {} Mutations on Chr 20".format(
                comparable_mutations.shape[0]
            )
        )
        self.save(self.name)


class RecurrentMutations(Figure):
    """
    Figure showing number of recurrent mutations in 1000G tree sequence inferred by
    tsinfer.
    """

    name = "recurrent_mutations"
    data_path = "data"
    filename = [
        "1kg_chr20_ma0.1_ms0.01_p13.recurrent_counts",
        "1kg_chr20_ma0.1_ms0.01_p13.recurrent_counts_nosamples",
        "1kg_chr20_ma0.1_ms0.01_p13.recurrent_counts_nodouble",
        "1kg_chr20_ma0.1_ms0.01_p13.recurrent_counts_nosamples_two_muts",
    ]
    plt_title = "recurrent_mutations_fig"

    def plot(self):
        fig, ax = plt.subplots(
            nrows=3, ncols=1, figsize=(18, 6), sharex=True, sharey=True
        )
        self.data[0] = self.data[0].set_index(self.data[0].columns[0])
        self.data[1] = self.data[1].set_index(self.data[1].columns[0])
        # self.data[2] = self.data[2].set_index(self.data[2].columns[0])
        # two_muts_count = np.unique(self.data[2]["recurrent_counts_two_muts"],
        # return_counts=True)
        ax[0].bar(self.data[0].index[0:4] + 1, self.data[0]["recurrent_counts"][0:4])
        ax[1].bar(
            self.data[1].index[0:4] + 1, self.data[1]["recurrent_counts_nosamples"][0:4]
        )
        ax[2].bar(
            self.data[2].index[0:4] + 1, self.data[2]["recurrent_counts_nodouble"][0:4]
        )
        # ax[2].bar(two_muts_count[0][0:100], two_muts_count[1][0:100])
        ax[0].set_title("Number of Mutations per site")
        ax[1].set_title(
            "Number of Mutations per Site, removing mutations on sample edges"
        )
        ax[2].set_title(
            "Number of Mutations per Site, removing mutations with one or two samples"
        )
        ax[0].set_ylabel("Mutations per site")
        ax[1].set_ylabel("Mutations per site")
        ax[2].set_ylabel("Mutations per site")
        fig.tight_layout()
        self.save(self.name)


class HgdpRecurrentMutations(RecurrentMutations):
    """
    Figure showing number of recurrent mutations in HGDP tree sequence inferred by
    tsinfer.
    """

    name = "hgdp_recurrent_mutations"
    data_path = "data"
    filename = [
        "hgdp_missing_data_chr20_ma0.5_ms0.05_p15.simplify.recurrent_counts",
        "hgdp_missing_data_chr20_ma0.5_ms0.05_p15.simplify.recurrent_counts_nosamples",
        "hgdp_missing_data_chr20_ma0.5_ms0.05_p15.simplify.recurrent_counts_nodouble",
        """hgdp_missing_data_chr20_ma0.5_ms0.05_p15.simplify.
        recurrent_counts_nosamples_two_muts""",
    ]
    plt_title = "hgdp_recurrent_mutations_fig"


class PriorEvaluation(Figure):
    """
    Supplementary Figure 2: Evaluating the Lognormal Prior
    """

    name = "prior_evaluation"
    data_path = "simulated-data"
    filename = "evaluateprior"
    plt_title = "prior_evaluation"

    def __init__(self):
        datafile_name = os.path.join(self.data_path, self.filename + ".csv")
        self.data = pickle.load(open(datafile_name, "rb"))

    def plot(self):
        fig, ax = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
        axes = ax.ravel()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1.8, 1050)
        plt.ylim(1e-3, 4e5)
        all_results = self.data

        for index, ((_, result), mixtures) in enumerate(
                zip(all_results.items(), [False, False, False, False])):
            num_tips_all = np.concatenate(result['num_tips']).ravel()
            num_tips_all_int = num_tips_all.astype(int)
            only_mixtures = np.full(len(num_tips_all), True)
            if mixtures:
                only_mixtures = np.where((num_tips_all - num_tips_all_int) != 0)[0]

            upper_bound_all = np.concatenate(
                    result['upper_bound']).ravel()[only_mixtures]
            lower_bound_all = np.concatenate(
                    result['lower_bound']).ravel()[only_mixtures]
            expectations_all = np.concatenate(
                    result['expectations']).ravel()[only_mixtures]

            real_ages_all = np.concatenate(result['real_ages']).ravel()[only_mixtures]
            num_tips_all = num_tips_all[only_mixtures]
            yerr = [expectations_all - lower_bound_all,
                    upper_bound_all - expectations_all]

            axes[index].errorbar(num_tips_all, expectations_all, ls='none', yerr=yerr,
                                 elinewidth=.3, alpha=0.4, color='grey', zorder=1,
                                 label="95% credible interval of the prior")
            axes[index].scatter(num_tips_all, real_ages_all, s=1, alpha=0.5,
                                zorder=2, color='blue', label="True Time")
            axes[index].scatter(num_tips_all, expectations_all, s=1, color='red',
                                zorder=3, label="expected time", alpha=0.5)
            coverage = (np.sum(
                np.logical_and(real_ages_all < upper_bound_all,
                               real_ages_all > lower_bound_all)) / len(expectations_all))
            axes[index].text(0.35, 0.25, "Overall Coverage Probability:" +
                             "{0:.3f}".format(coverage),
                             size=10, ha='center', va='center',
                             transform=axes[index].transAxes)
            less5_tips = np.where(num_tips_all < 5)[0]
            coverage = np.sum(np.logical_and(
                real_ages_all[less5_tips] < upper_bound_all[less5_tips],
                (real_ages_all[less5_tips] > lower_bound_all[less5_tips])) / len(
                expectations_all[less5_tips]))
            axes[index].text(0.35, 0.21,
                             "<10 Tips Coverage Probability:" + "{0:.3f}".format(
                                 coverage),
                             size=10, ha='center', va='center',
                             transform=axes[index].transAxes)
            mrcas = np.where(num_tips_all == 1000)[0]
            coverage = np.sum(np.logical_and(
                real_ages_all[mrcas] < upper_bound_all[mrcas],
                (real_ages_all[mrcas] > lower_bound_all[mrcas])) /
                len(expectations_all[mrcas]))
            axes[index].text(0.35, 0.17,
                             "MRCA Coverage Probability:" + "{0:.3f}".format(coverage),
                             size=10, ha='center', va='center',
                             transform=axes[index].transAxes)
            axins = zoomed_inset_axes(axes[index], 2.7, loc=4,
                                      bbox_to_anchor=(0.95, 0.1),
                                      bbox_transform=axes[index].transAxes)
            axins.errorbar(num_tips_all, expectations_all, ls='none', yerr=yerr,
                           elinewidth=0.7, alpha=0.1, color='grey',
                           solid_capstyle='projecting', capsize=4,
                           label="95% credible interval of the prior", zorder=1)
            axins.scatter(num_tips_all, real_ages_all, s=2, color='blue', alpha=0.5,
                          label="True Time", zorder=2)
            axins.scatter(num_tips_all, expectations_all, s=2, color='red',
                          label="Expected time", alpha=0.5, zorder=3)
            x1, x2, y1, y2 = 970, 1030, 5e3, 3e5
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xscale('log')
            axins.set_yscale('log')
            axins.set_xticks([], minor=True)
            axins.set_yticks([], minor=True)
            mark_inset(axes[index], axins, loc1=2, loc2=1, fc="none", ec="0.5")
        lgnd = axes[3].legend(loc=4, prop={'size': 12}, bbox_to_anchor=(1, -0.3))
        lgnd.legendHandles[0]._sizes = [30]
        lgnd.legendHandles[1]._sizes = [30]
        lgnd.legendHandles[2]._linewidths = [2]
        fig.text(0.5, 0.04, 'Number of Tips', ha='center', size=15)
        fig.text(0.04, 0.5, 'Node Age (Generations)',
                 va='center', rotation='vertical',
                 size=15)
        axes[0].set_title("p=0", size=14)
        axes[1].set_title("p=1e-8", size=14)
        axes[1].text(1.03, 0.2, "Lognormal Distribution", rotation=90,
                     color='Black', transform=axes[1].transAxes, size=14)
        axes[3].text(1.03, 0.2, "Gamma Distribution", rotation=90,
                     color='Black', transform=axes[3].transAxes, size=14)

        self.save(self.name)


class TsdateAccuracy(Figure):
    """
    Supplementary Figure 4: Evaluating tsdate's accuracy at various mutation rates
    """

    name = "tsdate_accuracy"
    data_path = "simulated-data"
    filename = "tsdate_accuracy"
    plt_title = "tsdate_accuracy"

    def __init__(self):
        datafile_name = os.path.join(self.data_path, self.filename + ".csv")
        self.data = pickle.load(open(datafile_name, "rb"))

    def plot(self):
        (sim, io, maxed, inf_io, inf_maxed, io_kc, max_kc,
         inf_io_kc, inf_maxed_kc) = self.data
        f, axes = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True,
                               gridspec_kw={"wspace": 0.1, "hspace": 0.1,
                                            "height_ratios": [1, 1, 1],
                                            "width_ratios": [1, 1, 1, 1]},
                               figsize=(20, 15))
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_xlim(2e-1, 2e5)
        axes[0, 0].set_ylim(2e-1, 2e5)
        x0, x1 = axes[0, 0].get_xlim()
        y0, y1 = axes[0, 0].get_ylim()

        parameter_arr = [1e-9, 1e-8, 1e-7]

        for index, param in enumerate(parameter_arr):
            true_ages = sim[index]["Simulated Age"].values
            inside_outside = io[index]["IO Age"].values
            maximized = maxed[index]["Max Age"].values
            inferred_inside_outside = inf_io[index]["IO Age"].values
            inferred_maximized = inf_maxed[index]["Max Age"].values

            for i, (method, kc) in enumerate(zip([
                inside_outside, maximized, inferred_inside_outside, inferred_maximized],
                    [io_kc, max_kc, inf_io_kc, inf_maxed_kc])):
                self.mutation_accuracy(axes[index, i], true_ages, method, "",
                                       kc_distance_1=kc[index])
            axes[index, 3].text(3.25, 0.15, "Mutation Rate: " + str(param), rotation=90,
                                transform=axes[index, 1].transAxes, size=20)

        axes[0, 0].set_title("Inside-Outside", size=20, color="Black")
        axes[0, 1].set_title("Maximization", size=20, color="Black")
        axes[0, 2].set_title("Inside-Outside", size=20, color="Black")
        axes[0, 3].set_title("Maximization", size=20, color="Black")

        f.text(0.5, 0.05, 'True Time', ha='center', size=25)
        f.text(0.08, 0.5, 'Estimated Time', va='center',
               rotation='vertical', size=25)
        f.text(0.31, 0.92, 'tsdate using True Topologies', ha='center', size=25)
        f.text(0.71, 0.92, 'tsdate using tsinfer Topologies', ha='center', size=25)

        self.save(self.name)


class NeutralSimulatedMutationAccuracy(Figure):
    """
    Supplementary Figure 5: Accuracy of tsdate, tsdate + tsinfer, Geva and Relate
    on a neutral coalescent simulation.
    """

    name = "neutral_simulated_mutation_accuracy"
    data_path = "simulated-data"
    filename = ["neutral_simulated_mutation_accuracy_mutations", "neutral_simulated_mutation_accuracy_kc_distances"]

    def plot(self):
        df = self.data[0]
        kc_distances = self.data[1]
        kc_distances = kc_distances.set_index(kc_distances.columns[0])
        #error_df = self.data[1]
        #anc_error_df = self.data[2]
        f, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.1, "hspace": 0.1}, figsize=(12, 12))
        
        ax[0, 0].set_xscale("log")
        ax[0, 0].set_yscale("log")
        ax[0, 0].set_xlim(1, 2e5)
        ax[0, 0].set_ylim(1, 2e5)

        #for row, (df, kc_distance) in enumerate(zip(no_error_df, kc_distances)):
        # We can only plot comparable mutations, so remove all rows where NaNs exist
        #df = df.drop(columns=["geva"])
        #df = df.replace([0, -np.inf], np.nan)
        #df = df.dropna()
        df = df[df["simulated_ts"] > 0]
        df = df[df["relate"] > 0]

        # tsdate on true tree
        self.mutation_accuracy(
            ax[0, 0],
            df["simulated_ts"],
            df["tsdate"],
            "tsdate (using true topology)",
            kc_distance_1=np.mean(kc_distances.loc[1]["tsdate"])
        )

        # tsdate on inferred tree
        self.mutation_accuracy(
            ax[0, 1], df["simulated_ts"], df["tsdate_inferred"], "tsinfer + tsdate",
            kc_distance_0=np.mean(kc_distances.loc[0]["tsdate_inferred"]),
            kc_distance_1=np.mean(kc_distances.loc[1]["tsdate_inferred"])
        )

        # Relate accuracy
        self.mutation_accuracy(ax[1, 1], df["simulated_ts"][~np.isnan(df["relate"])],
                df["relate"][~np.isnan(df["relate"])], "Relate", cmap="Greens",
                kc_distance_0=np.mean(kc_distances.loc[0]["relate"]),
                kc_distance_1=np.mean(kc_distances.loc[1]["relate"]))

        # GEVA accuracy
        self.mutation_accuracy(ax[1, 0], df["simulated_ts"][~np.isnan(df["geva"])],
                df["geva"][~np.isnan(df["geva"])], "GEVA", cmap="Reds")


        f.text(0.5, 0.05, 'True Time', ha='center', size=25)
        f.text(0.05, 0.5, 'Estimated Time', va='center',
               rotation='vertical', size=25)

        self.save(self.name)


class TsdateChr20Accuracy(Figure):
    """
    Supplementary Figure 6: Evaluating tsdate's accuracy on Simulated Chromosome 20
    """

    name = "tsdate_accuracy_chr20"
    data_path = "simulated-data"
    filename = ["tsdate_chr20_accuracy_mutations",
                "tsdate_chr20_accuracy_error_mutations",
                "tsdate_chr20_accuracy_anc_error_mutations",
                "tsdate_chr20_accuracy_kc_distances",
                "tsdate_chr20_accuracy_error_kc_distances",
                "tsdate_chr20_accuracy_anc_error_kc_distances"]

    plt_title = "tsdate_accuracy_chr20"


    def plot(self):
        df = self.data[0]
        error_df = self.data[1]
        anc_error_df = self.data[2]
        kc_distances = self.data[3]
        kc_distances = kc_distances.set_index(kc_distances.columns[0])
        error_kc_distances = self.data[4]
        error_kc_distances = error_kc_distances.set_index(error_kc_distances.columns[0])
        anc_error_kc_distances = self.data[5]
        anc_error_kc_distances = anc_error_kc_distances.set_index(
                anc_error_kc_distances.columns[0])

        f, axes = plt.subplots(ncols=3, nrows=5, sharex=True, sharey=True,
             gridspec_kw={"wspace": 0.1, "hspace": 0.1, "width_ratios": [1, 1, 1],
                          "height_ratios": [1, 0.1, 1, 1, 1]}, figsize=(15, 20))
        axes[0, 0].axis('off')
        axes[0, 2].axis('off')
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        axes[1, 2].axis('off')

        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_xlim(1, 2e5)
        axes[0, 0].set_ylim(1, 2e5)
        x0, x1 = axes[0, 0].get_xlim()
        y0, y1 = axes[0, 0].get_ylim()
        row_labels = ["tsdate", "", "tsinfer + tsdate", "tsinfer with mismatch \n+ tsdate",
                      "iteration tsinfer \n+ tsdate"]
        for (i, name), j in zip(enumerate(row_labels), [1, 2, 2, 2, 2]):
            axes[i, j].set_ylabel(name, rotation=90, size=20)
            axes[i, j].yaxis.set_label_position("right")

        sim = df["simulated_ts"]
        methods = ["inferred_dated", "mismatch_inferred_dated", "iter_dated_ts"]
        comparable_sites = np.logical_and(sim > 0, df["dated"] > 0)
        self.mutation_accuracy(axes[0, 1], sim[comparable_sites],
                               df["dated"][comparable_sites], "",
                               kc_distance_1=np.mean(kc_distances.loc[1]["dated"]))
        for col, prefix, (mut_df, kc_df) in zip(
                range(3), ["", "error_", "anc_error_"], [(df, kc_distances),
                    (error_df, error_kc_distances),
                    (anc_error_df, anc_error_kc_distances)]):
            for row, method, cmap in zip([2, 3, 4], methods, ["Blues", "Blues", "Blues"]):
                method = prefix + method
                result = mut_df[method]
                comparable_sites = np.logical_and(sim > 0, result > 0)
                cur_true_ages = sim[comparable_sites]
                cur_results = result[comparable_sites]
                self.mutation_accuracy(
                        axes[row, col], cur_true_ages,
                            cur_results, "", cmap=cmap,
                            kc_distance_0=np.mean(kc_df.loc[0][method]),
                                kc_distance_1=np.mean(kc_df.loc[1][method]))
        axes[0, 1].set_title("tsdate using Simulated Topology")
        axes[2, 0].set_title("No Error")
        axes[2, 1].set_title("Empirical Error")
        axes[2, 2].set_title("Empirical Error + 1% Ancestral State Error")
        f.text(0.5, 0.05, 'True Time', ha='center', size=25)
        f.text(0.08, 0.4, 'Estimated Time', va='center',
               rotation='vertical', size=25)

        self.save(self.name)

#        df = self.data[0]
#        error_df = self.data[1]
#        anc_error_df = self.data[2]
#        kc_distances = self.data[3]
#        kc_distances = kc_distances.set_index(kc_distances.columns[0])
#        error_kc_distances = self.data[4]
#        error_kc_distances = error_kc_distances.set_index(error_kc_distances.columns[0])
#        anc_error_kc_distances = self.data[5]
#        anc_error_kc_distances = anc_error_kc_distances.set_index(
#                anc_error_kc_distances.columns[0])
#        sim = self.data[0]["simulated_ts"]
#        dated = self.data[0]["tsdate"]
#
#        f, axes = plt.subplots(ncols=3, nrows=5, sharex=True, sharey=True,
#             gridspec_kw={"wspace": 0.1, "hspace": 0.1, "width_ratios": [1, 1, 1],
#                          "height_ratios": [1, 0.1, 1, 1, 1]}, figsize=(15, 20))
#        axes[0, 0].axis('off')
#        axes[0, 2].axis('off')
#        axes[1, 0].axis('off')
#        axes[1, 1].axis('off')
#        axes[1, 2].axis('off')
#
#        axes[0, 0].set_xscale('log')
#        axes[0, 0].set_yscale('log')
#        axes[0, 0].set_xlim(1, 2e5)
#        axes[0, 0].set_ylim(1, 2e5)
#        x0, x1 = axes[0, 0].get_xlim()
#        y0, y1 = axes[0, 0].get_ylim()
#        row_labels = ["tsdate", "", "tsinfer + tsdate", "mismatch tsinfer + tsdate",
#                      "Iteration"]
#        for (i, name), j in zip(enumerate(row_labels), [1, 2, 2, 2, 2]):
#            axes[i, j].set_ylabel(name, rotation=90,
#                                  size=20)
#            axes[i, j].yaxis.set_label_position("right")
#
#        ax_counter = 0
#        for i, (column_idx_list, mut_ages_list) in zip([0, 2, 3, 4], [([1], [dated]), ([0, 1, 2], inferred_mut_ages),
#                                                   ([0, 1, 2], mismatch_inferred_mut_ages),
#                                                   ([0, 1, 2], iter_inferred_mut_ages)]):
#            for j, result in zip(column_idx_list, mut_ages_list):
#                comparable_sites = np.logical_and(sim > 0, result > 0)
#                cur_true_ages = sim[comparable_sites]
#                cur_results = result[comparable_sites]
#                self.mutation_accuracy(
#                        axes[i, j], cur_true_ages, cur_results, "",
#                            kc_distance_0=list(kc_distances[0].values())[ax_counter],
#                            kc_distance_1=list(kc_distances[1].values())[ax_counter])
#                ax_counter += 1
#        axes[0, 1].set_title("tsdate using Simulated Topology", size=20)
#        axes[2, 0].set_title("No Error", size=20)
#        axes[2, 1].set_title("Empirical Error", size=20)
#        axes[2, 2].set_title("Empirical Error + \n1% Ancestral State Error", size=20)
#        f.text(0.5, 0.05, 'True Time', ha='center', size=25)
#        f.text(0.08, 0.4, 'Estimated Time', va='center',
#               rotation='vertical', size=25)
#
#        self.save(self.name)


class Chr20SimulatedMutationAccuracy(Figure):
    """
    Supplementary Figure 7: Evaluating tsdate, Relate, and GEVA accuracy on Simulated
    Chromosome 20 snippets
    """

    name = "simulated_accuracy_chr20"
    data_path = "simulated-data"
    filename = ["chr20_simulated_mutation_accuracy_mutations",
                "chr20_simulated_mutation_accuracy_error_mutations",
                "chr20_simulated_mutation_accuracy_anc_error_mutations",
                "chr20_simulated_mutation_accuracy_kc_distances",
                "chr20_simulated_mutation_accuracy_error_kc_distances",
                "chr20_simulated_mutation_accuracy_anc_error_kc_distances"]
    plt_title = "simulated_accuracy_chr20"

    def plot(self):
        df = self.data[0]
        error_df = self.data[1]
        anc_error_df = self.data[2]
        kc_distances = self.data[3]
        kc_distances = kc_distances.set_index(kc_distances.columns[0])
        error_kc_distances = self.data[4]
        error_kc_distances = error_kc_distances.set_index(error_kc_distances.columns[0])
        anc_error_kc_distances = self.data[5]
        anc_error_kc_distances = anc_error_kc_distances.set_index(
                anc_error_kc_distances.columns[0])

        f, axes = plt.subplots(ncols=3, nrows=5, sharex=True, sharey=True,
             gridspec_kw={"wspace": 0.1, "hspace": 0.1, "width_ratios": [1, 1, 1],
                          "height_ratios": [1, 0.1, 1, 1, 1]}, figsize=(15, 20))
        axes[0, 0].axis('off')
        axes[0, 2].axis('off')
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        axes[1, 2].axis('off')

        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_xlim(1, 2e5)
        axes[0, 0].set_ylim(1, 2e5)
        x0, x1 = axes[0, 0].get_xlim()
        y0, y1 = axes[0, 0].get_ylim()
        row_labels = ["tsdate", "", "tsinfer + tsdate", "Relate", "GEVA"]
        for (i, name), j, color in zip(enumerate(row_labels), [1, 2, 2, 2, 2], ["Blue", "", "Blue", "Red", "Green"]):
            axes[i, j].set_ylabel(name, rotation=90,
                                color=color, size=20)
            axes[i, j].yaxis.set_label_position("right")

        sim = df["simulated_ts"]
        methods = ["tsdate_inferred", "relate", "geva"]
        comparable_sites = np.logical_and(sim > 0, df["tsdate"] > 0)
        self.mutation_accuracy(axes[0, 1], sim[comparable_sites],
                               df["tsdate"][comparable_sites], "",
                               kc_distance_1=np.mean(kc_distances.loc[1]["tsdate"]))
        for col, (mut_df, kc_df) in zip(
                range(3), [(df, kc_distances),
                    (error_df, error_kc_distances),
                    (anc_error_df, anc_error_kc_distances)]):
            for row, method, cmap in zip([2, 3, 4], methods, ["Blues", "Greens", "Reds"]):
                result = mut_df[method]
                comparable_sites = np.logical_and(sim > 0, result > 0)
                cur_true_ages = sim[comparable_sites]
                cur_results = result[comparable_sites]
                self.mutation_accuracy(
                        axes[row, col], cur_true_ages,
                            cur_results, "", cmap=cmap,
                            kc_distance_0=np.mean(kc_df.loc[0][method]),
                                kc_distance_1=np.mean(kc_df.loc[1][method]))
        axes[0, 1].set_title("tsdate using Simulated Topology")
        axes[2, 0].set_title("No Error")
        axes[2, 1].set_title("Empirical Error")
        axes[2, 2].set_title("Empirical Error + 1% Ancestral State Error")
        f.text(0.5, 0.05, 'True Time', ha='center', size=25)
        f.text(0.08, 0.4, 'Estimated Time', va='center',
               rotation='vertical', size=25)

        self.save(self.name)


class TsdateIterationAccuracy(NeutralSimulatedMutationAccuracy):
    """
    Plot figure showing accuracy of tsdate iteration
    """

    name = "tsdate_iter_neutral_simulated_mutation_accuracy"
    data_path = "simulated-data"
    filename = ["tsdate_iteration_neutral_simulated_mutation_accuracy_mutations"]

    def plot(self):
        df = self.data[0]
        with sns.axes_style("white"):
            fig, ax = plt.subplots(
                nrows=3, ncols=2, figsize=(12, 12), sharex=True, sharey=True
            )
            # We can only plot comparable mutations, so remove all rows where NaNs exist
            df = df.dropna()

            ax[0, 0].set_xscale("log")
            ax[0, 0].set_yscale("log")
            ax[0, 0].set_xlim(1, 2e5)
            ax[0, 0].set_ylim(1, 2e5)

            # tsdate on true tree
            self.mutation_accuracy(
                ax[0, 0],
                df["simulated_ts"],
                df["tsdate"],
                "tsdate (using true topology)",
            )
            # tsdate on true tree
            self.mutation_accuracy(
                ax[0, 1],
                df["simulated_ts"],
                df["tsdate_1stiter"],
                "tsdate (using true topology) 1 iteration",
            )
            # tsdate on true tree
            self.mutation_accuracy(
                ax[1, 0],
                df["simulated_ts"],
                df["tsdate_2nditer"],
                "tsdate (using true topology) 2 iteration",
            )

            # tsdate on inferred tree
            self.mutation_accuracy(
                ax[1, 1], df["simulated_ts"], df["tsdate_inferred"], "tsinfer + tsdate"
            )
            # tsdate on inferred tree
            self.mutation_accuracy(
                ax[2, 0],
                df["simulated_ts"],
                df["tsdate_inferred_1stiter"],
                "tsinfer + tsdate",
            )
            # tsdate on inferred tree
            self.mutation_accuracy(
                ax[2, 1],
                df["simulated_ts"],
                df["tsdate_inferred_2nditer"],
                "tsinfer + tsdate",
            )

            self.save(self.name)


class OoaChr20SimulatedMutationAccuracy(NeutralSimulatedMutationAccuracy):
    """
    """

    name = "ooa_chr20_simulated_mutation_accuracy"
    data_path = "simulated-data"
    filename = ["chr20_simulated_mutation_accuracy_mutations",
                "chr20_simulated_mutation_accuracy_kc_distances",
                "chr20_simulated_mutation_accuracy_error_mutations",
                "chr20_simulated_mutation_accuracy_error_kc_distances",
                "chr20_simulated_mutation_accuracy_anc_error_mutations",
                "chr20_simulated_mutation_accuracy_anc_error_kc_distances"]

    def plot(self):
        df = self.data[0]
        kc_distances = self.data[1]
        kc_distances = kc_distances.set_index(kc_distances.columns[0])
        error_df = self.data[2]
        error_kc = self.data[3]
        error_kc = error_kc.set_index(error_kc.columns[0])
        anc_error_df = self.data[4]
        anc_error_kc = self.data[5]
        anc_error_kc = anc_error_kc.set_index(anc_error_kc.columns[0])

        f, axes = plt.subplots(ncols=3, nrows=5, sharex=True, sharey=True,
                               gridspec_kw={"wspace": 0.1, "hspace": 0.1,
                                   "width_ratios": [1, 1, 1],
                                   "height_ratios": [1, 0.1, 1, 1, 1]}, figsize=(15, 20))
        axes[0, 0].axis('off')
        axes[0, 2].axis('off')
        for i in range(0, 3):
            axes[1, i].axis("off")

        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_xlim(1, 2e5)
        axes[0, 0].set_ylim(1, 2e5)
        x0, x1 = axes[0, 0].get_xlim()
        y0, y1 = axes[0, 0].get_ylim()
        axes[0, 1].set_title("tsdate using Simulated Topology")
        axes[2, 0].set_title("No Error")
        axes[2, 1].set_title("Empirical Error")
        axes[2, 2].set_title("Empirical Error + 1% Ancestral State Error")
        row_labels = ["tsdate", "", "tsinfer + tsdate", "Relate", "GEVA"]
        for (i, name), j, color in zip(enumerate(row_labels), [1, 2, 2, 2, 2], ["Blue", "", "Blue", "Green", "Red"]):
            axes[i, j].set_ylabel(name, rotation=90,
                                color=color, size=20)
            axes[i, j].yaxis.set_label_position("right")

        self.mutation_accuracy(
            axes[0, 1],
            df["simulated_ts"][df["tsdate"] > 0],
            df["tsdate"][df["tsdate"] > 0],
            "",
            kc_distance_1=np.mean(kc_distances.loc[1]["tsdate"])
        )
        for col, (df, kc) in enumerate(zip([df, error_df, anc_error_df],
                                           [kc_distances, error_kc, anc_error_kc])):
            # tsdate on inferred tree
            tsdate_inferred_viable = np.logical_and(df["tsdate_inferred"] > 0,
                                                    df["simulated_ts"] > 0)
            self.mutation_accuracy(
                axes[2, col], df["simulated_ts"][tsdate_inferred_viable],
                df["tsdate_inferred"][tsdate_inferred_viable], "",
                kc_distance_0=np.mean(kc_distances.loc[0]["tsdate_inferred"]),
                kc_distance_1=np.mean(kc_distances.loc[1]["tsdate_inferred"])
            )

            # Relate accuracy
            relate_ages_viable = np.logical_and(
                    df["simulated_ts"] > 0,
                        np.logical_and(~np.isnan(df["relate_reage"]),
                        df["relate_reage"] > 0))
            self.mutation_accuracy(axes[3, col], df["simulated_ts"][relate_ages_viable],
                    df["relate_reage"][relate_ages_viable], "",
                        kc_distance_0=np.mean(kc_distances.loc[0]["relate"]),
                        kc_distance_1=np.mean(kc_distances.loc[1]["relate"]))

            # GEVA accuracy
            self.mutation_accuracy(axes[4, col],
                                   df["simulated_ts"][~np.isnan(df["geva"])],
                                   df["geva"][~np.isnan(df["geva"])], "")

        f.text(0.5, 0.05, 'True Time', ha='center', size=25)
        f.text(0.08, 0.5, 'Estimated Time', va='center',
               rotation='vertical', size=25)

        self.save(self.name)

class TmrcaClustermap(Figure):
    """
    """

    name = "tmrca_clustermap"
    data_path = "all-data"
    filename = ["merged_hgdp_1kg_sgdp_high_cov_ancients_chr20.dated.binned.historic.20nodes.tmrcas"]
    sgdp_region_map = {
                    "Abkhasian": "West Eurasia",
                    "Adygei": "West Eurasia",
                    "Albanian": "West Eurasia",
                    "Aleut": "Central Asia Siberia",
                    "Altaian": "Central Asia Siberia",
                    "Ami": "East Asia",
                    "Armenian": "West Eurasia",
                    "Atayal": "East Asia",
                    "Australian": "Oceania",
                    "Balochi": "South Asia",
                    "BantuHerero": "Africa",
                    "BantuKenya": "Africa",
                    "BantuTswana": "Africa",
                    "Basque": "West Eurasia",
                    "BedouinB": "West Eurasia",
                    "Bengali": "South Asia",
                    "Bergamo": "West Eurasia",
                    "Biaka": "Africa",
                    "Bougainville": "Oceania",
                    "Brahmin": "South Asia",
                    "Brahui": "South Asia",
                    "Bulgarian": "West Eurasia",
                    "Burmese": "East Asia",
                    "Burusho": "South Asia",
                    "Cambodian": "East Asia",
                    "Chane": "Americas",
                    "Chechen": "West Eurasia",
                    "Chipewyan": "Americas",
                    "Chukchi": "Central Asia Siberia",
                    "Cree": "Americas",
                    "Crete": "West Eurasia",
                    "Czech": "West Eurasia",
                    "Dai": "East Asia",
                    "Daur": "East Asia",
                    "Dinka": "Africa",
                    "Druze": "West Eurasia",
                    "Dusun": "Oceania",
                    "English": "West Eurasia",
                    "Esan": "Africa",
                    "Eskimo_Chaplin": "Central Asia Siberia",
                    "Eskimo_Naukan": "Central Asia Siberia",
                    "Eskimo_Sireniki": "Central Asia Siberia",
                    "Estonian": "West Eurasia",
                    "Even": "Central Asia Siberia",
                    "Finnish": "West Eurasia",
                    "French": "West Eurasia",
                    "Gambian": "Africa",
                    "Georgian": "West Eurasia",
                    "Greek": "West Eurasia",
                    "Han": "East Asia",
                    "Hawaiian": "Oceania",
                    "Hazara": "South Asia",
                    "Hezhen": "East Asia",
                    "Hungarian": "West Eurasia",
                    "Icelandic": "West Eurasia",
                    "Igbo": "Africa",
                    "Igorot": "Oceania",
                    "Iranian": "West Eurasia",
                    "Iraqi_Jew": "West Eurasia",
                    "Irula": "South Asia",
                    "Itelman": "Central Asia Siberia",
                    "Japanese": "East Asia",
                    "Jordanian": "West Eurasia",
                    "Ju_hoan_North": "Africa",
                    "Kalash": "South Asia",
                    "Kapu": "South Asia",
                    "Karitiana": "Americas",
                    "Kashmiri_Pandit": "South Asia",
                    "Kharia": "South Asia",
                    "Khomani_San": "Africa",
                    "Khonda_Dora": "South Asia",
                    "Kinh": "East Asia",
                    "Kongo": "Africa",
                    "Korean": "East Asia",
                    "Kurumba": "South Asia",
                    "Kusunda": "South Asia",
                    "Kyrgyz": "Central Asia Siberia",
                    "Lahu": "East Asia",
                    "Lemande": "Africa",
                    "Lezgin": "West Eurasia",
                    "Luhya": "Africa",
                    "Luo": "Africa",
                    "Madiga": "South Asia",
                    "Makrani": "South Asia",
                    "Mala": "South Asia",
                    "Mandenka": "Africa",
                    "Mansi": "Central Asia Siberia",
                    "Maori": "Oceania",
                    "Masai": "Africa",
                    "Mayan": "Americas",
                    "Mbuti": "Africa",
                    "Mende": "Africa",
                    "Miao": "East Asia",
                    "Mixe": "Americas",
                    "Mixtec": "Americas",
                    "Mongola": "Central Asia Siberia",
                    "Mozabite": "Africa",
                    "Nahua": "Americas",
                    "Naxi": "East Asia",
                    "North_Ossetian": "West Eurasia",
                    "Norwegian": "West Eurasia",
                    "Onge": "South Asia",
                    "Orcadian": "West Eurasia",
                    "Oroqen": "East Asia",
                    "Palestinian": "West Eurasia",
                    "Papuan": "Oceania",
                    "Pathan": "South Asia",
                    "Piapoco": "Americas",
                    "Pima": "Americas",
                    "Polish": "West Eurasia",
                    "Punjabi": "South Asia",
                    "Quechua": "Americas",
                    "Relli": "South Asia",
                    "Russian": "West Eurasia",
                    "Saami": "West Eurasia",
                    "Saharawi": "Africa",
                    "Samaritan": "West Eurasia",
                    "Sardinian": "West Eurasia",
                    "She": "East Asia",
                    "Sherpa": "South Asia",
                    "Sindhi": "South Asia",
                    "Somali": "Africa",
                    "Spanish": "West Eurasia",
                    "Surui": "Americas",
                    "Tajik": "West Eurasia",
                    "Thai": "East Asia",
                    "Tibetan": "South Asia",
                    "Tlingit": "Central Asia Siberia",
                    "Tubalar": "Central Asia Siberia",
                    "Tu": "East Asia",
                    "Tujia": "East Asia",
                    "Turkish": "West Eurasia",
                    "Tuscan": "West Eurasia",
                    "Ulchi": "Central Asia Siberia",
                    "Uygur": "East Asia",
                    "Xibo": "East Asia",
                    "Yadava": "South Asia",
                    "Yakut": "Central Asia Siberia",
                    "Yemenite_Jew": "West Eurasia",
                    "Yi": "East Asia",
                    "Yoruba": "Africa",
                    "Zapotec": "Americas",
                }
    hgdp_region_map = {
                    "Brahui": "Central South Asia",
                    "Balochi": "Central South Asia",
                    "Hazara": "Central South Asia",
                    "Makrani": "Central South Asia",
                    "Sindhi": "Central South Asia",
                    "Pathan": "Central South Asia",
                    "Kalash": "Central South Asia",
                    "Burusho": "Central South Asia",
                    "Mbuti": "Africa",
                    "Biaka": "Africa",
                    "Bougainville": "Oceania",
                    "French": "Europe",
                    "PapuanSepik": "Oceania",
                    "PapuanHighlands": "Oceania",
                    "Druze": "Middle East",
                    "Bedouin": "Middle East",
                    "Sardinian": "Europe",
                    "Palestinian": "Middle East",
                    "Colombian": "Americas",
                    "Cambodian": "East Asia",
                    "Japanese": "East Asia",
                    "Han": "East Asia",
                    "Orcadian": "Europe",
                    "Surui": "Americas",
                    "Maya": "Americas",
                    "Russian": "Europe",
                    "Mandenka": "Africa",
                    "Yoruba": "Africa",
                    "Yakut": "East Asia",
                    "San": "Africa",
                    "BantuSouthAfrica": "Africa",
                    "Karitiana": "Americas",
                    "Pima": "Americas",
                    "Tujia": "East Asia",
                    "BergamoItalian": "Europe",
                    "Tuscan": "Europe",
                    "Yi": "East Asia",
                    "Miao": "East Asia",
                    "Oroqen": "East Asia",
                    "Daur": "East Asia",
                    "Mongolian": "East Asia",
                    "Hezhen": "East Asia",
                    "Xibo": "East Asia",
                    "Mozabite": "Middle East",
                    "NorthernHan": "East Asia",
                    "Uygur": "Central South Asia",
                    "Dai": "East Asia",
                    "Lahu": "East Asia",
                    "She": "East Asia",
                    "Naxi": "East Asia",
                    "Tu": "East Asia",
                    "Basque": "Europe",
                    "Adygei": "Europe",
                    "BantuKenya": "Africa",
                }


    tgp_region_pop = {
        'Americas': ['CLM', 'MXL', 'PUR', 'PEL'],
        'Africa': ['LWK', 'ASW', 'GWD', 'MSL', 'YRI', 'ACB', 'ESN'],
        'East Asia': ['CHS', 'KHV', 'JPT', 'CHB', 'CDX'],
        'South Asia': ['BEB', 'STU', 'GIH', 'PJL', 'ITU'],
        'Europe': ['FIN', 'GBR', 'IBS', 'CEU', 'TSI']
    }

    def get_tgp_region_colours(self):
        return {
            "EAS": sns.color_palette("Greens", 2)[1],
            "EUR": sns.color_palette("Blues", 1)[0],
            "AFR": sns.color_palette("Wistia", 3)[0],
            "AMR": sns.color_palette("Reds", 2)[1],
            "SAS": sns.color_palette("Purples", 2)[1],
        }

    def get_sgdp_region_colours(self):
        cols = self.get_tgp_region_colours()
        return {
            'Africa': cols["AFR"],
            'America': cols["AMR"],
            'EastAsia': cols["EAS"],
            'SouthAsia': cols["SAS"],
            'Oceania': "brown",
            'WestEurasia': cols["EUR"],
            'CentralAsiaSiberia': "pink"
         }

    def get_hgdp_region_colours(self):
        cols = self.get_tgp_region_colours()
        return {
            'AFRICA': cols["AFR"],
            'AMERICA': cols["AMR"],
            'EAST_ASIA': cols["EAS"],
            'CENTRAL_SOUTH_ASIA': cols["SAS"],
            'OCEANIA': "brown",
            'EUROPE': cols["EUR"],
            'MIDDLE_EAST': "teal"
         }

    def get_tgp_hgdp_sgdp_region_colours(self):
        dict_combined = dict(self.get_hgdp_region_colours(), **self.get_tgp_region_colours(), **self.get_sgdp_region_colours())
        return dict_combined

    def make_symmetric(self, df):
        """
        Make TMRCA dataframe symmetric
        """
        df_arr = df.values
        i_upper = np.tril_indices(df.shape[0], 0)
        df_arr[i_upper] = df_arr.T[i_upper]
        return df 

    def plot(self):
        df = self.data[0]
        df = df.set_index(df.columns[0])
        tmrcas = self.make_symmetric(df)
        tgp_region_map = {}
        for region, pop in self.tgp_region_pop.items():
            for p in pop:
                tgp_region_map[p] = region

        pop_names = tmrcas.columns
        pop_names = [pop.split(".")[0] for pop in pop_names]
        regions = list()
        pop_name_suffixes = list()
        for pop in pop_names[0:54]:
            pop_name_suffixes.append(pop + "_HGDP")
            regions.append(self.hgdp_region_map[pop])
        for pop in pop_names[54:80]:
            pop_name_suffixes.append(pop + "_TGP")
            regions.append(tgp_region_map[pop])
        for pop in pop_names[80:210]:
            pop_name_suffixes.append(pop + "_SGDP")
            regions.append(self.sgdp_region_map[pop])
        for pop in pop_names[210:]:
            pop_name_suffixes.append(pop)
            regions.append("Ancients")
        pop_names = pop_name_suffixes
        tmrcas.columns=pop_names
        tmrcas["region"] = regions
        tgp_origin = {pop: "white" for pop in tmrcas.columns}
        sgdp_origin = {pop: "white" for pop in tmrcas.columns}
        hgdp_origin = {pop: "white" for pop in tmrcas.columns}
        ancient_origin = {pop: "white" for pop in tmrcas.columns}
        for pop in tmrcas.columns:
            if "TGP" in pop:
                tgp_origin[pop] = "black"
            elif "SGDP" in pop:
                sgdp_origin[pop] = "black"
            elif "HGDP" in pop:
                hgdp_origin[pop] = "black"
            else:
                ancient_origin[pop] = "black"

        colours = {}
        region_colours = self.get_tgp_hgdp_sgdp_region_colours()
        region_colours["Ancients"] = "orange"

        new_names = ["Africa", "Americas", "East Asia", "Central South Asia", "Oceania", "Europe", "Middle East", "East Asia",
        "Europe", "Africa", "Americas", "South Asia", "Africa", "Americas", "East Asia", "South Asia", "Oceania", "West Eurasia",
        "Central Asia Siberia", "Ancients"]
        new_region_colours = {}
        for new_name, (key, val) in zip(new_names, region_colours.items()):
            new_region_colours[new_name] = val
        for pop_suffix, region in zip(tmrcas.columns, tmrcas["region"]):
            colours[pop_suffix] = new_region_colours[region]


        tmrcas = tmrcas.drop(columns="region")
        tmrcas.index = tmrcas.columns
        mergedg = tmrcas

        colours = pd.Series(colours)
        colours.name = "Region"
        tgp_origin = pd.Series(tgp_origin)
        tgp_origin.name = "TGP"
        hgdp_origin = pd.Series(hgdp_origin)
        hgdp_origin.name = "HGDP"
        sgdp_origin = pd.Series(sgdp_origin)
        sgdp_origin.name = "SGDP"
        ancient_origin = pd.Series(ancient_origin)
        ancient_origin.name = "Ancient"
        col_colors = pd.concat([tgp_origin, hgdp_origin, sgdp_origin, ancient_origin], axis=1)
        cg = sns.clustermap(mergedg, method="average", xticklabels=True, yticklabels=True, figsize=(30,30), rasterized=True,
                            row_colors=colours, col_colors= col_colors ,cbar_pos=(0.55, 0.77, 0.4, 0.07),cmap=plt.cm.inferno_r,
                            cbar_kws = dict(orientation="horizontal",
                                            label="Average TMRCA (generations)"))
        cg.cax.tick_params(labelsize=20)
        cg.cax.set_xlabel("Average TMRCA (generations)", size=20)

        cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xmajorticklabels(), fontsize=7)
        cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_xmajorticklabels(), fontsize=7)

        for region, col in new_region_colours.items():
            cg.ax_col_dendrogram.bar(0, 0, color=col, label=region, linewidth=0)

        cg.ax_col_dendrogram.set_xlim([0,0])
        pos = cg.ax_col_colors.get_position()
        cg.ax_col_colors.set_position([pos.bounds[0], pos.bounds[1], pos.bounds[2], pos.bounds[3]/4])

        handles, labels = cg.ax_col_dendrogram.get_legend_handles_labels()

        cg.ax_col_dendrogram.legend(handles, labels, loc="upper left", ncol=3, fontsize=20, frameon=True,
                                   bbox_to_anchor=(-0.1, 0.2), title="Region", title_fontsize=25)

        self.save(self.name)

######################################
#
# Helper functions
#
######################################


def get_subclasses(cls):
    for subclass in cls.__subclasses__():
        yield from get_subclasses(subclass)
        yield subclass


def latex_float(f):
    """
    Return an exponential number in nice LaTeX form.
    In titles etc for plots this needs to be encased in $ $ signs, and r'' strings used
    """
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


######################################
#
# Main
#
######################################


def main():
    figures = list(get_subclasses(Figure))

    name_map = {fig.name: fig for fig in figures if fig.name is not None}

    parser = argparse.ArgumentParser(description="Make the plots for specific figures.")
    parser.add_argument(
        "name",
        type=str,
        help="figure name",
        choices=sorted(list(name_map.keys()) + ["all"]),
    )

    args = parser.parse_args()
    if args.name == "all":
        for _, fig in name_map.items():
            if fig in figures:
                fig().plot()
    else:
        fig = name_map[args.name]()
        fig.plot()


if __name__ == "__main__":
    main()
