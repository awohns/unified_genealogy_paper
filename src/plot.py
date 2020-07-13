#!/usr/bin/env python3
"""
Generates all the actual figures. Run like
 python3 src/plot.py PLOT_NAME
"""
import argparse
import os
import scipy
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_log_error
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mplc

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

    def mutation_accuracy(self, ax, x, y, label):
        ax.scatter(x, y, s=0.05, alpha=0.05, cmap=plt.cm.plasma)
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        ax.set_title(label, fontsize=24)
        ax.text(
            0.36,
            0.16,
            "RMSLE:" + "{0:.2f}".format(np.sqrt(mean_squared_log_error(x, y))),
            fontsize=19,
            transform=ax.transAxes,
        )
        ax.text(
            0.36,
            0.1,
            "Spearman's "
            + "$\\rho$: "
            + "{0:.2f}".format(scipy.stats.spearmanr(x, y)[0]),
            fontsize=19,
            transform=ax.transAxes,
        )
        ax.text(
            0.36,
            0.04,
            "Pearson's r: "
            + "{0:.2f}".format(scipy.stats.pearsonr(np.log(x), np.log(y))[0]),
            fontsize=19,
            transform=ax.transAxes,
        )


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
        with sns.axes_style("white"):
            fig, ax = plt.subplots(
                nrows=1, ncols=2, figsize=(12, 6), sharex=True, sharey=True
            )
            true_vals = df["simulated_ts"]
            tsdate = df["tsdate"]
            tsdate_inferred = df["tsdate_inferred"]

            ax[0].set_xscale("log")
            ax[0].set_yscale("log")
            ax[0].set_xlim(1, 2e5)
            ax[0].set_ylim(1, 2e5)

            # tsdate on true tree
            self.mutation_accuracy(
                ax[0], true_vals, tsdate, "tsdate (using true topology)"
            )

            # tsdate on inferred tree
            self.mutation_accuracy(
                ax[1], true_vals, tsdate_inferred, "tsinfer + tsdate"
            )

            self.save("tsdate_simulated_accuracy_neutral")


class NeutralSimulatedMutationAccuracy(Figure):
    """
    """

    name = "neutral_simulated_mutation_accuracy"
    data_path = "simulated-data"
    filename = ["neutral_simulated_mutation_accuracy_mutations"]

    def plot(self):
        df = self.data[0]
        with sns.axes_style("white"):
            fig, ax = plt.subplots(
                nrows=2, ncols=2, figsize=(12, 12), sharex=True, sharey=True
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

            # tsdate on inferred tree
            self.mutation_accuracy(
                ax[0, 1], df["simulated_ts"], df["tsdate_inferred"], "tsinfer + tsdate"
            )

            # GEVA accuracy
            self.mutation_accuracy(ax[1, 0], df["simulated_ts"], df["geva"], "GEVA")

            # Relate accuracy
            self.mutation_accuracy(ax[1, 1], df["simulated_ts"], df["relate"], "Relate")

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
    filename = ["chr20_simulated_mutation_accuracy_mutations"]


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
        # ax[3].set_xticklabels(["\nDated \n with True Topology"])
        ax[1].tick_params(left="off")
        ax[2].tick_params(left="off")
        ax[3].tick_params(left="off")
        ax[0].set_title("i")
        ax[1].set_title("ii")
        ax[2].set_title("iii")
        ax[3].set_title("iv")
        # ax[2].set_xlabel('Simulated \nTopo')
        # plt.xlabel("Ancient Sample Size")
        plt.suptitle("Mutation Estimation Accuracy: " + self.plt_title)
        #plt.ylim(0.05, 0.8)
        #plt.tight_layout()
        #plt.savefig(self.filename[0] + ".pdf")
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
            sns.boxplot(x=lambda_results[lambda_results["ancient_sample_size"] == 0]["tsdateTime"], orient="v", ax=ax[ax_index, 0])

            sns.boxplot(
                x=lambda_results[lambda_results["ancient_sample_size"] == 0]["IterationTime"],
                orient="v",
                ax=ax[ax_index, 1],
            )
            sns.lineplot(
                x="ancient_sample_size",
                y="IterationTime",
                data=lambda_results[lambda_results["ancient_sample_size"] != 0],
                ax=ax[ax_index, 2],
            )
            sns.boxplot(x=lambda_results["tsinfer_keep_time"], orient="v", ax=ax[ax_index, 3])
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
        #plt.ylim(0.05, 0.8)
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


class AncientConstraints(Figure):
    """
    Figure 3: Ancient Constraints on Age of Mutations from 1000 Genomes Project
    """
    name = "ancient_constraints_1000g"
    data_path = "all-data"
    filename = ["tgp_muts_constraints"]
    plt_title = "ancient_constraint_1kg"

    def jitter(self, array):
        max_min = (np.max(array) - np.min(array))
        return array + np.random.randn(len(array)) * (max_min * 0.01) 

    def plot(self):
        df = self.data[0]
        fig = plt.figure(figsize=(15,5), constrained_layout=True)
        widths = [3, 3, 3, 0.1]
        spec5 = fig.add_gridspec(ncols=4, nrows=1, width_ratios=widths)
        ax0 = fig.add_subplot(spec5[0])
        ax0.set_xlim([1000, 2e5])
        ax0.set_ylim([1000, 9e6])
        ax0.set_xscale("log")
        ax0.set_yscale("log")

        ax1 = fig.add_subplot(spec5[1], sharex=ax0, sharey=ax0)
        ax2 = fig.add_subplot(spec5[2], sharex=ax0, sharey=ax0)
        ax3 = fig.add_subplot(spec5[3])
        scatter_size = 0.2
        scatter_alpha = 0.5
        ax0.scatter(
            self.jitter(df["Ancient Bound"]),
            30 * df["tsdate_upper_bound"],
            c=df["frequency"],
            s=scatter_size,
            alpha=scatter_alpha, cmap="plasma_r"
        )
        ax1.scatter(
            self.jitter(df["Ancient Bound"]),
            30 * df["relate_upper_bound"],
            c=df["frequency"],
            s=scatter_size,
            alpha=scatter_alpha, cmap="plasma_r"
        )
        ax2.scatter(
            self.jitter(df["Ancient Bound"]),
            30 * df["AgeCI95Upper_Jnt"],
            c=df["frequency"],
            s=scatter_size,
            alpha=scatter_alpha, cmap="plasma_r"
        )
        shading_alpha = 0.1
        diag = [ax0.get_xlim(), ax0.get_xlim()]
        upper_lim = ax0.get_ylim()
        ax0.plot(diag[0], diag[1], c="black")
        ax0.fill_between(diag[0], diag[1], (diag[1][0], diag[1][0]), color="red", alpha=shading_alpha)
        ax0.fill_between(diag[0], diag[1], (upper_lim[1], upper_lim[1]), color="blue", alpha=shading_alpha)
        ax1.plot(diag[0], diag[1], c="black")
        ax1.fill_between(diag[0], diag[1], [diag[1][0], diag[1][0]], color="red", alpha=shading_alpha)
        ax1.fill_between(diag[0], diag[1], (upper_lim[1], upper_lim[1]), color="blue", alpha=shading_alpha)
        ax2.plot(diag[0], diag[1], c="black")
        ax2.fill_between(diag[0], diag[1], [diag[1][0], diag[1][0]], color="red", alpha=shading_alpha)
        ax2.fill_between(diag[0], diag[1], (upper_lim[1], upper_lim[1]), color="blue", alpha=shading_alpha)
        ax0.text(
            0.25,
            0.1,
            "{0:.2f}% of variants >= lower bound".format(
                100 * np.sum((30 * df["tsdate_upper_bound"]) > df["Ancient Bound"])
                / df.shape[0]
            ),
            fontsize=10,
            transform=ax0.transAxes,
        )
        ax1.text(
            0.25,
            0.1,
            "{0:.2f}% of variants >= lower bound".format(
                100* np.sum((30 * df["relate_upper_bound"]) > df["Ancient Bound"])
                / df.shape[0]
            ),
            fontsize=10,
            transform=ax1.transAxes,
        )
        ax2.text(
            0.25,
            0.1,
            "{0:.2f}% of variants >= lower bound".format(
                100 * np.sum((30 * df["AgeCI95Upper_Jnt"]) > df["Ancient Bound"])
                / df.shape[0]
            ),
            fontsize=10,
            transform=ax2.transAxes,
        )

        ax0.set_xlabel(
            "Oldest ancient sample with derived allele (years)",
        )
        ax0.set_ylabel("tsdate Inferred Age")
        ax1.set_xlabel(
            "Oldest ancient sample with derived allele (years)",
        )
        ax1.set_ylabel("Relate Estimated Age")
        ax2.set_xlabel(
            "Oldest ancient sample with derived allele (years)",
        )

        ax2.set_ylabel("GEVA Estimated Age")
        ax0.set_title(
            "Ancient Derived Variant Lower Bound \n vs. tsdate upper bound",
        )
        ax1.set_title(
            "Ancient Derived Variant Lower Bound \n vs. relate upper bound",
        )
        ax2.set_title(
            "Ancient Derived Variant Lower Bound \n vs. GEVA upper bound",
        )

        cm = plt.cm.ScalarMappable(
            cmap="plasma_r",
            norm=plt.Normalize(vmin=np.min(df["frequency"]), vmax=np.max(df["frequency"]) + 0.1),
        )
        cbar = plt.colorbar(cm, format="%.1f", cax=ax3)
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
        self, ax, means_arr, time=False, memory=False, samplesize=False, length=False
    ):
        if memory:
            means_arr = [1e-9 * means for means in means_arr]
            ax.set_ylabel("Memory Requirements (Gb)", fontsize=12)
        elif time:
            means_arr = [means * (1 / 3600) for means in means_arr]
            ax.set_ylabel("CPU Runtime (hours)", fontsize=12)
        if samplesize:
            ax.set_xlabel("Sample Size", fontsize=12)
        elif length:
            ax.set_xlabel("Length (Kb)", fontsize=12)
        ax.plot(
            self.index, means_arr[0], label="tsdate", color=constants.colors["tsdate"]
        )
        ax.plot(
            self.index, means_arr[1], label="tsinfer", color=constants.colors["tsinfer"]
        )
        ax.plot(
            self.index,
            means_arr[0] + means_arr[1],
            label="tsinfer +\n tsdate",
            color=constants.colors["tsinfer + tsdate"],
        )
        ax.plot(
            self.index, means_arr[2], label="relate", color=constants.colors["relate"]
        )
        ax.plot(self.index, means_arr[3], label="geva", color=constants.colors["geva"])

    def plot_inset_ax(self, ax, means_arr, time=False, memory=False):
        axins1 = inset_axes(ax, width="40%", height="40%", loc=2, borderpad=1)
        if memory:
            means_arr = [1e-9 * means for means in means_arr]
        elif time:
            means_arr = [means * (1 / 3600) for means in means_arr]
        axins1.plot(
            self.index, means_arr[0], label="tsdate", color=constants.colors["tsdate"]
        )
        axins1.plot(
            self.index, means_arr[1], label="tsinfer", color=constants.colors["tsinfer"]
        )
        axins1.plot(
            self.index,
            means_arr[0] + means_arr[1],
            label="tsinfer + tsdate",
            color=constants.colors["tsinfer + tsdate"],
        )
        axins1.plot(
            self.index, means_arr[2], label="relate", color=constants.colors["relate"]
        )
        axins1.yaxis.tick_right()

    def plot(self):
        samples_scaling = self.data[0]
        length_scaling = self.data[1]
        samples_means = samples_scaling.groupby("sample_size").mean()
        length_means = length_scaling.groupby("length").mean()
        self.index = samples_means.index

        with sns.axes_style("white"):
            fig, ax = plt.subplots(
                nrows=1, ncols=2, figsize=(18, 6), sharex=False, sharey=False
            )
            self.plot_subplot(
                ax[0],
                [
                    samples_means["tsdate_cpu"],
                    samples_means["tsinfer_cpu"],
                    samples_means["relate_cpu"],
                    samples_means["geva_cpu"],
                ],
                time=True,
                samplesize=True,
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
            #            self.plot_subplot(
            #                ax[1, 0],
            #                [
            #                    samples_means["tsdate_memory"],
            #                    samples_means["tsinfer_memory"],
            #                    samples_means["relate_memory"],
            #                    samples_means["geva_memory"],
            #                ],
            #                memory=True, samplesize=True
            #            )
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
                ax[1],
                [
                    length_means["tsdate_cpu"],
                    length_means["tsinfer_cpu"],
                    length_means["relate_cpu"],
                    length_means["geva_cpu"],
                ],
                time=True,
                length=True,
            )
            #            self.plot_inset_ax(
            #                ax[0, 1],
            #                [
            #                    length_means["tsdate_cpu"],
            #                    length_means["tsinfer_cpu"],
            #                    length_means["relate_cpu"],
            #                ],
            #                time=True,
            #            )
            #            self.plot_subplot(
            #                ax[1, 1],
            #                [
            #                    length_means["tsdate_memory"],
            #                    length_means["tsinfer_memory"],
            #                    length_means["relate_memory"],
            #                    length_means["geva_memory"],
            #                ],
            #                memory=True, length=True
            #            )
            #            self.plot_inset_ax(
            #                ax[1, 1],
            #                [
            #                    length_means["tsdate_memory"],
            #                    length_means["tsinfer_memory"],
            #                    length_means["relate_memory"],
            #                ],
            #                memory=True,
            #            )
            #    for cur_ax in ax.reshape(-1):
            #        cur_ax.set_yscale("log")
            ax[0].set_title("Scaling with Sample Size, length fixed at 5Mb")
            ax[1].set_title("Scaling with Length, sample size fixed at 1000")
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc=7, fontsize=14, ncol=1)

        self.save(self.name)

class TgpMutEstsFrequency(Figure):
    """
    Figure showing TGP mutation age estimates from tsdate, Relate, GEVA vs. frequency.
    """

    name = "tgp_muts_frequency"
    data_path = "all-data"
    filename = ["tgp_mutations_allmethods_new_snipped_allsites"]
    plt_title = "TGP Mutation Age vs Frequency"

    def plot(self):
        comparable_mutations = self.data[0][
            ["tsdate_age", "relate_age", "AgeMean_Jnt", "frequency"]
        ]
        frequency = comparable_mutations["frequency"]
        fig, ax = plt.subplots(
            nrows=1, ncols=3, figsize=(15, 5), sharey=True, sharex=True
        )
        ax[0].scatter(
            frequency,
            comparable_mutations["tsdate_age"],
            s=0.2,
            alpha=0.2,
        )
        ax[1].scatter(
            frequency,
            comparable_mutations["relate_age"],
            s=0.2,
            alpha=0.2,
        )
        ax[2].scatter(
            frequency,
            comparable_mutations["AgeMean_Jnt"],
            s=0.2,
            alpha=0.2,
        )
        plt.xlim(3e-3, 1)
        plt.ylim(10, 2.2e5)
        plt.xscale("log")
        plt.yscale("log")
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
    filename = ["tgp_mutations_allmethods_new_snipped_allsites"]
    plt_title = "Compare Mutation Age Estimates"

    def plot(self):
        comparable_mutations = self.data[0][
            ["tsdate_age", "relate_age", "AgeMean_Jnt", "frequency"]
        ]
        frequency = comparable_mutations["frequency"]
        fig, ax = plt.subplots(
            nrows=1, ncols=3, figsize=(15, 5), sharey=True, sharex=True
        )
        ax[0].scatter(
            comparable_mutations["tsdate_age"],
            comparable_mutations["AgeMean_Jnt"],
            c=frequency,
            norm=mplc.LogNorm(),
            cmap="plasma_r",
            s=0.03,
            alpha=0.03,
        )
        ax[1].scatter(
            comparable_mutations["tsdate_age"],
            comparable_mutations["relate_age"],
            c=frequency,
            norm=mplc.LogNorm(),
            cmap="plasma_r",
            s=0.03,
            alpha=0.03,
        )
        ax[2].scatter(
            comparable_mutations["relate_age"],
            comparable_mutations["AgeMean_Jnt"],
            c=frequency,
            norm=mplc.LogNorm(),
            cmap="plasma_r",
            s=0.03,
            alpha=0.03,
        )
        plt.xlim(1, 2e5)
        plt.ylim(1, 2e5)
        plt.xscale("log")
        plt.yscale("log")
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
        cm = plt.cm.ScalarMappable(
            cmap="plasma_r",
            norm=plt.Normalize(vmin=np.min(frequency), vmax=np.max(frequency) + 0.1),
        )
        cbar = plt.colorbar(cm, format="%.1f")
        cbar.set_alpha(1)
        cbar.draw_all()
        cbar.set_label("Variant Frequency", rotation=270, labelpad=12)
        plt.tight_layout()

        self.save(self.name)


class TgpMutationAverageAge(Figure):
    """
    Compare mutation age estimates from tsdate, Relate, and GEVA for tgp chromosome 20.
    """

    name = "mutation_average_age"
    data_path = "all-data"
    filename = ["tgp_mutations_allmethods_new_snipped_allsites"]
    plt_title = "Average TGP Mutation Age"

    def plot(self):
        comparable_mutations = self.data[0][["tsdate_age", "relate_age", "AgeMean_Jnt"]]
        ax = sns.boxplot(
            data=comparable_mutations.rename(
                columns={
                    "tsdate_age": "tsdate",
                    "relate_age": "relate",
                    "AgeMean_Jnt": "GEVA",
                }
            )
        )
        ax.set_yscale("log")
        plt.ylabel("Estimated Mutation Age (generations)")
        plt.title(
            "Average Estimated Mutation Age from Thousand Genomes\n {} Mutations on Chromosome 20".format(
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
        # two_muts_count = np.unique(self.data[2]["recurrent_counts_two_muts"], return_counts=True)
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
            "Number of Mutations per Site, removing mutations leading to one or two samples"
        )
        ax[0].set_ylabel("Mutations per site")
        ax[1].set_ylabel("Mutations per site")
        ax[2].set_ylabel("Mutations per site")
        # fig.text(0.5, 0.04, 'Mutations per Site', ha='center')
        # fig.text(0.04, 0.5, 'Number of Sites', va='center', rotation='vertical')
        fig.tight_layout()
        # plt.suptitle("Recurrent Mutations in 1000g Tree Sequence (without singletons included)")
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
        "hgdp_missing_data_chr20_ma0.5_ms0.05_p15.simplify.recurrent_counts_nosamples_two_muts",
    ]
    plt_title = "hgdp_recurrent_mutations_fig"


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
        for name, fig in name_map.items():
            if fig in figures:
                fig().plot()
    else:
        fig = name_map[args.name]()
        fig.plot()


if __name__ == "__main__":
    main()
