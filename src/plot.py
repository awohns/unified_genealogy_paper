#!/usr/bin/env python3
"""
Generates all the actual figures. Should be called with following format:
 python3 src/plot.py PLOT_NAME
"""
import argparse
import collections
import json
import os
import pickle
from operator import attrgetter
import math

import scipy
from sklearn.metrics import mean_squared_log_error
import pandas as pd
import numpy as np
import dask.distributed
import dask.array as da
import numba
from Bio import SeqIO
import tskit

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import (
    inset_axes,
    mark_inset,
    zoomed_inset_axes,
)
import matplotlib.colors as mplc
import cartopy
import cartopy.crs as ccrs

from matplotlib.animation import FuncAnimation

import constants
import utility

sgdp_region_map = {
    "Abkhasian": "West Eurasia",
    "Adygei": "West Eurasia",
    "Albanian": "West Eurasia",
    "Aleut": "Central Asia/Siberia",
    "Altaian": "Central Asia/Siberia",
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
    "Chukchi": "Central Asia/Siberia",
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
    "Eskimo_Chaplin": "Central Asia/Siberia",
    "Eskimo_Naukan": "Central Asia/Siberia",
    "Eskimo_Sireniki": "Central Asia/Siberia",
    "Estonian": "West Eurasia",
    "Even": "Central Asia/Siberia",
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
    "Itelman": "Central Asia/Siberia",
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
    "Kyrgyz": "Central Asia/Siberia",
    "Lahu": "East Asia",
    "Lemande": "Africa",
    "Lezgin": "West Eurasia",
    "Luhya": "Africa",
    "Luo": "Africa",
    "Madiga": "South Asia",
    "Makrani": "South Asia",
    "Mala": "South Asia",
    "Mandenka": "Africa",
    "Mansi": "Central Asia/Siberia",
    "Maori": "Oceania",
    "Masai": "Africa",
    "Mayan": "Americas",
    "Mbuti": "Africa",
    "Mende": "Africa",
    "Miao": "East Asia",
    "Mixe": "Americas",
    "Mixtec": "Americas",
    "Mongola": "Central Asia/Siberia",
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
    "Tlingit": "Central Asia/Siberia",
    "Tubalar": "Central Asia/Siberia",
    "Tu": "East Asia",
    "Tujia": "East Asia",
    "Turkish": "West Eurasia",
    "Tuscan": "West Eurasia",
    "Ulchi": "Central Asia/Siberia",
    "Uygur": "East Asia",
    "Xibo": "East Asia",
    "Yadava": "South Asia",
    "Yakut": "Central Asia/Siberia",
    "Yemenite_Jew": "West Eurasia",
    "Yi": "East Asia",
    "Yoruba": "Africa",
    "Zapotec": "Americas",
}

hgdp_region_map = {
    "Brahui": "Central/South Asia",
    "Balochi": "Central/South Asia",
    "Hazara": "Central/South Asia",
    "Makrani": "Central/South Asia",
    "Sindhi": "Central/South Asia",
    "Pathan": "Central/South Asia",
    "Kalash": "Central/South Asia",
    "Burusho": "Central/South Asia",
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
    "Uygur": "Central/South Asia",
    "Dai": "East Asia",
    "Lahu": "East Asia",
    "She": "East Asia",
    "Naxi": "East Asia",
    "Tu": "East Asia",
    "Basque": "Europe",
    "Adygei": "Europe",
    "BantuKenya": "Africa",
}


tgp_region_map = {
    "CLM": "Americas",
    "MXL": "Americas",
    "PUR": "Americas",
    "PEL": "Americas",
    "LWK": "Africa",
    "ASW": "Africa",
    "GWD": "Africa",
    "MSL": "Africa",
    "YRI": "Africa",
    "ACB": "Africa",
    "ESN": "Africa",
    "CHS": "East Asia",
    "KHV": "East Asia",
    "JPT": "East Asia",
    "CHB": "East Asia",
    "CDX": "East Asia",
    "BEB": "South Asia",
    "STU": "South Asia",
    "GIH": "South Asia",
    "PJL": "South Asia",
    "ITU": "South Asia",
    "FIN": "Europe",
    "GBR": "Europe",
    "IBS": "Europe",
    "CEU": "Europe",
    "TSI": "Europe",
}


def get_tgp_hgdp_sgdp_region_colors():
    return {
        "East Asia": sns.color_palette("Greens", 2)[1],
        "West Eurasia": sns.color_palette("Blues", 1)[0],
        "Europe": sns.color_palette("Blues", 1)[0],
        "Africa": sns.color_palette("Wistia", 3)[0],
        "Americas": sns.color_palette("Reds", 2)[1],
        "South Asia": sns.color_palette("Purples", 2)[1],
        "Central/South Asia": sns.color_palette("Purples", 2)[1],
        "Middle East": matplotlib.colors.to_rgb(
            matplotlib.colors.get_named_colors_mapping()["teal"]
        ),
        "Oceania": matplotlib.colors.to_rgb(
            matplotlib.colors.get_named_colors_mapping()["saddlebrown"]
        ),
        "Central Asia/Siberia": matplotlib.colors.to_rgb(
            matplotlib.colors.get_named_colors_mapping()["pink"]
        ),
        "Ancients": matplotlib.colors.to_rgb(
            matplotlib.colors.get_named_colors_mapping()["orange"]
        ),
    }


region_colors = get_tgp_hgdp_sgdp_region_colors()


class Figure(object):
    """
    Superclass for creating figures. Each figure is a subclass
    """

    name = None
    data_path = None
    filename = None
    delimiter = None
    header = "infer"

    def main_ts(self):
        """Return unified TS used in multiple plots; cache if necessary"""
        try:
            return self._main_ts
        except AttributeError:
            self._main_ts = tskit.load(
                "all-data/hgdp_1kg_sgdp_high_cov_ancients_dated_chr20.trees")
            return self._main_ts


    def __init__(self):
        self.data = list()
        if self.filename is not None:
            for fn in self.filename:
                datafile_name = os.path.join(self.data_path, fn + ".csv")
                self.data.append(
                    pd.read_csv(
                        datafile_name, delimiter=self.delimiter, header=self.header
                    )
                )

    def save(self, figure_name=None, animation=None, bbox_inches="tight"):
        if figure_name is None:
            figure_name = self.name
        print("Saving figure '{}'".format(figure_name))
        if animation is not None:
            animation.save("figures/{}.mp4".format(figure_name), dpi=300)
        else:
            plt.savefig(
                "figures/{}.pdf".format(figure_name), bbox_inches="tight", dpi=400
            )
            plt.savefig(
                "figures/{}.png".format(figure_name), bbox_inches="tight", dpi=400
            )
            plt.savefig(
                "figures/{}.svg".format(figure_name), bbox_inches="tight", format="svg"
            )
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

    def mutation_accuracy(
        self, ax, x, y, label, cmap="Blues", kc_distance_0=None, kc_distance_1=None
    ):
        hb = ax.hexbin(
            x, y, xscale="log", yscale="log", bins="log", cmap=cmap, mincnt=1
        )
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        if label is not None:
            ax.set_title(label, fontsize=24, color=cmap[:-1])
        assert len(x) == len(y)
        ax.text(0.05, 0.9, str(len(x)) + " mutations", transform=ax.transAxes, size=14)
        ax.text(
            0.05,
            0.85,
            "RMSLE: " + "{0:.2f}".format(np.sqrt(mean_squared_log_error(x, y))),
            transform=ax.transAxes,
            size=14,
        )
        ax.text(
            0.05,
            0.8,
            "Pearson's r: "
            + "{0:.2f}".format(scipy.stats.pearsonr(np.log(x), np.log(y))[0]),
            transform=ax.transAxes,
            size=14,
        )
        ax.text(
            0.05,
            0.75,
            "Spearman's $\\rho$: " + "{0:.2f}".format(scipy.stats.spearmanr(x, y)[0]),
            transform=ax.transAxes,
            size=14,
        )
        ax.text(
            0.05,
            0.7,
            "Bias:" + "{0:.2f}".format(np.mean(y - x)),
            transform=ax.transAxes,
            size=14,
        )
        if kc_distance_0 is not None:
            ax.text(
                0.3,
                0.11,
                "KC Dist. ($\lambda$=0):" + "{:.2E}".format(kc_distance_0),
                transform=ax.transAxes,
                size=14,
            )
        if kc_distance_1 is not None:
            ax.text(
                0.3,
                0.03,
                "KC Dist. ($\lambda$=1):" + "{:.2E}".format(kc_distance_1),
                transform=ax.transAxes,
                size=14,
            )
        return hb


class TsdateNeutralSims(Figure):
    """
    Figure 1b: accuracy of tsdate on simulated data under a neutral model.
    Compares age of mutations: simulated time vs. tsdate estimation using
    simulated topology and tsdate using tsinfer inferred topologies.
    """

    name = "tsdate_neutral_sims"
    data_path = "simulated-data"
    filename = ["tsdate_neutral_sims_mutations"]

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
        self.mutation_accuracy(ax[0], true_vals[tsdate > 0], tsdate[tsdate > 0], None, cmap="Blues")
        ax[0].set_title("tsdate (using true topology)", fontsize=24)

        # tsdate on inferred tree
        hb = self.mutation_accuracy(ax[1], true_vals[tsdate_inferred > 0], tsdate_inferred[tsdate_inferred > 0], None, cmap="Blues")
        ax[1].set_title("tsinfer + tsdate", fontsize=24)
        fig.subplots_adjust(right=0.9)
        colorbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
        cb = fig.colorbar(hb, cax=colorbar_ax)
        cb.set_label("Number of Mutations")
        fig.text(0.5, 0.03, "True Mutation Ages (Generations)", size=20, ha="center")
        fig.text(
            0.03,
            0.5,
            "Estimated Mutation \n Ages (Generations)",
            size=20,
            va="center",
            rotation="vertical",
        )
        self.save(self.name)


class MismatchSimulationWithError(Figure):
    # Create the files (takes a day or so) using the Makefile in ../data/
    name = "mismatch_parameter_chr20_simulated"
    data_path = "data"
    filename = ["OutOfAfrica_3G09_chr20_n1500_seed1_ae0.01_results_plus_RF"]
    plt_title = "Effect of mismatch parameters on inference accuracy via simulation"

    focal_ma = 1
    focal_ms = 1
    cmap='viridis_r'
    linestyles = {
        "ma_mis_ratio": dict(linestyle="--", dashes=(10, 2)),
        "ms_mis_ratio": dict(linestyle=":"),
    }
    
    def plot(self):
        sim_data = self.data[0]
        sim_data['edges1000'] = sim_data['edges'] / 1000
        sim_data['muts1000'] = sim_data['muts'] / 1000
        sim_data['edges_muts_1000'] = (sim_data['edges1000'] + sim_data['muts1000'])

        assert np.allclose(np.diff(sim_data['kc_max'][np.isfinite(sim_data['kc_max'])]), 0)
        kc_max = np.mean(sim_data['kc_max'][np.isfinite(sim_data['kc_max'])])
        assert np.allclose(np.diff(sim_data['kc_max_split'][np.isfinite(sim_data['kc_max_split'])]), 0)
        kc_max_split = np.mean(sim_data['kc_max_split'][np.isfinite(sim_data['kc_max_split'])])
        assert np.allclose(np.diff(sim_data['num_sites']), 0)
        num_sites = np.mean(sim_data['num_sites'])

        sim_data['rel_ts_size'] = sim_data['ts_bytes']/sim_data['sim_ts_min_bytes']
        sim_data['KCpoly'] = sim_data['kc_poly']/kc_max
        sim_data['KCsplit'] = sim_data['kc_split']/kc_max_split
        # Rough RF max given by 2 * num_internal_nodes - 2 - if bifurcating, 2 * num_tips - 4
        #sim_data['RFsplit'] = sim_data['RFsplit'] / (2 * sim_data['n'] - 4)

        sim_data=sim_data.sort_values(["ma_mis_ratio", "ms_mis_ratio"])
        unique_vals = {
            "ma_mis_ratio": np.unique(sim_data["ma_mis_ratio"]),
            "ms_mis_ratio": np.unique(sim_data["ms_mis_ratio"]),
        }

        ma_map = {v:i for i, v in enumerate(unique_vals['ma_mis_ratio'])}
        ms_map = {v:i for i, v in enumerate(unique_vals['ms_mis_ratio'])}
        
        metrics = {
            "edges_muts_1000": "# edges + mutations (1000's)",
            "rel_ts_size": "Relative filesize",
            "KCpoly": "Normalised KC distance",
            "KCsplit": "Normalised KC distance (polytomies split)",
            #"RFsplit": "Normalised RF distance (polytomies split)",
            #"arity_mean": "Mean arity",
        }

        fig, axs = plt.subplots(2, len(metrics), figsize=(24, 10))
        plt.subplots_adjust(wspace=0.3)

        legend=False
        for i, (metric, metric_lab) in enumerate(metrics.items()):
            # Top (heatmap) plot
            Z = np.zeros((len(ms_map), len(ma_map)))
            for _, row in sim_data.iterrows():
                Z[ms_map[row.ms_mis_ratio], ma_map[row.ma_mis_ratio]] = row[metric]
            ax_top = axs[0, i]
            cs = ax_top.contour(
                unique_vals['ma_mis_ratio'], unique_vals['ms_mis_ratio'], Z, colors='gray')
            ax_top.contourf(cs, cmap=self.cmap)
            ax_top.clabel(cs, inline=0, colors=["k"])
            ax_top.axvline(self.focal_ma, c="k", **self.linestyles['ms_mis_ratio'])
            ax_top.axhline(self.focal_ms, c="k", **self.linestyles['ma_mis_ratio'])
            if i==0:
                ax_top.set_ylabel(r"Sample mismatch rate (relative to $\tilde{\rho}$)")
            ax_top.set_xlabel(r"Ancestor mismatch rate (relative to $\tilde{\rho}$)")
            ax_top.set_title(metric_lab, pad=15, fontsize="x-large")
            ax_top.set_xscale("log")
            ax_top.set_yscale("log")

            # Bottom (line) plot(s)
            ma_mask = sim_data["ma_mis_ratio"]==self.focal_ma
            ms_mask = sim_data["ms_mis_ratio"]==self.focal_ms
            ax_bottom = axs[1, i]
            if metric == "edges_muts_1000":
                # Edges vs muts plot is different
                gs = ax_bottom.get_gridspec()
                ax_bottom.set_ylabel(metric_lab, labelpad=35)
                ax_bottom.xaxis.set_visible(False) # make this subplot x axis invisible
                plt.setp(ax_bottom.spines.values(), visible=False) # make box invisible
                ax_bottom.tick_params(left=False, labelleft=False) # remove ticks+labels 
                gs_sub = gs[1,i].subgridspec(2, 1, hspace=0.5)
                for i, (mm_lab, mask, title) in enumerate([
                    ('ma_mis_ratio', ms_mask, 'Ancestor'),
                    ('ms_mis_ratio', ma_mask, 'Sample')]
                ):
                    ax_sub_bottom = fig.add_subplot(gs_sub[i, 0])
                    mm = sim_data[mm_lab][mask]
                    ax_sub_bottom.fill_between(
                        mm, 0, sim_data['muts1000'][mask], color="orange")
                    ax_sub_bottom.fill_between(
                        mm, sim_data['muts1000'][mask],
                        sim_data['muts1000'][mask] + sim_data['edges1000'][mask],
                        color="tab:brown"
                    )
                    ax_sub_bottom.plot(
                        sim_data[mm_lab][mask],
                        sim_data[metric][mask],
                        c="k", **self.linestyles[mm_lab])
                    ax_sub_bottom.text(
                        unique_vals[mm_lab][-2],
                        np.mean(sim_data['muts1000'][mask]/2),
                        "Mutations",
                        ha="right",
                        va="bottom",
                        bbox=dict(facecolor='w', alpha=0.9, ec='none')
                    )
                    ax_sub_bottom.text(
                        unique_vals[mm_lab][-2],
                        np.mean(
                            (
                                sim_data['muts1000'][ma_mask] * 2 +
                                sim_data['edges1000'][ma_mask]
                            ) / 2),
                        "Edges",
                        ha="right",
                        va="bottom",
                        bbox=dict(facecolor='w', alpha=0.9, ec='none')
                    )
                    ax_sub_bottom.set_xlabel(
                        title + r" mismatch rate (relative to $\tilde{\rho}$)")
                    ax_sub_bottom.set_xlim(np.min(mm), np.max(mm))
                    ax_sub_bottom.set_ylim(0)
                    ax_sub_bottom.set_xscale("log")
                    ax_sub_bottom.axhline(num_sites / 1000, c="grey")
                    ax_sub_bottom.text(
                        unique_vals[mm_lab][2], (num_sites / 1000) * 0.95, "Min # mutations",
                        va="center", color="k",
                        bbox=dict(boxstyle='square,pad=0', facecolor='orange', alpha=0.9, ec='none'))
            else:
                for i, (mm_lab, mask, title) in enumerate([
                    ('ma_mis_ratio', ms_mask, 'Ancestor'),
                    ('ms_mis_ratio', ma_mask, 'Sample')]
                ):
                    ax_bottom.plot(
                        sim_data[mm_lab][mask],
                        sim_data[metric][mask],
                        c="k", label=title + " mismatch",
                        **self.linestyles[mm_lab]
                    )
                if not legend:
                    ax_bottom.legend(loc='upper left')
                    legend = True
                ax_bottom.set_xlabel(r"Mismatch rate (relative to $\tilde{\rho}$)")
                ax_bottom.set_xlim(
                    np.min(np.concatenate(list(unique_vals.values()))),
                    np.max(np.concatenate(list(unique_vals.values()))))
                ax_bottom.set_ylabel(metric_lab)
                ax_bottom.set_xscale("log")

        self.save(self.name)

class MismatchRealDataWithError(Figure):
    # Create the files (takes a day or so) using the Makefile in ../data/
    name = "mismatch_parameter_chr20_tgp_hgdp"
    data_path = "data"
    filename = [
        "1kg_chr20_1000000-1100000_results",
        "hgdp_chr20_1000000-1100000_results",
    ]
    plt_title = "Effect of mismatch parameters on inference of real data"

    focal_ma = 1
    focal_ms = 1
    cmap='viridis_r'
    linestyles = {
        "ma_mis_ratio": dict(linestyle="--", dashes=(10, 2)),
        "ms_mis_ratio": dict(linestyle=":"),
    }

    def plot(self):
        metrics = {
            "edges_muts_1000": "# edges + mutations (1000's)",
            "ts_size_Mb": "Filesize (Mb)",
            #"arity_mean": "Mean arity",
        }
        fig, axs = plt.subplots(2, 5, figsize=(24, 10))
        plt.subplots_adjust(wspace=0.3)
        axs[0,2].remove()
        axs[1,2].remove()
        fig.text(
            0.55, 0.5, 'Human Genome Diversity Project',
            ha='center', va='center', rotation=90, size=30)

        for label, rot, lab_x, data, start_subplot_col in [
            ('Thousand Genomes Project', -90, 0.45, self.data[0], 0),
            ('Human Genome Diversity Project', 90, 0.55, self.data[1], 3),
        ]:
            data['ts_size_Mb'] = data['ts_bytes'] / 1e6
            data['edges1000'] = data['edges'] / 1000
            data['muts1000'] = data['muts'] / 1000
            data['edges_muts_1000'] = (data['edges1000'] + data['muts1000'])
            data=data.sort_values(["ma_mis_ratio", "ms_mis_ratio"])
            assert np.allclose(np.diff(data['num_sites']), 0)
            num_sites = np.mean(data['num_sites'])

            fig.text(lab_x, 0.5, label, ha='center', va='center', rotation=rot, size=30)
            unique_vals = {
                "ma_mis_ratio": np.unique(data["ma_mis_ratio"]),
                "ms_mis_ratio": np.unique(data["ms_mis_ratio"]),
            }
            ms_map = {v:i for i, v in enumerate(unique_vals['ms_mis_ratio'])}
            ma_map = {v:i for i, v in enumerate(unique_vals['ma_mis_ratio'])}
            legend = False
            for i, (metric, metric_lab) in enumerate(metrics.items()):
                Z = np.zeros((len(ma_map), len(ms_map)))
                for _, row in data.iterrows():
                    Z[ma_map[row.ma_mis_ratio], ms_map[row.ms_mis_ratio]] = row[metric]
        
                ax_top = axs[0, i + start_subplot_col]
                cs = ax_top.contour(
                    unique_vals['ms_mis_ratio'],
                    unique_vals['ma_mis_ratio'],
                    Z,
                    colors='gray',
                )
                ax_top.contourf(cs, cmap=self.cmap)
                ax_top.clabel(cs, inline=0, colors=["k"])
        
                ax_top.axvline(self.focal_ma, c="k", **self.linestyles['ms_mis_ratio'])
                ax_top.axhline(self.focal_ms, c="k", **self.linestyles['ma_mis_ratio'])
                if i==0:
                    ax_top.set_ylabel(
                        r"Sample mismatch rate (relative to $\tilde{\rho}$)")
                ax_top.set_xlabel(
                    r"Ancestor mismatch rate (relative to $\tilde{\rho}$)")
        
                ax_top.set_title(metric_lab, pad=15, fontsize="x-large")
                ax_top.set_xscale("log")
                ax_top.set_yscale("log")
        
        
                ma_mask = data["ma_mis_ratio"] == self.focal_ma
                ms_mask = data["ms_mis_ratio"] == self.focal_ms
        
                ax_bottom = axs[1, i + start_subplot_col]
                if metric == "edges_muts_1000":
                    gs = ax_bottom.get_gridspec()
                    ax_bottom.set_ylabel(metric_lab, labelpad=35)
                    ax_bottom.xaxis.set_visible(False)  # subplot x axis invisible
                    plt.setp(ax_bottom.spines.values(), visible=False)  # Box invisible
                    ax_bottom.tick_params(left=False, labelleft=False)  # No ticks/labels 
                    gs_sub = gs[1, i + start_subplot_col].subgridspec(2, 1, hspace=0.5)

                    for i, (mm_lab, mask, title) in enumerate([
                        ('ma_mis_ratio', ms_mask, 'Ancestor'),
                        ('ms_mis_ratio', ma_mask, 'Sample')]
                    ):
                        ax_sub_bottom = fig.add_subplot(gs_sub[i, 0])
                        mm = data[mm_lab][mask]
                        ax_sub_bottom.fill_between(
                            mm, 0, data['muts1000'][mask], color="orange")
                        ax_sub_bottom.fill_between(
                            mm, data['muts1000'][mask],
                            data['muts1000'][mask] + data['edges1000'][mask],
                            color="tab:brown"
                        )
                        ax_sub_bottom.plot(
                            data[mm_lab][mask],
                            data[metric][mask],
                            c="k", **self.linestyles[mm_lab])
                        ax_sub_bottom.text(
                            unique_vals[mm_lab][-2],
                            np.mean(data['muts1000'][mask]/2),
                            "Mutations",
                            ha="right",
                            va="bottom",
                            bbox=dict(facecolor='w', alpha=0.9, ec='none')
                        )
                        ax_sub_bottom.text(
                            unique_vals[mm_lab][-2],
                            np.mean((data['muts1000'][ma_mask]*2+data['edges1000'][ma_mask])/2),
                            "Edges",
                            ha="right",
                            va="bottom",
                            bbox=dict(facecolor='w', alpha=0.9, ec='none')
                        )
                        ax_sub_bottom.set_xlabel(
                            title + r" mismatch rate (relative to $\tilde{\rho}$)")
                        ax_sub_bottom.set_xlim(np.min(mm), np.max(mm))
                        ax_sub_bottom.set_ylim(0)
                        ax_sub_bottom.set_xscale("log")
                        ax_sub_bottom.axhline(num_sites / 1000, c="grey")
                        ax_sub_bottom.text(
                            unique_vals[mm_lab][2], (num_sites / 1000) * 0.95,
                            "Min # mutations",
                            va="center", color="k",
                            bbox=dict(
                                boxstyle='square,pad=0',
                                facecolor='orange', alpha=0.9, ec='none'))
                else:
                    ax_bottom.plot(
                        data["ms_mis_ratio"][ma_mask],
                        data[metric][ma_mask],
                        c="k", linestyle=":", label="Samples mismatch"
                    )
                    ax_bottom.plot(
                        data["m`_mis_ratio"][ms_mask],
                        data[metric][ms_mask],
                        c="k", linestyle="--", dashes=(10, 2), label="Ancestors mismatch"
                    )
                    if not legend:
                        ax_bottom.legend(loc='upper right')
                        legend = True
                    ax_bottom.set_xlabel("Mismatch rate")
                    ax_bottom.set_xlim(
                        np.min(np.concatenate(list(unique_vals.values()))),
                        np.max(np.concatenate(list(unique_vals.values()))),
                    )
                    ax_bottom.set_ylabel(metric_lab)
                    ax_bottom.set_xscale("log")
        
        self.save(self.name)


class Chr20AncientIteration(Figure):
    """
    Figure 1d. Accuracy of increasing number of ancient samples.
    """

    name = "chr20_ancient_iteration"
    data_path = "simulated-data"
    filename = [
        "chr20_ancient_iteration_msle",
        "chr20_ancient_iteration_spearman",
        "chr20_ancient_iteration_kc",
        "chr20_ancient_iteration_ooa_msle",
        "chr20_ancient_iteration_ooa_spearman",
        "chr20_ancient_iteration_ooa_kc",
        "chr20_ancient_iteration_amh_msle",
        "chr20_ancient_iteration_amh_spearman",
        "chr20_ancient_iteration_amh_kc",
    ]
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
        widths = [0.5, 0.5, 3]
        heights = [3, 3]
        gs_kw = dict(width_ratios=widths, height_ratios=heights)
        gs_kw.update(wspace=0.03)
        fig, ax = plt.subplots(
            ncols=3, nrows=2, constrained_layout=True, gridspec_kw=gs_kw, sharey="row"
        )

        msle = msle.apply(np.sqrt)
        msle_ooa = msle_ooa.apply(np.sqrt)
        msle_amh = msle_amh.apply(np.sqrt)
        df = msle
        comb_df = pd.concat([msle, msle_ooa, msle_amh])
        sns.boxplot(
            x=comb_df["tsdate_inferred"], orient="v", ax=ax[0, 0], color="silver"
        )
        sns.boxplot(
            x=comb_df["tsdate_iterate"], orient="v", ax=ax[0, 1], color="silver"
        )
        plt.setp(ax[0, 0].artists, edgecolor="k", facecolor="silver")
        plt.setp(ax[0, 1].artists, edgecolor="k", facecolor="silver")
        plt.setp(ax[0, 0].lines, color="k")
        plt.setp(ax[0, 1].lines, color="k")
        cols = ["Subset " + str(subset) for subset in [1, 5, 10]]
        df_melt = df.melt(value_vars=cols)
        df_melt["variable"] = df_melt["variable"].str.split().str[-1]

        sns.lineplot(
            x="variable",
            y="value",
            data=df_melt,
            sort=False,
            ax=ax[0, 2],
            alpha=0.8,
            color="grey",
        )
        groupby = df_melt.groupby("variable").mean()
        ax[0, 2].scatter(
            groupby.index, groupby["value"], s=80, color="black", zorder=3, alpha=0.8
        )

        df_melt = msle_ooa.melt(value_vars=cols)
        df_melt["variable"] = df_melt["variable"].str.split().str[-1]
        sns.lineplot(
            x="variable",
            y="value",
            data=df_melt,
            sort=False,
            ax=ax[0, 2],
            alpha=0.7,
            color="grey",
        )
        groupby = df_melt.groupby("variable").mean()
        ax[0, 2].scatter(
            groupby.index,
            groupby["value"],
            s=80,
            marker="X",
            color="black",
            zorder=3,
            alpha=0.8,
        )
        df_melt = msle_amh.melt(value_vars=cols)
        df_melt["variable"] = df_melt["variable"].str.split().str[-1]
        sns.lineplot(
            x="variable",
            y="value",
            data=df_melt,
            sort=False,
            ax=ax[0, 2],
            alpha=0.7,
            color="grey",
        )
        groupby = df_melt.groupby("variable").mean()
        ax[0, 2].scatter(
            groupby.index,
            groupby["value"],
            s=80,
            marker="P",
            color="black",
            zorder=3,
            alpha=0.8,
        )

        comb_df = pd.concat([spearman, spearman_ooa, spearman_amh])

        sns.boxplot(x=comb_df["inferred"], orient="v", ax=ax[1, 0], color="silver")
        sns.boxplot(x=comb_df["reinferred"], orient="v", ax=ax[1, 1], color="silver")
        plt.setp(ax[1, 0].artists, edgecolor="k", facecolor="silver")
        plt.setp(ax[1, 1].artists, edgecolor="k", facecolor="silver")
        plt.setp(ax[1, 0].lines, color="k")
        plt.setp(ax[1, 1].lines, color="k")

        cols = ["Subset " + str(subset) for subset in [1, 5, 10]]

        df_melt = spearman.melt(value_vars=cols)
        df_melt["variable"] = df_melt["variable"].str.split().str[-1]
        sns.lineplot(
            x="variable",
            y="value",
            data=df_melt,
            sort=False,
            ax=ax[1, 2],
            alpha=0.8,
            color="grey",
        )
        groupby = df_melt.groupby("variable").mean()
        ax[1, 2].scatter(
            groupby.index, groupby["value"], s=80, color="black", zorder=3, alpha=0.8
        )
        df_melt = spearman_ooa.melt(value_vars=cols)
        df_melt["variable"] = df_melt["variable"].str.split().str[-1]
        sns.lineplot(
            x="variable",
            y="value",
            data=df_melt,
            sort=False,
            ax=ax[1, 2],
            alpha=0.7,
            color="grey",
        )
        groupby = df_melt.groupby("variable").mean()
        ax[1, 2].scatter(
            groupby.index,
            groupby["value"],
            s=80,
            marker="X",
            color="black",
            zorder=3,
            alpha=0.8,
        )
        df_melt = spearman_amh.melt(value_vars=cols)
        df_melt["variable"] = df_melt["variable"].str.split().str[-1]
        sns.lineplot(
            x="variable",
            y="value",
            data=df_melt,
            sort=False,
            ax=ax[1, 2],
            alpha=0.7,
            color="grey",
        )
        groupby = df_melt.groupby("variable").mean()
        ax[1, 2].scatter(
            groupby.index,
            groupby["value"],
            s=80,
            marker="P",
            color="black",
            zorder=3,
            alpha=0.8,
        )

        ax[0, 1].set_ylabel("")
        ax[0, 0].set_ylabel("Root Mean Squared Log Error")
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

        # plt.suptitle("Mutation Estimation Accuracy: " + self.plt_title)
        self.save(self.name)


class TmrcaClustermap(Figure):
    """
    Figure 2: Plot the TMRCA clustermap
    """

    name = "tmrca_clustermap"
    data_path = "data"
    filename = [
        "hgdp_1kg_sgdp_high_cov_ancients_dated_chr20.20nodes_all.tmrcas"
    ]

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

        pop_names = tmrcas.columns
        pop_names = [pop.split(".")[0] for pop in pop_names]
        pop_names = [pop.split(" ")[0] for pop in pop_names]
        regions = list()
        pop_name_suffixes = list()
        for pop in pop_names[0:54]:
            pop_name_suffixes.append(pop + "_HGDP")
            regions.append(hgdp_region_map[pop])
        for pop in pop_names[54:80]:
            pop_name_suffixes.append(pop + "_TGP")
            regions.append(tgp_region_map[pop])
        for pop in pop_names[80:210]:
            pop_name_suffixes.append(pop + "_SGDP")
            regions.append(sgdp_region_map[pop])
        for pop in pop_names[210:]:
            pop_name_suffixes.append(pop)
            regions.append("Ancients")
        pop_names = pop_name_suffixes
        tmrcas.columns = pop_names
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
        row_colors = {}
        region_colors["Ancients"] = "orange"
        for pop_suffix, region in zip(tmrcas.columns, tmrcas["region"]):
            row_colors[pop_suffix] = region_colors[region]

        tmrcas = tmrcas.drop(columns="region")
        tmrcas.index = tmrcas.columns
        mergedg = tmrcas

        row_colors = pd.Series(row_colors)
        row_colors.name = "Region"
        tgp_origin = pd.Series(tgp_origin)
        tgp_origin.name = "TGP"
        hgdp_origin = pd.Series(hgdp_origin)
        hgdp_origin.name = "HGDP"
        sgdp_origin = pd.Series(sgdp_origin)
        sgdp_origin.name = "SGDP"
        ancient_origin = pd.Series(ancient_origin)
        ancient_origin.name = "Ancient"
        col_colors = pd.concat(
            [tgp_origin, hgdp_origin, sgdp_origin, ancient_origin], axis=1
        )
        mask = np.zeros_like(mergedg, dtype=np.bool)
        mask[np.tril_indices_from(mask, k=-1)] = True
        cg = sns.clustermap(mergedg, mask=mask, method="average")
        mask = mask[np.argsort(cg.dendrogram_row.reordered_ind), :]
        mask = mask[:, np.argsort(cg.dendrogram_col.reordered_ind)]
        cg = sns.clustermap(
            mergedg,
            mask=mask,
            method="average",
            xticklabels=True,
            yticklabels=True,
            figsize=(30, 30),
            rasterized=True,
            row_colors=row_colors,
            col_colors=col_colors,
            cbar_pos=(0.04, 0.28, 0.04, 0.2),
            cmap=plt.cm.inferno_r,
            dendrogram_ratio=0.18,
            cbar_kws=dict(orientation="vertical"),
        )
        cg.ax_heatmap.invert_xaxis()
        cg.ax_heatmap.xaxis.tick_top()
        cg.cax.tick_params(labelsize=20)
        cg.cax.set_xlabel("Average TMRCA\n(generations)", size=20)

        cg.ax_heatmap.set_xticklabels(
            [
                label.get_text().rsplit("_", 1)[0]
                for label in cg.ax_heatmap.get_xmajorticklabels()
            ],
            fontsize=7,
            rotation=90,
        )
        cg.ax_heatmap.set_yticks([])

        for region, col in region_colors.items():
            cg.ax_col_dendrogram.bar(0, 0, color=col, label=region, linewidth=0)

        cg.ax_col_dendrogram.set_xlim([0, 0])

        # Uncomment to Log Scale the Row Dendrogram
        # coord = np.array(cg.dendrogram_row.dependent_coord)
        # coord += 1
        # coord[coord!= 0] = np.log(coord[coord!= 0] )
        # cg.dendrogram_row.dependent_coord = coord.tolist()
        # cg.ax_row_dendrogram.clear()
        # cg.dendrogram_row.plot(cg.ax_row_dendrogram, {})

        pos = cg.ax_col_colors.get_position()
        cg.ax_col_colors.set_position(
            [pos.bounds[0], pos.bounds[1], pos.bounds[2], pos.bounds[3] / 5]
        )

        pos = cg.ax_col_colors.get_position()
        points = pos.get_points()
        points[0][1] = points[0][1] + 0.03  # - 0.72
        points[1][1] = points[1][1] + 0.03  # - 0.72
        cg.ax_col_colors.set_position(matplotlib.transforms.Bbox.from_extents(points))
        handles, labels = cg.ax_col_dendrogram.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles)))
        labels = np.array(labels)
        remove_bool = ~np.isin(labels, ["West Eurasia", "South Asia"])
        labels[labels == "Europe"] = "Europe/West Eurasia"
        handles = [handles[i] for i in np.where(remove_bool)[0]]
        labels = list(np.array(labels)[remove_bool])
        cg.ax_col_dendrogram.legend(
            handles,
            labels,
            loc="lower left",
            ncol=1,
            fontsize=20,
            frameon=True,
            bbox_to_anchor=(-0.25, -4.6),
            title="Region",
            title_fontsize=25,
        )

        # Remove box around the legend
        cg.ax_col_dendrogram.get_legend().get_frame().set_linewidth(0.0)

        self.save(self.name)


class InsetTmrcaHistograms(Figure):
    """
    Figure 2 inset: Plot the three inset histograms to the clustermap
    """

    name = "inset_tmrca_histograms"
    data_path = "data"
    filename = [
        "hgdp_1kg_sgdp_high_cov_ancients_dated_chr20.20nodes_all.tmrcas"
    ]

    def __init__(self):
        base_name = self.filename[0]
        hist_data = np.load(os.path.join(self.data_path, base_name + ".npz"))
        raw_data = np.load(os.path.join(self.data_path, base_name + "_RAW.npz"))
        raw_logtimes = raw_data[list(raw_data.keys())[0]]
        # Make data accessible to plot code: everything under 1 generation get put at 1
        self.raw_logtimes = np.where(np.exp(raw_logtimes) < 1, np.log(1), raw_logtimes)
        self.raw_weights = raw_data[list(raw_data.keys())[1]]
        self.data_rownames = hist_data["combos"]
        super().__init__()

    def plot(self):
        df = self.data[0]
        df = df.set_index(df.columns[0])
        # region_colors = get_tgp_hgdp_sgdp_region_colors()
        pop_names = df.columns
        pop_names = [pop.split(".")[0] for pop in pop_names]
        pop_names = np.array([pop.split(" ")[0] for pop in pop_names])

        regions = list()
        for pop in pop_names[0:54]:
            regions.append(hgdp_region_map[pop])
        for pop in pop_names[54:80]:
            regions.append(tgp_region_map[pop])
        for pop in pop_names[80:210]:
            regions.append(sgdp_region_map[pop])
        for _ in pop_names[210:]:
            regions.append("Ancients")
        regions = np.array(regions)

        def plot_hist(rows, label, color, num_bins, min_bin, ax, fill=False, alpha=1):
            ax.set_facecolor("lightgrey")
            av_weight = np.mean(self.raw_weights[rows, :], axis=0)
            assert av_weight.shape[0] == len(self.raw_logtimes)
            keep = av_weight != 0
            _, bins = np.histogram(
                self.raw_logtimes[keep],
                weights=av_weight[keep],
                bins=num_bins,
                range=[np.log(1), max(self.raw_logtimes)],
            )
            # If any bins are < 20 generations, merge them into the lowest bin
            bins = np.concatenate((bins[:1], bins[np.exp(bins) >= 20]))
            values, bins = np.histogram(
                self.raw_logtimes[keep],
                weights=av_weight[keep],
                bins=bins,
                density=True,
            )
            x1, y1 = (
                np.append(bins, bins[-1]),
                np.zeros(values.shape[0] + 2),
            )
            y1[1:-1] = values
            ax.step(x1, y1, "-", color=color, label=label)
            if fill:
                ax.fill_between(x1, y1, step="pre", color=color, alpha=alpha)
            ax.legend(fancybox=True, fontsize=18, facecolor="white")

        #############
        xticks = np.array([10, 20, 50, 1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4])
        minor_xticks = np.outer(10 ** np.arange(1, 5), np.arange(1, 10)).flatten()
        xmax = np.log(1e5)
        archaic_names = ["Altai", "Chagyrskaya", "Denisovan", "Vindija"]
        #############

        fig, axes = plt.subplots(
            3, 1, constrained_layout=True, figsize=(15, 10), sharex=True
        )

        # Bottom Plot:
        # Samaritan/Samaritan and Samaritan/Others
        xmin = np.log(10)
        ax2 = axes[2]
        params = {"num_bins": 60, "min_bin": 10, "ax": ax2}
        exsamaritan = np.logical_and(
            np.any(self.data_rownames == "Samaritan (SGDP)", axis=1),
            ~np.all(self.data_rownames == "Samaritan (SGDP)", axis=1),
        )
        exarchaic_exsamaritan = np.logical_and(
            exsamaritan, np.all(~np.isin(self.data_rownames, archaic_names), axis=1)
        )
        label, col = "Samaritan/Modern Humans \n(ex Samaritan)", "white"
        plot_hist(exarchaic_exsamaritan, label, col, fill=True, **params)
        samaritan = np.all(self.data_rownames == "Samaritan (SGDP)", axis=1)
        label, col = "Samaritan/Samaritan", region_colors["West Eurasia"]
        plot_hist(samaritan, label, col, **params)

        ax2.set_yticks([])
        ax2.set_xlim(xmin, xmax)
        ax2.set_xticks(np.log(xticks))
        ax2.set_xticks(np.log(minor_xticks[minor_xticks > np.exp(xmin)]), minor=True)
        ax2.set_xticklabels([str(int(x)) for x in xticks])

        # Middle Plot:
        # Papuan+Australian/Denisovan & Denisovan/modern humans (ex papuan + australian)
        xmin = np.log(100)
        ylim = axes[1].get_ylim()
        ax1 = axes[1].inset_axes(
            [xmin, ylim[0], xmax - xmin, ylim[1] - ylim[0]],
            transform=axes[1].transData,
        )
        params = {"num_bins": 60, "min_bin": 1000, "ax": ax1}
        sahul_names = [
            "Bougainville",
            "Bougainville (SGDP)",
            "PapuanHighlands",
            "PapuanSepik",
            "Australian",
        ]
        exsahul_denisovan = np.logical_and(
            np.any(self.data_rownames == "Denisovan", axis=1),
            np.all(~np.isin(self.data_rownames, sahul_names), axis=1),
        )
        neanderthal_names = ["Altai", "Vindija", "Chagyrskaya"]
        exarchaic_exsahul_denisovan = np.logical_and(
            exsahul_denisovan,
            np.all(~np.isin(self.data_rownames, neanderthal_names), axis=1),
        )
        exarchaic_exsahul_denisovan = np.logical_and(
            exarchaic_exsahul_denisovan,
            ~np.all(self.data_rownames == "Denisovan", axis=1),
        )
        label, col = "Denisovan/Modern Humans \n(ex Papauans, Australians)", "white"
        plot_hist(exarchaic_exsahul_denisovan, label, col, fill=True, **params)
        sahul = np.logical_and(
            np.any(self.data_rownames == "Denisovan", axis=1),
            np.any(np.isin(self.data_rownames, sahul_names), axis=1),
        )
        label, col = "Denisovan/Papuans+Australians", region_colors["Oceania"]
        plot_hist(sahul, label, col, **params)

        axes[1].axis("off")  # Hide the encapsulating axis
        ax1.set_yticks([])
        ax1.set_xlim([xmin, xmax])
        ax1.set_xticks(np.log(xticks[xticks > np.exp(xmin)]))
        ax1.set_xticks(np.log(minor_xticks[minor_xticks > np.exp(xmin)]), minor=True)
        ax1.set_xticklabels([])

        # Top Plot:
        # African/African and Non-African/Non-African
        xmin = np.log(1000)
        ylim = axes[0].get_ylim()
        ax0 = axes[0].inset_axes(
            [xmin, ylim[0], xmax - xmin, ylim[1] - ylim[0]], transform=axes[0].transData
        )
        params = {"num_bins": 60, "min_bin": 1000, "ax": ax0}
        african_names = pop_names[regions == "Africa"]
        african = np.all(np.isin(self.data_rownames, african_names), axis=1)
        label, col = "African/African", region_colors["Africa"]
        plot_hist(african, label, col, fill=True, alpha=0.4, **params)
        nonafrican_names = pop_names[
            np.logical_and(regions != "Africa", ~np.isin(pop_names, archaic_names))
        ]
        nonafricans = np.all(np.isin(self.data_rownames, nonafrican_names), axis=1)
        label, col = "Non-African/Non-African \n(ex Archaics)", "black"
        plot_hist(nonafricans, label, col, **params)

        axes[0].axis("off")  # Hide the encapsulating axis
        ax0.set_yticks([])
        ax0.set_xlim([xmin, xmax])
        ax0.set_xticks(np.log(xticks[xticks > np.exp(xmin)]))
        ax0.set_xticks(np.log(minor_xticks[minor_xticks > np.exp(xmin)]), minor=True)
        ax0.set_xticklabels([])
        plt.xlabel("Time to Most Recent Common Ancestor (generations)", fontsize=16)

        self.save(self.name)


class AncientConstraints(Figure):
    """
    Figure 3: Ancient Constraints on Age of Mutations from 1000 Genomes Project
    """

    name = "ancient_constraints_tgp"
    data_path = "data"
    filename = ["tgp_muts_constraints"]
    plt_title = "ancient_constraint_1kg"

    def jitter(self, array, log=True):
        max_min = np.max(array) - np.min(array)
        if log:
            return np.exp(
                np.log(array) + np.random.randn(len(array)) * (max_min * 0.0000003)
            )
        else:
            return array + np.random.randn(len(array))

    def plot(self):
        df = self.data[0]
        fig = plt.figure(figsize=(15, 5), constrained_layout=False)
        widths = [3, 3, 3, 0.1]
        spec5 = fig.add_gridspec(ncols=4, nrows=1, width_ratios=widths)
        for a in range(3):
            inner_spec = spec5[a].subgridspec(
                ncols=2, nrows=1, wspace=0, hspace=0, width_ratios=[1, 10]
            )
            if a == 0:
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
                ancient.spines["left"].set_visible(False)
                ancient.yaxis.set_visible(False)
                ax_main = [
                    [contemp, ancient],
                ]
            else:
                ax_main.append(
                    [
                        fig.add_subplot(inner_spec[0], sharey=contemp, sharex=contemp),
                        fig.add_subplot(inner_spec[1], sharey=contemp, sharex=ancient),
                    ]
                )
                ax_main[-1][0].set_xticks([0])
                ax_main[-1][0].set_xticklabels(["0"])
                ax_main[-1][1].spines["left"].set_visible(False)
                ax_main[-1][1].yaxis.set_visible(False)
        ax_scale = fig.add_subplot(spec5[3])
        ax_scale.set_yscale("linear")

        # Make one column a list of all relate estimates
        relate_estimates = [c for c in df.columns if "est_" in c]
        # df= df.melt(id_vars="position", value_vars=relate_estimates, var_name="relate_pop_est", value_name="relate_age")
        # df["relate_age"] = list(df[relate_estimates].to_numpy())
        relate_upper_estimates = [c for c in df.columns if "upper_age_" in c]
        # df = df.melt(id_vars="position", value_vars=relate_upper_estimates, var_name="relate_pop_upper", value_name="relate_upper_age")
        # df["relate_upper_age"] = list(df[relate_upper_estimates].to_numpy())

        df["Ancient Bound"] = df["Ancient Bound"] * constants.GENERATION_TIME
        df = df[df["tsdate_frequency"] > 0]

        # df_old contains mutations seen in ancients, df_new is only contemporary
        df_old = df[df["Ancient Bound"] > 0].set_index("Ancient Bound").sort_index()
        df_new = df[np.logical_not(df["Ancient Bound"] > 0)]
        df_old["Ancient Bound Bins"] = pd.cut(df_old.index, 30)
        smoothed_mean = df_old.groupby("Ancient Bound Bins").mean()
        smoothed_mean["bin_right"] = smoothed_mean.index.map(attrgetter("right"))
        smoothed_mean = smoothed_mean.dropna()

        scatter_size = 0.2
        scatter_alpha = 0.2
        shading_alpha = 0.2
        # Plot each method. Relate has estimates for each population, so plot all estimates
        for i, method in enumerate(
            [
                # Hack the titles with extra spaces to centre properly, as it's too
                # tricky to centre over a pair or subplots
                ("tsdate      ", ["tsdate_upper_bound", "tsdate_age"]),
                ("Relate      ", [relate_upper_estimates, relate_estimates]),
                ("GEVA      ", ["AgeCI95Upper_Jnt", "AgeMean_Jnt"]),
            ]
        ):
            ax = ax_main[i][0]
            if (
                i == 1
            ):  # Relate has a list of estimates, so needs to be handled differently
                for estimate in method[1][1]:
                    cur_df = df_new.dropna(subset=[estimate])
                    ax.scatter(
                        self.jitter(np.zeros(len(cur_df.index)), log=False),
                        constants.GENERATION_TIME * cur_df[estimate],
                        c=cur_df["tsdate_frequency"],
                        s=scatter_size / 2,
                        alpha=scatter_alpha / 6,
                        cmap="plasma_r",
                        norm=mplc.LogNorm(
                            vmin=np.min(cur_df["tsdate_frequency"]), vmax=1
                        ),
                    )
                age_est = np.unique(
                    df_old[relate_estimates].apply(
                        lambda x: np.where(
                            constants.GENERATION_TIME * x > df_old.index, 1, -1
                        ),
                        axis=0,
                    )[~pd.isna(df_old[relate_estimates])],
                    return_counts=True,
                )
                age_accuracy = 100 * (age_est[1][1] / (age_est[1][0] + age_est[1][1]))
                upper_age_est = np.unique(
                    df_old[relate_upper_estimates].apply(
                        lambda x: np.where(
                            constants.GENERATION_TIME * x > df_old.index, 1, -1
                        ),
                        axis=0,
                    )[~pd.isna(df_old[relate_upper_estimates])],
                    return_counts=True,
                )
                upper_age_accuracy = 100 * (
                    upper_age_est[1][1] / (upper_age_est[1][0] + upper_age_est[1][1])
                )
            else:
                ax.scatter(
                    self.jitter(np.zeros(len(df_new.index)), log=False),
                    constants.GENERATION_TIME * df_new[method[1][1]],
                    c=df_new["tsdate_frequency"],
                    s=scatter_size / 2,
                    alpha=scatter_alpha / 6,
                    cmap="plasma_r",
                    norm=mplc.LogNorm(vmin=np.min(df_new["tsdate_frequency"]), vmax=1),
                )
                upper_age_accuracy = (
                    100
                    / df_old.shape[0]
                    * np.sum(
                        (constants.GENERATION_TIME * df_old[method[1][0]])
                        > df_old.index
                    )
                )
                age_accuracy = (
                    100
                    / df_old.shape[0]
                    * np.sum(
                        (constants.GENERATION_TIME * df_old[method[1][1]])
                        > df_old.index
                    )
                )

            ax = ax_main[i][1]
            ax.set_title(method[0])
            ax.text(
                0.1,
                0.09,
                "Ancient Derived Variant Lower Bound",
                rotation=39.6,
                transform=ax.transAxes,
            )
            diag = [ax.get_xlim(), ax.get_xlim()]
            ax.plot(diag[0], diag[1], "--", c="black")
            ax.fill_between(
                diag[0],
                diag[1],
                (diag[1][0], diag[1][0]),
                color="grey",
                alpha=shading_alpha,
            )
            ax.text(
                0.16,
                0.06,
                "{0:.2f}% est. upper bound $>=$ lower bound".format(upper_age_accuracy),
                fontsize=8,
                transform=ax.transAxes,
            )
            ax.text(
                0.16,
                0.02,
                "{0:.2f}% est. age $>=$ lower bound".format(age_accuracy),
                fontsize=8,
                transform=ax.transAxes,
            )
            if i == 1:
                for estimate in method[1][1]:
                    cur_df = df_old.dropna(subset=[estimate])
                    scatter = ax.scatter(
                        self.jitter(cur_df.index),
                        constants.GENERATION_TIME * cur_df[estimate],
                        c=cur_df["tsdate_frequency"],
                        s=scatter_size / 2,
                        alpha=scatter_alpha,
                        cmap="plasma_r",
                        norm=mplc.LogNorm(
                            vmin=np.min(cur_df["tsdate_frequency"]), vmax=1
                        ),
                    )
                # If Relate, take average across population estimates
                ax.plot(
                    smoothed_mean["bin_right"].astype(int).values,
                    np.mean(
                        constants.GENERATION_TIME * smoothed_mean[method[1][1]].values,
                        axis=1,
                    ),
                    alpha=0.7,
                    marker="P",
                    color="black",
                )

            else:
                scatter = ax.scatter(
                    self.jitter(df_old.index),
                    constants.GENERATION_TIME * df_old[method[1][1]],
                    c=df_old["tsdate_frequency"],
                    s=scatter_size,
                    alpha=scatter_alpha,
                    cmap="plasma_r",
                    norm=mplc.LogNorm(vmin=np.min(df_old["tsdate_frequency"]), vmax=1),
                )
                ax.plot(
                    smoothed_mean["bin_right"].astype(int).values,
                    constants.GENERATION_TIME * smoothed_mean[method[1][1]].values,
                    alpha=0.7,
                    marker="P",
                    color="black",
                )

        fig.text(
            0.5,
            0.01,
            "Age of oldest sample with derived allele (years)",
            ha="center",
            size=15,
        )
        fig.text(
            0.08,
            0.5,
            "Estimated age (years)",
            va="center",
            rotation="vertical",
            size=15,
        )

        cbar = plt.colorbar(
            scatter, format="%.3f", cax=ax_scale, ticks=[0.001, 0.01, 0.1, 0.5, 1]
        )
        cbar.set_alpha(1)
        cbar.draw_all()
        cbar.set_label("Variant Frequency", rotation=270, labelpad=12)
        plt.show()
        self.save(self.name)


class ScalingFigure(Figure):
    """
    Extended Data Figure 5: CPU and memory scaling of tsdate, tsinfer, Relate and GEVA.
    With both samples and length of sequence.
    """

    name = "scaling"
    data_path = "simulated-data"
    filename = ["cpu_scaling_samplesize", "cpu_scaling_length"]
    plt_title = "scaling_fig"
    include_geva = False
    col_1_name = "Length fixed at 1Mb"
    col_2_name = "Sample size fixed at 250"

    def plot_subplot(
        self,
        ax,
        index,
        means_arr,
        time=False,
        memory=False,
        samplesize=False,
        length=False,
        xlabel=False,
        ylabel=False,
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
            ax.set_xlabel("Length (Gb)", fontsize=12)
        ax.plot(
            index,
            means_arr[0],
            ":",
            label="tsdate",
            color=constants.colors["tsdate"],
            marker="^",
        )
        ax.plot(
            index,
            means_arr[1],
            "--",
            label="tsinfer",
            color=constants.colors["tsdate"],
            marker="v",
        )
        ax.plot(
            index,
            means_arr[0] + means_arr[1],
            label="tsinfer +\n tsdate",
            color=constants.colors["tsdate"],
            marker="D",
        )
        ax.plot(
            index,
            means_arr[2],
            label="Relate",
            color=constants.colors["relate"],
            marker="h",
        )

        if self.include_geva:
            ax.plot(
                index,
                means_arr[3],
                label="GEVA",
                color=constants.colors["geva"],
                marker="s",
            )
        max_val = np.max(means_arr[2])
        ax.set_ylim(0, max_val + (0.05 * max_val))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    def plot_inset_ax(self, ax, index, means_arr, time=False, memory=False, left=True):
        if left:
            left_pos = 0.01
        else:
            left_pos = 0.55
        axins1 = inset_axes(
            ax,
            width="40%",
            height="40%",
            loc=2,
            borderpad=2,
            bbox_to_anchor=(left_pos, 0.05, 1, 1),
            bbox_transform=ax.transAxes,
        )
        if memory:
            means_arr = [1e-9 * means for means in means_arr]
        elif time:
            means_arr = [means * (1 / 3600) for means in means_arr]
        axins1.plot(
            index,
            means_arr[0],
            ":",
            label="tsdate",
            color=constants.colors["tsdate"],
            marker="^",
        )
        axins1.plot(
            index,
            means_arr[1],
            "--",
            label="tsinfer",
            color=constants.colors["tsdate"],
            marker="v",
        )
        axins1.plot(
            index,
            means_arr[0] + means_arr[1],
            label="tsinfer + tsdate",
            color=constants.colors["tsdate"],
            marker="D",
        )
        axins1.plot(
            index,
            means_arr[2],
            label="relate",
            color=constants.colors["relate"],
            marker="h",
        )
        axins1.plot(
            index,
            means_arr[3],
            label="GEVA",
            color=constants.colors["geva"],
            marker="s",
        )
        axins1.tick_params(axis="both", labelsize=7)
        return axins1

    def plot(self):
        samples_scaling = self.data[0]
        length_scaling = self.data[1]
        samples_means = samples_scaling.groupby("sample_size").mean()
        length_means = length_scaling.groupby("length").mean()
        self.samples_index = samples_means.index
        self.length_index = length_means.index / 1000000

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 9), sharex=False,)
        self.plot_subplot(
            ax[0, 0],
            self.samples_index,
            [
                samples_means["tsdate_infer_cpu"],
                samples_means["tsinfer_cpu"],
                samples_means["relate_cpu"],
                samples_means["geva_cpu"],
            ],
            time=True,
            samplesize=True,
            ylabel=True,
        )
        self.plot_inset_ax(
            ax[0, 0],
            self.samples_index,
            [
                samples_means["tsdate_infer_cpu"],
                samples_means["tsinfer_cpu"],
                samples_means["relate_cpu"],
                samples_means["geva_cpu"],
            ],
            time=True,
        )
        self.plot_subplot(
            ax[1, 0],
            self.samples_index,
            [
                samples_means["tsdate_infer_memory"],
                samples_means["tsinfer_memory"],
                samples_means["relate_memory"],
                samples_means["geva_memory"],
            ],
            memory=True,
            samplesize=True,
            xlabel=True,
            ylabel=True,
        )
        self.plot_inset_ax(
            ax[1, 0],
            self.samples_index,
            [
                samples_means["tsdate_infer_memory"],
                samples_means["tsinfer_memory"],
                samples_means["relate_memory"],
                samples_means["geva_memory"],
            ],
            memory=True,
        )
        self.plot_subplot(
            ax[0, 1],
            self.length_index,
            [
                length_means["tsdate_infer_cpu"],
                length_means["tsinfer_cpu"],
                length_means["relate_cpu"],
                length_means["geva_cpu"],
            ],
            time=True,
            length=True,
            ylabel=True,
        )
        self.plot_inset_ax(
            ax[0, 1],
            self.length_index,
            [
                length_means["tsdate_infer_cpu"],
                length_means["tsinfer_cpu"],
                length_means["relate_cpu"],
                length_means["geva_cpu"],
            ],
            time=True,
        )
        self.plot_subplot(
            ax[1, 1],
            self.length_index,
            [
                length_means["tsdate_infer_memory"],
                length_means["tsinfer_memory"],
                length_means["relate_memory"],
                length_means["geva_memory"],
            ],
            memory=True,
            length=True,
            xlabel=True,
            ylabel=True,
        )
        axins = self.plot_inset_ax(
            ax[1, 1],
            self.length_index,
            [
                length_means["tsdate_infer_memory"],
                length_means["tsinfer_memory"],
                length_means["relate_memory"],
                length_means["geva_memory"],
            ],
            memory=True,
        )
        ax[0, 1].get_xaxis().get_major_formatter().set_scientific(False)
        ax[1, 1].get_xaxis().get_major_formatter().set_scientific(False)
        ax[0, 0].set_title(self.col_1_name)
        ax[0, 1].set_title(self.col_2_name)
        handles, labels = ax[0, 0].get_legend_handles_labels()
        insert_handles, insert_labels = axins.get_legend_handles_labels()
        fig.legend(
            handles + [insert_handles[-1]],
            labels + [insert_labels[-1]],
            fontsize=14,
            ncol=1,
            loc=7,
        )
        self.save(self.name)


class TgpMutEstsFrequency(Figure):
    """
    Supplementary Figure 5: Figure showing TGP mutation age estimates from tsdate,
    Relate, GEVA vs. frequency.
    """

    name = "tgp_muts_frequency"
    data_path = "data"
    filename = ["tgp_mutations"]
    plt_title = "TGP Mutation Age vs Frequency"

    def plot(self):
        comparable_mutations = self.data[0][
            ["tsdate_age", "relate_avg_age", "AgeMean_Jnt", "tsdate_frequency"]
        ]
        comparable_mutations = comparable_mutations[
            comparable_mutations["tsdate_age"] > 0
        ]
        frequency = comparable_mutations["tsdate_frequency"]
        fig, ax = plt.subplots(
            nrows=1, ncols=3, figsize=(15, 5), sharey=True, sharex=True
        )
        ax[0].hexbin(
            frequency,
            comparable_mutations["tsdate_age"],
            xscale="log",
            yscale="log",
            bins="log",
            cmap="Blues",
            mincnt=1,
        )
        ax[1].hexbin(
            frequency,
            comparable_mutations["relate_avg_age"],
            xscale="log",
            yscale="log",
            bins="log",
            cmap="Greens",
            mincnt=1,
        )
        ax[2].hexbin(
            frequency,
            comparable_mutations["AgeMean_Jnt"],
            xscale="log",
            yscale="log",
            bins="log",
            cmap="Reds",
            mincnt=1,
        )
        plt.xlim(3e-3, 1.05)
        plt.ylim(10, 2.4e5)
        ax[0].set_title("Frequency vs. GEVA Estimated Allele Age")
        ax[1].set_title("Frequency vs. Relate Estimated Allele Age")
        ax[2].set_title("Frequency vs. GEVA Estimated Allele Age")
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
    Supplementary Figure 7: Comparing TGP mutation age estimates from tsdate, Relate,
    and GEVA.
    """

    name = "tgp_dates_comparison"
    data_path = "data"
    filename = ["tgp_mutations"]
    plt_title = "Compare Mutation Age Estimates"

    def plot(self):
        df = self.data[0]
        relate_estimates = [c for c in df.columns if "est_" in c]
        comparable_mutations = df[
            ["tsdate_age", "AgeMean_Jnt", "tsdate_frequency"] + relate_estimates]
        fig, ax = plt.subplots(
            nrows=1, ncols=3, figsize=(15, 5), sharey=True, sharex=True
        )
        ax[0].hexbin(
            comparable_mutations["tsdate_age"],
            comparable_mutations["AgeMean_Jnt"],
            xscale="log",
            yscale="log",
            bins="log",
            mincnt=1,
        )
        comparable_mutations = comparable_mutations.melt(id_vars=["tsdate_age", "AgeMean_Jnt", "tsdate_frequency"], 
            var_name="relate_est", 
            value_name="relate_age")
        ax[1].hexbin(
            comparable_mutations["tsdate_age"],
            comparable_mutations["relate_age"],
            xscale="log",
            yscale="log",
            bins="log",
            mincnt=1,
        )

        ax[2].hexbin(
            comparable_mutations["relate_age"],
            comparable_mutations["AgeMean_Jnt"],
            xscale="log",
            yscale="log",
            bins="log",
            mincnt=1,
        )

        plt.xlim(1, 2e5)
        plt.ylim(1, 2e5)
        ax[0].set_title("tsdate vs. GEVA Estimated Allele Age")
        ax[1].set_title("tsdate vs. Relate Estimated Allele Age")
        ax[2].set_title("Relate vs. GEVA Estimated Allele Age")
        ax[0].set_xlabel("Estimated Age by tsdate (generations)")
        ax[0].set_ylabel("Estimated Age by GEVA (generations)")
        ax[1].set_xlabel("Estimated Age by tsdate (generations)")
        ax[1].set_ylabel("Estimated Age by Relate (generations)")
        ax[2].set_xlabel("Estimated Age by Relate (generations)")
        ax[2].set_ylabel("Estimated Age by GEVA (generations)")
        ax[0].plot([0.1, 4e5], [0.1, 4e5], c="black")
        ax[1].plot([0.1, 4e5], [0.1, 4e5], c="black")
        ax[2].plot([0.1, 4e5], [0.1, 4e5], c="black")
        plt.tight_layout()

        self.save(self.name)


class TgpMutationAverageAge(Figure):
    """
    Supplementary Figure 6: Compare mutation age estimates from tsdate, Relate, and
    GEVA for tgp chromosome 20.
    """

    name = "mutation_average_age"
    data_path = "data"
    filename = ["tgp_mutations"]
    plt_title = "Average TGP Mutation Age"

    def plot(self):
        comparable_mutations = self.data[0][
            ["tsdate_age", "relate_avg_age", "AgeMean_Jnt"]
        ]
        comparable_mutations = comparable_mutations[
            comparable_mutations["tsdate_age"] > 0
        ]
        ax = sns.boxplot(
            data=comparable_mutations.rename(
                columns={
                    "tsdate_age": "tsdate",
                    "relate_avg_age": "relate",
                    "AgeMean_Jnt": "GEVA",
                }
            ),
        )
        ax.set_yscale("log")
        ax.artists[0].set_facecolor("blue")
        ax.artists[1].set_facecolor("green")
        ax.artists[2].set_facecolor("red")
        plt.ylabel("Estimated Allele Age (generations)")
        plt.title(
            "Estimated TGP Allele Ages \n {} Variant Sites on Chr 20".format(
                comparable_mutations.shape[0]
            )
        )
        self.save(self.name)


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
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(1.8, 1050)
        plt.ylim(1e-3, 4e5)
        all_results = self.data

        for index, ((_, result), mixtures) in enumerate(
            zip(all_results.items(), [False, False, False, False])
        ):
            num_tips_all = np.concatenate(result["num_tips"]).ravel()
            num_tips_all_int = num_tips_all.astype(int)
            only_mixtures = np.full(len(num_tips_all), True)
            if mixtures:
                only_mixtures = np.where((num_tips_all - num_tips_all_int) != 0)[0]

            upper_bound_all = np.concatenate(result["upper_bound"]).ravel()[
                only_mixtures
            ]
            lower_bound_all = np.concatenate(result["lower_bound"]).ravel()[
                only_mixtures
            ]
            expectations_all = np.concatenate(result["expectations"]).ravel()[
                only_mixtures
            ]

            real_ages_all = np.concatenate(result["real_ages"]).ravel()[only_mixtures]
            num_tips_all = num_tips_all[only_mixtures]
            yerr = [
                expectations_all - lower_bound_all,
                upper_bound_all - expectations_all,
            ]

            axes[index].errorbar(
                num_tips_all,
                expectations_all,
                ls="none",
                yerr=yerr,
                elinewidth=0.3,
                alpha=0.4,
                color="grey",
                zorder=1,
                label="95% credible interval of the prior",
            )
            axes[index].scatter(
                num_tips_all,
                real_ages_all,
                s=1,
                alpha=0.5,
                zorder=2,
                color="blue",
                label="True Time",
            )
            axes[index].scatter(
                num_tips_all,
                expectations_all,
                s=1,
                color="red",
                zorder=3,
                label="expected time",
                alpha=0.5,
            )
            coverage = np.sum(
                np.logical_and(
                    real_ages_all < upper_bound_all, real_ages_all > lower_bound_all
                )
            ) / len(expectations_all)
            axes[index].text(
                0.35,
                0.25,
                "Overall Coverage Probability:" + "{0:.3f}".format(coverage),
                size=10,
                ha="center",
                va="center",
                transform=axes[index].transAxes,
            )
            less5_tips = np.where(num_tips_all < 5)[0]
            coverage = np.sum(
                np.logical_and(
                    real_ages_all[less5_tips] < upper_bound_all[less5_tips],
                    (real_ages_all[less5_tips] > lower_bound_all[less5_tips]),
                )
                / len(expectations_all[less5_tips])
            )
            axes[index].text(
                0.35,
                0.21,
                "<10 Tips Coverage Probability:" + "{0:.3f}".format(coverage),
                size=10,
                ha="center",
                va="center",
                transform=axes[index].transAxes,
            )
            mrcas = np.where(num_tips_all == 1000)[0]
            coverage = np.sum(
                np.logical_and(
                    real_ages_all[mrcas] < upper_bound_all[mrcas],
                    (real_ages_all[mrcas] > lower_bound_all[mrcas]),
                )
                / len(expectations_all[mrcas])
            )
            axes[index].text(
                0.35,
                0.17,
                "MRCA Coverage Probability:" + "{0:.3f}".format(coverage),
                size=10,
                ha="center",
                va="center",
                transform=axes[index].transAxes,
            )
            axins = zoomed_inset_axes(
                axes[index],
                2.7,
                loc=4,
                bbox_to_anchor=(0.95, 0.1),
                bbox_transform=axes[index].transAxes,
            )
            axins.errorbar(
                num_tips_all,
                expectations_all,
                ls="none",
                yerr=yerr,
                elinewidth=0.7,
                alpha=0.1,
                color="grey",
                solid_capstyle="projecting",
                capsize=4,
                label="95% credible interval of the prior",
                zorder=1,
            )
            axins.scatter(
                num_tips_all,
                real_ages_all,
                s=2,
                color="blue",
                alpha=0.5,
                label="True Time",
                zorder=2,
            )
            axins.scatter(
                num_tips_all,
                expectations_all,
                s=2,
                color="red",
                label="Expected time",
                alpha=0.5,
                zorder=3,
            )
            x1, x2, y1, y2 = 970, 1030, 5e3, 3e5
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xscale("log")
            axins.set_yscale("log")
            axins.set_xticks([], minor=True)
            axins.set_yticks([], minor=True)
            mark_inset(axes[index], axins, loc1=2, loc2=1, fc="none", ec="0.5")
        lgnd = axes[3].legend(loc=4, prop={"size": 12}, bbox_to_anchor=(1, -0.3))
        lgnd.legendHandles[0]._sizes = [30]
        lgnd.legendHandles[1]._sizes = [30]
        lgnd.legendHandles[2]._linewidths = [2]
        fig.text(0.5, 0.04, "Number of Tips", ha="center", size=15)
        fig.text(
            0.04,
            0.5,
            "Node Age (Generations)",
            va="center",
            rotation="vertical",
            size=15,
        )
        axes[0].set_title("p=0", size=14)
        axes[1].set_title("p=1e-8", size=14)
        axes[1].text(
            1.03,
            0.2,
            "Lognormal Distribution",
            rotation=90,
            color="Black",
            transform=axes[1].transAxes,
            size=14,
        )
        axes[3].text(
            1.03,
            0.2,
            "Gamma Distribution",
            rotation=90,
            color="Black",
            transform=axes[3].transAxes,
            size=14,
        )

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
        (
            sim,
            io,
            maxed,
            inf_io,
            inf_maxed,
            io_kc,
            max_kc,
            inf_io_kc,
            inf_maxed_kc,
        ) = self.data
        f, axes = plt.subplots(
            nrows=3,
            ncols=4,
            sharex=True,
            sharey=True,
            gridspec_kw={
                "wspace": 0.1,
                "hspace": 0.1,
                "height_ratios": [1, 1, 1],
                "width_ratios": [1, 1, 1, 1],
            },
            figsize=(20, 15),
        )
        axes[0, 0].set_xscale("log")
        axes[0, 0].set_yscale("log")
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

            for i, (method, kc) in enumerate(
                zip(
                    [
                        inside_outside,
                        maximized,
                        inferred_inside_outside,
                        inferred_maximized,
                    ],
                    [io_kc, max_kc, inf_io_kc, inf_maxed_kc],
                )
            ):
                self.mutation_accuracy(
                    axes[index, i], true_ages, method, "", kc_distance_1=kc[index]
                )
            axes[index, 3].text(
                3.25,
                0.15,
                "Mutation Rate: " + str(param),
                rotation=90,
                transform=axes[index, 1].transAxes,
                size=20,
            )

        axes[0, 0].set_title("Inside-Outside", size=20, color="Black")
        axes[0, 1].set_title("Maximization", size=20, color="Black")
        axes[0, 2].set_title("Inside-Outside", size=20, color="Black")
        axes[0, 3].set_title("Maximization", size=20, color="Black")

        f.text(0.5, 0.05, "True Time", ha="center", size=25)
        f.text(0.08, 0.5, "Estimated Time", va="center", rotation="vertical", size=25)
        f.text(0.31, 0.92, "tsdate using True Topologies", ha="center", size=25)
        f.text(0.71, 0.92, "tsdate using tsinfer Topologies", ha="center", size=25)

        self.save(self.name)


class NeutralSims(Figure):
    """
    Supplementary Figure 3: Accuracy of tsdate, tsdate + tsinfer, Geva and Relate
    on a neutral coalescent simulation.
    """

    name = "neutral_simulated_mutation_accuracy"
    data_path = "simulated-data"
    filename = [
        "neutral_simulated_mutation_accuracy_mutations",
        "neutral_simulated_mutation_accuracy_kc_distances",
    ]

    def plot(self):
        df = self.data[0]
        kc_distances = self.data[1]
        kc_distances = kc_distances.set_index(kc_distances.columns[0])
        # error_df = self.data[1]
        # anc_error_df = self.data[2]
        f, ax = plt.subplots(
            nrows=2,
            ncols=2,
            sharex=True,
            sharey=True,
            gridspec_kw={"wspace": 0.1, "hspace": 0.1},
            figsize=(12, 12),
        )

        ax[0, 0].set_xscale("log")
        ax[0, 0].set_yscale("log")
        ax[0, 0].set_xlim(1, 2e5)
        ax[0, 0].set_ylim(1, 2e5)

        # for row, (df, kc_distance) in enumerate(zip(no_error_df, kc_distances)):
        # We can only plot comparable mutations, so remove all rows where NaNs exist
        # df = df.drop(columns=["geva"])
        # df = df.replace([0, -np.inf], np.nan)
        # df = df.dropna()
        df = df[df["simulated_ts"] > 0]
        df = df[np.all(df > 0, axis=1)]
        # df = df[df["relate"] > 0]
        # df = df[df["tsdate"] > 0]
        # df = df[df["tsdate_inferred"] > 0]

        # tsdate on true tree
        self.mutation_accuracy(
            ax[0, 0],
            df["simulated_ts"][df["tsdate"] > 0],
            df["tsdate"][df["tsdate"] > 0],
            "tsdate (using true topology)",
            kc_distance_1=np.mean(kc_distances.loc[1]["tsdate"]),
        )

        # tsdate on inferred tree
        self.mutation_accuracy(
            ax[0, 1],
            df["simulated_ts"][df["tsdate_inferred"] > 0],
            df["tsdate_inferred"][df["tsdate_inferred"] > 0],
            "tsinfer + tsdate",
            kc_distance_0=np.mean(kc_distances.loc[0]["tsdate_inferred"]),
            kc_distance_1=np.mean(kc_distances.loc[1]["tsdate_inferred"]),
        )

        df = df[df["relate"] > 0]
        # Relate accuracy
        self.mutation_accuracy(
            ax[1, 1],
            df["simulated_ts"][~np.isnan(df["relate"])],
            df["relate"][~np.isnan(df["relate"])],
            "Relate",
            cmap="Greens",
            kc_distance_0=np.mean(kc_distances.loc[0]["relate"]),
            kc_distance_1=np.mean(kc_distances.loc[1]["relate"]),
        )

        # GEVA accuracy
        self.mutation_accuracy(
            ax[1, 0],
            df["simulated_ts"][~np.isnan(df["geva"])],
            df["geva"][~np.isnan(df["geva"])],
            "GEVA",
            cmap="Reds",
        )

        f.text(0.5, 0.05, "True Time", ha="center", size=25)
        f.text(0.05, 0.5, "Estimated Time", va="center", rotation="vertical", size=25)

        self.save(self.name)


class TsdateChr20Accuracy(Figure):
    """
    Supplementary Figure 6: Evaluating tsdate's accuracy on Simulated Chromosome 20
    """

    name = "tsdate_accuracy_chr20"
    data_path = "simulated-data"
    filename = [
        "tsdate_chr20_accuracy_mutations",
        "tsdate_chr20_accuracy_error_mutations",
        "tsdate_chr20_accuracy_anc_error_mutations",
        "tsdate_chr20_accuracy_kc_distances",
        "tsdate_chr20_accuracy_error_kc_distances",
        "tsdate_chr20_accuracy_anc_error_kc_distances",
    ]

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
            anc_error_kc_distances.columns[0]
        )

        f, axes = plt.subplots(
            ncols=3,
            nrows=5,
            sharex=True,
            sharey=True,
            gridspec_kw={
                "wspace": 0.1,
                "hspace": 0.1,
                "width_ratios": [1, 1, 1],
                "height_ratios": [1, 0.1, 1, 1, 1],
            },
            figsize=(15, 20),
        )
        axes[0, 0].axis("off")
        axes[0, 2].axis("off")
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")
        axes[1, 2].axis("off")

        axes[0, 0].set_xscale("log")
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_xlim(1, 2e5)
        axes[0, 0].set_ylim(1, 2e5)
        x0, x1 = axes[0, 0].get_xlim()
        y0, y1 = axes[0, 0].get_ylim()
        row_labels = [
            "tsdate",
            "",
            "tsinfer + tsdate",
            "tsinfer with mismatch \n+ tsdate",
            "iteration tsinfer \n+ tsdate",
        ]
        for (i, name), j in zip(enumerate(row_labels), [1, 2, 2, 2, 2]):
            axes[i, j].set_ylabel(name, rotation=90, size=20)
            axes[i, j].yaxis.set_label_position("right")

        sim = df["simulated_ts"]
        methods = ["tsdate_inferred", "tsdate_mismatch_inferred", "tsdate_iterate_frommismatch"]
        comparable_sites = np.logical_and(sim > 0, df["tsdate"] > 0)
        self.mutation_accuracy(
            axes[0, 1],
            sim[comparable_sites],
            df["tsdate"][comparable_sites],
            "",
            kc_distance_1=np.mean(kc_distances.loc[1]["tsdate"]),
        )
        for col, prefix, (mut_df, kc_df) in zip(
            range(3),
            ["", "error_", "anc_error_"],
            [
                (df, kc_distances),
                (error_df, error_kc_distances),
                (anc_error_df, anc_error_kc_distances),
            ],
        ):
            mut_df = mut_df.dropna(axis=1, how="all")
            mut_df = mut_df[np.all(mut_df > 0, axis=1)]
            for row, method, cmap in zip(
                [2, 3, 4], methods, ["Blues", "Blues", "Blues"]
            ):
                method = prefix + method
                result = mut_df[method]
                #comparable_sites = np.logical_and(sim > 0, result > 0)
                comparable_sites = np.logical_and(mut_df["simulated_ts"] > 0, result > 0)
                #cur_true_ages = sim[comparable_sites]
                cur_true_ages = mut_df["simulated_ts"][comparable_sites]
                cur_results = result[comparable_sites]
                self.mutation_accuracy(
                    axes[row, col],
                    cur_true_ages,
                    cur_results,
                    "",
                    cmap=cmap,
                    kc_distance_0=np.mean(kc_df.loc[0][method]),
                    kc_distance_1=np.mean(kc_df.loc[1][method]),
                )
        axes[0, 1].set_title("tsdate using Simulated Topology")
        axes[2, 0].set_title("No Error")
        axes[2, 1].set_title("Empirical Error")
        axes[2, 2].set_title("Empirical Error + 1% Ancestral State Error")
        f.text(0.5, 0.05, "True Time", ha="center", size=25)
        f.text(0.08, 0.4, "Estimated Time", va="center", rotation="vertical", size=25)

        self.save(self.name)


class Chr20Sims(Figure):
    """
    Extended Data Figure 4: Evaluating tsdate, Relate, and GEVA accuracy on Simulated
    Chromosome 20 snippets
    """

    name = "chr20_sims"
    data_path = "simulated-data"
    filename = [
        "chr20_sims_mutations",
        "chr20_sims_error_mutations",
        "chr20_sims_anc_error_mutations",
        "chr20_sims_kc_distances",
        "chr20_sims_error_kc_distances",
        "chr20_sims_anc_error_kc_distances",
    ]
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
            anc_error_kc_distances.columns[0]
        )

        f, axes = plt.subplots(
            ncols=3,
            nrows=5,
            sharex=True,
            sharey=True,
            gridspec_kw={
                "wspace": 0.1,
                "hspace": 0.1,
                "width_ratios": [1, 1, 1],
                "height_ratios": [1, 0.1, 1, 1, 1],
            },
            figsize=(15, 20),
        )
        axes[0, 0].axis("off")
        axes[0, 2].axis("off")
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")
        axes[1, 2].axis("off")

        axes[0, 0].set_xscale("log")
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_xlim(1, 2e5)
        axes[0, 0].set_ylim(1, 2e5)
        x0, x1 = axes[0, 0].get_xlim()
        y0, y1 = axes[0, 0].get_ylim()
        row_labels = [
            "tsdate",
            "",
            "mismatch tsinfer + tsdate \n(iteration)",
            "Relate",
            "GEVA",
        ]
        for (i, name), j, color in zip(
            enumerate(row_labels),
            [1, 2, 2, 2, 2],
            [
                constants.colors["tsdate"],
                "",
                constants.colors["tsdate"],
                constants.colors["relate"],
                constants.colors["geva"],
            ],
        ):
            axes[i, j].set_ylabel(name, rotation=90, color=color, size=20)
            axes[i, j].yaxis.set_label_position("right")

        sim = df["simulated_ts"]
        methods = ["tsdate_iterate", "relate_iterate", "geva"]
        comparable_sites = np.logical_and(sim > 0, df["tsdate"] > 0)
        df = df[np.all(df > 0, axis=1)]
        self.mutation_accuracy(
            axes[0, 1],
            df["simulated_ts"],
            df["tsdate"],
            # sim[comparable_sites],
            # df["tsdate"][comparable_sites],
            "",
            kc_distance_1=np.mean(kc_distances.loc[1]["tsdate"]),
        )
        for col, (mut_df, kc_df) in zip(
            range(3),
            [
                (df, kc_distances),
                (error_df, error_kc_distances),
                (anc_error_df, anc_error_kc_distances),
            ],
        ):
            for row, method, cmap in zip(
                [2, 3, 4], methods, ["Blues", "Greens", "Reds"]
            ):
                result = mut_df[method]
                comparable_sites = np.logical_and(sim > 0, result > 0)
                mut_df = mut_df[np.all(mut_df > 0, axis=1)]
                # cur_true_ages = sim[comparable_sites]
                # cur_results = result[comparable_sites]
                cur_true_ages = mut_df["simulated_ts"]
                cur_results = mut_df[method]
                kc_0 = np.mean(kc_df.loc[0][method])
                kc_1 = np.mean(kc_df.loc[1][method])
                if np.isnan(kc_0) or np.isnan(kc_1):
                    self.mutation_accuracy(
                        axes[row, col], cur_true_ages, cur_results, "", cmap=cmap
                    )
                else:
                    self.mutation_accuracy(
                        axes[row, col],
                        cur_true_ages,
                        cur_results,
                        "",
                        cmap=cmap,
                        kc_distance_0=kc_0,
                        kc_distance_1=kc_1,
                    )

        axes[0, 1].set_title("tsdate using Simulated Topology", color="black", size=20)
        axes[2, 0].set_title("No Error", color="black", size=20)
        axes[2, 1].set_title("Empirical Error", color="black", size=20)
        axes[2, 2].set_title(
            "Empirical Error + 1% Ancestral State Error", color="black"
        )
        f.text(0.5, 0.06, "True Time", ha="center", size=25)
        f.text(0.06, 0.4, "Estimated Time", va="center", rotation="vertical", size=25)

        self.save(self.name)


class plot_sample_locations(Figure):
    """
    Figure 4a: Plot the locations of samples
    """

    name = "sample_locations"
    data_path = "data"
    filename = ["hgdp_sgdp_ancients_ancestor_coordinates"]
    delimiter = " "
    header = None

    def plot(self):
        # Remove samples in 1kg
        hgdp_sgdp_ancients = self.main_ts().simplify(
            np.where(~np.isin(
                self.main_ts().tables.nodes.population[self.main_ts().samples()],
                np.arange(54, 80),
            ))[0]
        )
        tgp_hgdp_sgdp_ancestor_locations = self.data[0]

        _ = plt.figure(figsize=(15, 6))
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=41))
        ax.coastlines(linewidth=0.1)
        ax.add_feature(cartopy.feature.LAND, facecolor="lightgray")
        ax.set_global()

        ax.set_extent([-170, 180, -60, 80], crs=ccrs.Geodetic())

        def jitter(array):
            max_min = np.max(array) - np.min(array)
            return array + np.random.randn(len(array)) * (max_min * 0.009)

        unique_hgdp_locations = np.unique(
            tgp_hgdp_sgdp_ancestor_locations[
                np.isin(hgdp_sgdp_ancients.tables.nodes.population, np.arange(0, 54))
            ],
            axis=0,
            return_counts=True,
        )
        unique_sgdp_locations = np.unique(
            tgp_hgdp_sgdp_ancestor_locations[
                np.isin(hgdp_sgdp_ancients.tables.nodes.population, np.arange(54, 184))
            ],
            axis=0,
            return_counts=True,
        )
        unique_ancient_locations = np.unique(
            tgp_hgdp_sgdp_ancestor_locations[
                np.isin(
                    hgdp_sgdp_ancients.tables.nodes.population,
                    np.arange(184, hgdp_sgdp_ancients.num_populations),
                )
            ],
            axis=0,
            return_counts=True,
        )

        ax.scatter(
            unique_hgdp_locations[0][:, 1],
            unique_hgdp_locations[0][:, 0],
            transform=ccrs.PlateCarree(),
            s=unique_hgdp_locations[1] * 2,
            label="HGDP",
            alpha=0.85,
            zorder=3,
        )
        ax.scatter(
            jitter(unique_sgdp_locations[0][:, 1]),
            jitter(unique_sgdp_locations[0][:, 0]),
            transform=ccrs.PlateCarree(),
            s=unique_sgdp_locations[1] * 2,
            marker="s",
            label="SGDP",
            alpha=0.85,
            zorder=3,
        )
        ax.scatter(
            unique_ancient_locations[0][:, 1],
            unique_ancient_locations[0][:, 0],
            transform=ccrs.PlateCarree(),
            s=unique_ancient_locations[1] * 2,
            marker="*",
            label="Ancient",
            alpha=0.85,
            zorder=3,
        )

        lgnd = ax.legend(loc="lower left", fontsize=15)
        lgnd.legendHandles[0]._sizes = [200]
        lgnd.legendHandles[1]._sizes = [200]
        lgnd.legendHandles[2]._sizes = [200]
        self.save(self.name)


class PopulationAncestors(Figure):
    """
    Figure 4b: Plot average position of ancestors of each population
    """

    name = "population_ancestors"
    data_path = "data"
    filename = [
        "avg_pop_ancestral_location_LATS",
        "avg_pop_ancestral_location_LONGS",
        "num_ancestral_lineages",
    ]
    delimiter = ","
    header = None

    def colorline(self, x, y, z, transform, cmap, norm, ax, linewidth=3, alpha=1.0):
        """
        This function is from:
        http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
        http://matplotlib.org/examples/pylab_examples/multicolored_line.html

        Plot a colored line with coordinates x and y
        Optionally specify colors in the array z
        Optionally specify a colormap, a norm function and a line width
        """

        # Default colors equally spaced on [0,1]:
        if z is None:
            z = np.linspace(0.0, 1.0, len(x))

        # Special case if a single number:
        if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
            z = np.array([z])

        z = np.asarray(z)

        segments = self.make_segments(x, y)

        lc = matplotlib.collections.LineCollection(
            segments,
            array=z,
            cmap=cmap,
            norm=norm,
            linewidth=linewidth,
            alpha=alpha,
            transform=transform,
        )

        ax.add_collection(lc)

        return lc

    def make_segments(self, x, y):
        """
        This function is from:
            https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
        http://matplotlib.org/examples/pylab_examples/multicolored_line.html

        Create list of line segments from x and y coordinates, in the correct format
        for LineCollection: an array of the form numlines x (points per line) x 2 (x
        and y) array
        """

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        return segments

    def plot(self):
        lat_list = self.data[0].to_numpy()
        long_list = self.data[1].to_numpy()
        num_ancestral_lineages = self.data[2].to_numpy()
        print(num_ancestral_lineages)
        # Remove samples in 1kg
        hgdp_sgdp_ancients = ts.simplify(
            np.where(
                ~np.isin(ts.tables.nodes.population[ts.samples()], np.arange(54, 80))
            )[0]
        )
        fig = plt.figure(figsize=(16, 10))
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=41))
        ax.coastlines(linewidth=0.1)
        ax.add_feature(cartopy.feature.LAND, facecolor="lightgray")
        ax.set_global()
        ax.set_extent([-170, 180, -40, 90], crs=ccrs.Geodetic())
        time_windows_smaller = np.concatenate(
            [np.array([0]), np.logspace(3.5, 11, num=40, base=2.718)]
        )
        for population in np.arange(0, 55):
            pop_size = np.sum(hgdp_sgdp_ancients.tables.nodes.population == population)
            result_colorline = self.colorline(
                long_list[population],
                lat_list[population],
                np.linspace(0, 1, len(long_list)),
                cmap=plt.get_cmap("plasma_r"),
                norm=plt.Normalize(0.0, 1.0),
                linewidth=0.002 * (num_ancestral_lineages[population] / pop_size),
                transform=ccrs.Geodetic(),
                ax=ax,
            )
            cax = fig.add_axes(
                [ax.get_position().x0, ax.get_position().y0 - 0.1, 0.78, 0.05]
            )
            cbar = fig.colorbar(result_colorline, cax=cax, orientation="horizontal")

            cbar.set_label(label="Time in Years", size=20)
            cbar.ax.set_xticklabels(
                25 * np.round(time_windows_smaller[[0, 4, 8, 12, 16, 20]])
            )
        self.save(self.name)


class WorldDensity(Figure):
    """
    Figure 4c: World map with all ancestors plotted at six timepoints
    """

    name = "world_density"
    data_path = "data"
    filename = ["hgdp_sgdp_ancients_ancestor_coordinates"]
    delimiter = " "
    header = None

    def plot(self):
        locations = self.data[0].to_numpy()
        # Remove samples in 1kg
        ts = self.ts.simplify(
            np.where(
                ~np.isin(self.ts.tables.nodes.population[self.ts.samples()], np.arange(54, 80))
            )[0]
        )
        times = ts.tables.nodes.time[:]
        for time in [100, 1000, 2240, 5600, 11200, 33600]:
            _ = plt.figure(figsize=(15, 6))
            ax = plt.axes(projection=ccrs.Robinson(central_longitude=41))
            ax.coastlines(linewidth=0.1)
            ax.add_feature(cartopy.feature.LAND, facecolor="lightgray")
            ax.set_global()
            ax.set_extent([-170, 180, -40, 90], crs=ccrs.Geodetic())
            edges = np.logical_and(
                times[ts.tables.edges.child] <= time,
                times[ts.tables.edges.parent] > time,
            )
            time_slice_child = ts.tables.edges.child[edges]
            time_slice_parent = ts.tables.edges.parent[edges]
            edge_lengths = times[time_slice_parent] - times[time_slice_child]
            weight_parent = 1 - ((times[time_slice_parent] - time) / edge_lengths)
            weight_child = 1 - ((time - times[time_slice_child]) / edge_lengths)
            lat_arr = np.vstack(
                [locations[time_slice_parent][:, 0], locations[time_slice_child][:, 0]]
            ).T
            long_arr = np.vstack(
                [locations[time_slice_parent][:, 1], locations[time_slice_child][:, 1]]
            ).T
            weights = np.vstack([weight_parent, weight_child]).T
            lats, longs = utility.vectorized_weighted_geographic_center(
                lat_arr, long_arr, weights
            )
            avg_locations = np.array([lats, longs]).T
            xynps = ax.projection.transform_points(
                ccrs.Geodetic(), avg_locations[:, 1], avg_locations[:, 0]
            )
            h = ax.hist2d(
                xynps[:, 0], xynps[:, 1], bins=100, zorder=10, alpha=0.5, cmin=10
            )
            _ = plt.colorbar(h[3], ax=ax, shrink=0.7, format="%.1f", pad=0.02)
            ax.set_global()
            plt.title(str(xynps.shape[0]) + " Ancestral Lineages", fontsize=20)
            self.save(self.name + "_" + str(time))


class AncientDescent(Figure):
    """
    Parent class for all ancient descent figures
    """

    def __init__(self):
        super().__init__()
        self.pop_names = np.loadtxt("data/combined_ts_pop_names.csv", dtype="str")
        self.reference_sets = pickle.load(
            open("data/combined_ts_reference_sets.p", "rb"))
        self.ref_set_map = np.loadtxt(
            "data/combined_ts_reference_set_map.csv").astype(int)
        self.regions = np.loadtxt(
            "data/combined_ts_regions.csv", delimiter=",", dtype="str")

    def plot_total_median_descent(
        self,
        proxy_node_age,
        exclude_pop_names,
        normalised_descendants,
        descent_sum_sample,
        axis_label,
        filename,
    ):
        # Remove populations which should not be plotted
        # For example, don't plot other archaics as scale will be off
        exclude_pop = ~np.in1d(self.pop_names, exclude_pop_names)
        index = np.where(exclude_pop)[0]
        # Determine population level descent from ancients
        vals = np.sum(normalised_descendants, axis=0)[exclude_pop]
        vals = pd.Series(vals, index=index)
        median_descent = {}
        for pop in index:
            median_descent[pop] = descent_sum_sample[self.reference_sets[pop]]
        median_descent = pd.Series(median_descent)
        reference_set_lens = np.array([len(ref_set) for ref_set in self.reference_sets])
        df = pd.DataFrame(
            {
                "descent": vals,
                "regions": self.regions[exclude_pop],
                "populations": self.pop_names[exclude_pop],
                "colors": [region_colors[reg] for reg in self.regions[exclude_pop]],
                "population_size": reference_set_lens[exclude_pop],
                "median_descent": median_descent,
            },
            index=index,
        )
        df = df.sort_values(["regions", "descent"])
        df = df[df["descent"] > 0.00003]
        fig, axes = plt.subplots(
            2, 1, figsize=(55, 10), sharex=True, gridspec_kw={"wspace": 0, "hspace": 0}
        )
        total = 0
        for region in np.unique(df["regions"]):
            x_value = np.arange(np.sum(df["regions"] == region)) + 1 + total
            axes[0].bar(
                x_value,
                df[df["regions"] == region]["descent"],
                color=df[df["regions"] == region]["colors"],
            )
            total += np.sum(df["regions"] == region)

        axes[0].set_xticks([])
        axes[0].set_yticklabels(axes[0].get_yticks(), size=16)
        axes[0].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
        axes[0].set_ylabel(
            "Normalised Descent from \n" + axis_label + "-like Ancestry", size=19
        )
        boxes = axes[1].boxplot(df["median_descent"].to_numpy())
        for color, box in zip(df["colors"], boxes["boxes"]):
            box.set_color(color)
        yticks = axes[1].get_yticks()
        axes[1].set_yticklabels([int(abs(tick)) for tick in yticks], size=16)

        major_tick_label = np.unique(df["regions"], return_index=True)

        axes[1].set_xticks(1 + np.arange(0, df.shape[0]))
        pop_labels = [
            pop + " (" + str(pop_size) + ")"
            for pop, pop_size in zip(df["populations"], df["population_size"])
        ]
        axes[1].set_xticklabels(pop_labels, rotation=90, size=18)

        axes[1].set_xlim(0, df.shape[0] + 1)
        axes[1].set_ylabel(
            "Total Length of " + axis_label + "-\n like Ancestry (Kb)", size=19
        )

        pos = np.concatenate([major_tick_label[1], [df.shape[0]]]) / df.shape[0]
        pos = np.array((pos[1:] + pos[:-1]) / 2) - np.repeat(
            0.01, len(pos) - 1
        )  # - np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for index, cur_pos in enumerate(pos):
            axes[0].text(
                cur_pos,
                0.85,
                major_tick_label[0][index],
                transform=axes[0].transAxes,
                size=24,
            )
        axes[1].xaxis.grid(alpha=0.5)
        self.save(filename)

    def plot_haplotype_linkage(self, df, children, descendants, filename):
        cmap = matplotlib.colors.ListedColormap(["white", "black"])
        fig = plt.figure(figsize=(40, 20))
        Y = scipy.cluster.hierarchy.linkage(df, method="average")
        Z2 = scipy.cluster.hierarchy.dendrogram(Y, orientation="left", no_plot=True)
        idx1 = np.array(Z2["leaves"][:])
        region_matrix = fig.add_axes([0.0, 0.1, 0.04, 0.6])

        num_rows = len(idx1)
        height = [1 / num_rows for descent in range(num_rows)]
        errorboxes = []

        facecolors = [
            region_colors[region]
            for region in self.regions[self.ref_set_map[descendants[idx1]]]
        ]

        for x, y, xe, ye in zip(
            np.repeat(0, num_rows),
            list(reversed(np.arange(0, 1, 1 / num_rows))),
            np.repeat(1, num_rows),
            height,
        ):
            rect = matplotlib.patches.Rectangle((x, y), xe, ye)
            errorboxes.append(rect)

        region_matrix.add_collection(
            matplotlib.collections.PatchCollection(errorboxes, facecolor=facecolors)
        )
        region_matrix.set_xticklabels([])
        region_matrix.set_yticklabels([])
        haplo_matrix_1 = fig.add_axes([0.04, 0.1, 0.45, 0.6])
        # D = children[descendants][idx1]
        D = children[idx1]
        _ = haplo_matrix_1.imshow(
            D[:, :25000], aspect="auto", origin="upper", cmap=cmap
        )
        haplo_matrix_1.set_xticks(np.arange(0, 25000, 5000))
        haplo_matrix_1.set_xticklabels(np.arange(0, 25000, 5000) / 1000, size=18)
        haplo_matrix_1.set_yticklabels([])
        haplo_matrix_1.grid({"color": "lightgray"})
        haplo_matrix_2 = fig.add_axes([0.51, 0.1, 0.45, 0.6])
        _ = haplo_matrix_2.imshow(
            D[:, 30000:], aspect="auto", origin="upper", cmap=cmap
        )
        haplo_matrix_2.set_xticks(np.concatenate([np.arange(0, 35000, 5000), [34444]]))
        haplo_matrix_2.set_xticklabels(
            np.concatenate([np.arange(30000, 65000, 5000) / 1000, [64]]), size=18
        )

        haplo_matrix_2.yaxis.set_label_position("right")
        haplo_matrix_2.yaxis.tick_right()
        haplo_matrix_2.grid({"color": "lightgray"})
        fig.text(0.5, 0.06, "Chromosome 20 Position (Mb)", ha="center", size=30)
        fig.text(
            0.99,
            0.4,
            "Descendant Chromosomes",
            va="center",
            rotation="vertical",
            size=30,
        )
        self.save(filename)

    def plot(self):
        descent_arr = np.genfromtxt(
            "data/combined_ts_" + self.plotname + "_descent_arr.csv", delimiter=","
        )
        descendants = self.data[0].to_numpy().ravel().astype(int)
        corrcoef_df = self.data[1]
        sample_desc_sum = self.data[2].T.to_numpy()[0]
        genomic_descent = self.data[3]
        genomic_descent = genomic_descent.set_index(genomic_descent.columns[0])
        genomic_descent.columns = genomic_descent.iloc[0]
        genomic_descent = genomic_descent[1:]
        genomic_descent.columns.name = "Population ID"
        genomic_descent = genomic_descent[
            genomic_descent.index == self.plotname.capitalize()
        ]
        corrcoef_df = corrcoef_df[1:]
        corrcoef_df = corrcoef_df.set_index(corrcoef_df.columns[0])
        self.plot_total_median_descent(
            self.proxy_time,
            self.exclude_pop_names,
            genomic_descent,
            sample_desc_sum,
            self.plotname.capitalize(),
            self.plotname + "_median_descent",
        )
        self.plot_haplotype_linkage(
            corrcoef_df, descent_arr, descendants, self.plotname + "_haplotypes"
        )


class AfanasievoDescent(AncientDescent):
    """
    Find Descendants of the Afanasievo Sons
    """

    name = "afanasievo_descent"
    data_path = "data"
    filename = [
        "combined_ts_afanasievo_descendants",
        "combined_ts_afanasievo_corrcoef_df",
        "combined_ts_afanasievo_sample_desc_sum",
        "combined_ts_ancient_descendants",
    ]
    header = None
    plotname = "afanasievo"
    proxy_time = 164.01
    exclude_pop_names = ["Afanasievo"]

class VindijaDescent(AncientDescent):
    """
    Find Descendants of the Vindija Neanderthal
    """

    name = "vindija_descent"
    data_path = "data"
    filename = [
        "combined_ts_vindija_descendants",
        "combined_ts_vindija_corrcoef_df",
        "combined_ts_vindija_sample_desc_sum",
        "combined_ts_ancient_descendants",
    ]
    header = None
    plotname = "vindija"
    proxy_time = 2000.01
    exclude_pop_names = ["Vindija"]


class DenisovanDescent(AncientDescent):
    """
    Find Descendants of the Denisovan
    """

    name = "denisovan_descent"
    data_path = "data"
    filename = [
        "combined_ts_denisovan_descendants",
        "combined_ts_denisovan_corrcoef_df",
        "combined_ts_denisovan_sample_desc_sum",
        "combined_ts_ancient_descendants",
    ]
    header = None
    plotname = "denisovan"
    proxy_time = 2556.01
    exclude_pop_names = ["Denisovan"]


class ChagyrskayaDescent(AncientDescent):
    """
    Find Descendants of the Chagyrskaya
    """

    name = "chagyrskaya_descent"
    data_path = "data"
    filename = [
        "combined_ts_chagyrskaya_descendants",
        "combined_ts_chagyrskaya_corrcoef_df",
        "combined_ts_chagyrskaya_sample_desc_sum",
        "combined_ts_ancient_descendants",
    ]
    header = None
    plotname = "chagyrskaya"
    proxy_time = 3200.01
    exclude_pop_names = ["Chagyrskaya", "Vindija", "Denisovan"]


class AltaiDescent(AncientDescent):
    """
    Find Descendants of the Altai
    """

    name = "altai_descent"
    data_path = "data"
    filename = [
        "combined_ts_altai_descendants",
        "combined_ts_altai_corrcoef_df",
        "combined_ts_altai_sample_desc_sum",
        "combined_ts_ancient_descendants",
    ]
    header = None
    plotname = "altai"
    proxy_time = 4400.01
    exclude_pop_names = ["Altai", "Chagyrskaya", "Vindija", "Denisovan"]


class SiteLinkageAndQuality(Figure):
    """
    Plot proportion of sites with low quality or linkage as a function of the number
    of mutations at those sites.
    """

    name = "ld_quality_by_mutations"

    def gen_log_space(self, limit, n):
        result = [1]
        ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
        while len(result) < n:
            next_value = result[-1] * ratio
            if next_value - result[-1] >= 1:
                # desired state - next_value will be a different integer
                result.append(next_value)
            else:
                # problem! same integer. we need to find next_value by artificially incrementing previous value
                result.append(result[-1] + 1)
                # recalculate the ratio so that the remaining values will scale correctly
                ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
        # round, re-adjust to 0 indexing
        return np.array(list(map(lambda x: round(x) - 1, result)), dtype=np.uint64)

    def plot(self):

        client = dask.distributed.Client(
            dashboard_address="localhost:22222", processes=False,
        )

        haploid = ts.genotype_matrix()
        haploid = da.from_array(haploid, chunks=(10000, haploid.shape[1]))
        # Convert to bi-allelic
        haploid[haploid > 1] = 1

        # Calculate AF
        alt_count = (haploid == 1).sum(axis=1)
        af = (alt_count / haploid.shape[1]).astype(np.float32)
        af = af.compute()

        # Filter sites by AF
        sites_to_keep = np.logical_and(af >= 0.01, af <= 0.99)
        f_haploid_gt = haploid[sites_to_keep]

        @numba.jit(nopython=True, nogil=True, fastmath=True)
        def ld(site_a, site_b):
            rr, ra, ar, aa = (0, 0, 0, 0)
            for j in range(len(site_a)):
                f = site_a[j]
                c = site_b[j]
                if f == 0:
                    if c == 0:
                        rr += 1
                    elif c == 1:
                        ra += 1
                elif f == 1:
                    if c == 0:
                        ar += 1
                    elif c == 1:
                        aa += 1
            s = rr + ra + ar + aa
            if s > 0:
                rr = rr / s
                ra = ra / s
                ar = ar / s
                aa = aa / s
                D = (rr * aa) - (ra * ar)
                # D_max = min((r0 * a1), (r1 * a0))
                # D_prime = D / D_max
                m = (rr + ra) * (rr + ar) * (aa + ar) * (aa + ra)
                if m > 0:
                    r_squared = (D / math.sqrt(m)) ** 2
                    return r_squared
            return np.nan

        # For a given region sum the LD for each site in a 100-site window around that site
        @numba.guvectorize(
            ["void(int8[:,:], float32[:])"],
            "(variants,samples)->(variants)",
            nopython=True,
        )
        def ld_window_sum(region, ld_out):
            for i in range(len(region)):
                ld_out[i] = 0
            for i in range(len(region) - 50):
                for j in range(50):
                    ld_ij = ld(region[i], region[i + j])
                    ld_out[i] = ld_out[i] + ld_ij
                    ld_out[i + j] = ld_out[i + j] + ld_ij

        window_ld = da.overlap.map_overlap(
            ld_window_sum,
            f_haploid_gt.rechunk((9995, f_haploid_gt.chunks[1])),
            depth=50,
            boundary=np.nan,
            drop_axis=1,
            dtype=np.float32,
        ).compute()

        ld = np.full((len(haploid),), np.nan)
        ld[sites_to_keep] = window_ld

        # From https://www.internationalgenome.org/announcements/genome-accessibility-masks/
        mask_chr20 = SeqIO.index("data/20160622.chr20.mask.fasta", "fasta")["chr20"].seq
        mask = []
        for site in ts.tables.sites:
            mask.append(mask_chr20[int(site.position) - 1])
        mask = np.asarray(mask, dtype="U1")

        muts_per_site = np.unique(ts.tables.mutations.site, return_counts=True)[1]

        masked = mask != "P"
        low_ld = ld < 10
        no_ld = np.isnan(ld)

        total_hist, bin_edges = np.histogram(
            muts_per_site, bins=self.gen_log_space(1000, 50)[1:],
        )
        masked_hist, bins = np.histogram(muts_per_site[masked], bins=bin_edges)
        prop_masked_hist = masked_hist / total_hist
        ld_hist, bins = np.histogram(muts_per_site[low_ld], bins=bin_edges)
        prop_ld_hist = ld_hist / total_hist
        neither_hist, bins = np.histogram(
            muts_per_site[np.logical_and(~masked, ~low_ld)], bins=bin_edges,
        )
        prop_neither_hist = neither_hist / total_hist

        fig, ax = plt.subplots()
        fig.patch.set_facecolor("white")
        ax.plot(
            bin_edges[:-1],
            prop_masked_hist,
            drawstyle="steps",
            label="Low quality",
            color="red",
        )
        ax.plot(
            bin_edges[:-1],
            prop_ld_hist,
            drawstyle="steps",
            label="Low linkage",
            color="orange",
        )
        ax.plot(
            bin_edges[:-1],
            prop_neither_hist,
            drawstyle="steps",
            label="Neither",
            color="green",
        )
        ax.legend(prop={"size": 10})
        ax.set_ylabel("Proportion of sites")
        ax.set_xlabel("Number of mutations at site")
        ax.set_xlim(0, 500)
        fig.savefig("figures/ld_quality_by_mutations.eps")


class AncestryVideo(Figure):
    """
    Geography of all ancestors
    """

    name = "ancestry_video"
    data_path = "data"
    filename = ["hgdp_sgdp_ancients_ancestor_coordinates"]
    delimiter = " "
    header = None

    def mix_colors(self, color1_arr, color2_arr):
        new_color = np.zeros((color1_arr.shape[0], 3))
        new_color[:, 0] = (color1_arr[:, 0] + color2_arr[:, 0]) / 2
        new_color[:, 1] = (color1_arr[:, 1] + color2_arr[:, 1]) / 2
        new_color[:, 2] = (color1_arr[:, 2] + color2_arr[:, 2]) / 2
        return new_color

    def plot(self):
        locations = self.data[0].to_numpy()
        ts_no_tgp = self.ts.simplify(
            np.where(
                ~np.isin(self.ts.tables.nodes.population[self.ts.samples()], np.arange(54, 80))
            )[0]
        )
        tables = self.ts.tables
        times = tables.nodes.time[:]
        reference_sets = []
        population_names = []
        pop_region_map = []
        regions = []
        for pop in ts_no_tgp.populations():
            reference_sets.append(
                np.where(tables.nodes.population == pop.id)[0].astype(np.int32)
            )
            name = json.loads(pop.metadata.decode())["name"]
            population_names.append(name)
            if name in sgdp_region_map:
                region = sgdp_region_map[name]
            elif name in hgdp_region_map:
                region = hgdp_region_map[name]
                if region == "Europe":
                    region = "West Eurasia"
            elif name == "Afanasievo":
                region = "Central Asia/Siberia"
            else:
                region = "Archaics"
            regions.append(region)
            pop_region_map.append(region)
        descendants = ts_no_tgp.mean_descendants(reference_sets)
        regions = set(regions)
        pop_region_map = np.array(pop_region_map)
        regions = np.array(sorted(list(regions)))
        region_ancestors = collections.defaultdict(list)
        all_ancestors = []
        for region in regions:
            region_ancestors[region] = np.where(
                np.any(
                    descendants[:, np.where(pop_region_map == region)[0]] != 0, axis=1
                )
            )[0]
            all_ancestors.append(region_ancestors[region])

        region_unique = collections.defaultdict(list)
        for region, ancestors in region_ancestors.items():
            region_unique[region] = ancestors
            for cur_region, ancestors in region_ancestors.items():
                if cur_region != region:
                    region_unique[region] = region_unique[region][
                        ~np.isin(region_unique[region], ancestors)
                    ]

        ancestor_colors = np.full(
            (ts_no_tgp.num_nodes, 3),
            matplotlib.colors.to_rgb(
                matplotlib.colors.get_named_colors_mapping()["black"]
            ),
        )
        # colors = get_tgp_hgdp_sgdp_region_colors()
        for region, ancestors in region_unique.items():
            if region == "Archaics":
                region_colors[region] = region_colors["Ancients"]
                region = "Ancients"
            ancestor_colors[ancestors] = region_colors[region]

        region_unique = collections.defaultdict(list)
        ancestor_by_region = np.zeros((ts_no_tgp.num_nodes, len(regions)))
        for i, (region, ancestors) in enumerate(region_ancestors.items()):
            region_unique[region] = ancestors
            ancestor_colors[ancestors] = region_colors[region]
            ancestor_by_region[ancestors, i] = 1
            for cur_region, cur_ancestors in region_ancestors.items():
                if cur_region != region:
                    overlap = ancestors[np.isin(ancestors, cur_ancestors)]
                    ancestor_colors[overlap] = self.mix_colors(
                        ancestor_colors[overlap],
                        np.tile(region_colors[cur_region], len(overlap)).reshape(
                            len(overlap), 3
                        ),
                    )
                    ancestor_by_region[ancestors, i] = 1
        ancestor_colors[np.where(ancestor_colors > 1)] = 0

        time_locations = []
        logtime = np.exp(np.geomspace(np.log(100), np.log(np.max(times)), 384))
        lintime = np.linspace(0, logtime[0], 64)[:-1]

        time_intervals_log = np.concatenate([lintime, logtime])
        ancestor_children = []
        for time in time_intervals_log:
            edges = np.logical_and(
                times[ts_no_tgp.tables.edges.child] <= time,
                times[ts_no_tgp.tables.edges.parent] > time,
            )
            time_slice_child = ts_no_tgp.tables.edges.child[edges]
            time_slice_parent = ts_no_tgp.tables.edges.parent[edges]
            ancestor_children.append(time_slice_child)
            edge_lengths = times[time_slice_parent] - times[time_slice_child]
            weight_parent = 1 - ((times[time_slice_parent] - time) / edge_lengths)
            weight_child = 1 - ((time - times[time_slice_child]) / edge_lengths)
            lat_arr = np.vstack(
                [locations[time_slice_parent][:, 0], locations[time_slice_child][:, 0]]
            ).T
            long_arr = np.vstack(
                [locations[time_slice_parent][:, 1], locations[time_slice_child][:, 1]]
            ).T
            weights = np.vstack([weight_parent, weight_child]).T
            lats, longs = utility.vectorized_weighted_geographic_center(
                lat_arr, long_arr, weights
            )
            avg_locations = np.array([lats, longs]).T
            time_locations.append(avg_locations)

        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=41))
        ax.set_global()
        ax.coastlines(linewidth=0.1)
        ax.add_feature(cartopy.feature.LAND, facecolor="lightgray")

        xynps = ax.projection.transform_points(
            ccrs.Geodetic(), time_locations[0][:, 1], time_locations[0][:, 0]
        )
        scat = ax.scatter(
            xynps[:, 0],
            xynps[:, 1],
            s=0.05,
            alpha=0.1,
            c=ancestor_colors[ancestor_children[0]][:],
        )

        def animate(i):
            #############
            # Code to plot individual ancestors
            xynps = ax.projection.transform_points(
                ccrs.Geodetic(), time_locations[i][:, 1], time_locations[i][:, 0]
            )
            scat.set_offsets(np.c_[xynps[:, 0], xynps[:, 1]])
            scat.set_color(c=ancestor_colors[ancestor_children[i]][:])
            #############
            plt.title(str(int(np.round(time_intervals_log[i] * 25, -2))) + " Years Ago")

        # prevlayers = [h]
        anim = FuncAnimation(
            fig, animate, interval=90, frames=len(time_intervals_log), repeat=True
        )
        self.save(self.name, animation=anim)


class Timeline(Figure):
    """
    Timeline for Ancestry Map
    """

    name = "timeline"
    data_path = None
    filename = None

    def plot(self):
        times = ts.tables.nodes.time[:]
        fig, ax = plt.subplots(2, 1, figsize=(10, 1))
        fig = plt.figure(figsize=(10, 1))
        spec = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[1, 0.1])
        ax = fig.add_subplot(spec[0])
        dummy = fig.add_subplot(spec[1])
        dummy.axis("off")
        ax.set_yticks([])
        logtime = np.exp(np.geomspace(np.log(100), np.log(np.max(times)), 384))
        lintime = np.linspace(0, logtime[0], 64)[:-1]
        time_intervals_log = np.concatenate([lintime, logtime])
        vline = ax.axvline(time_intervals_log[0], linewidth=11, color="grey")
        text = ax.text(
            time_intervals_log[0] + (time_intervals_log[0] * 0.5),
            0.5,
            str(int(np.round(time_intervals_log[0] * 25, -1)))
            + " Years \n"
            + "("
            + str(int(np.round(time_intervals_log[0], -1)))
            + " Generations)",
        )
        ax.set_xscale("log")
        ax.set_xlim(time_intervals_log[1], time_intervals_log[-1])
        ax.tick_params(labelsize=10)

        def animate(i):
            vline.set_xdata(time_intervals_log[i])
            text.set_text(
                str(int(np.round(time_intervals_log[i] * 25, -2)))
                + " Years \n"
                + "("
                + str(int(np.round(time_intervals_log[i], -2)))
                + " Generations)"
            )
            text.set_position(
                time_intervals_log[i] + (time_intervals_log[i] * 0.5), 0.5
            )

        anim = FuncAnimation(
            fig, animate, interval=90, frames=len(time_intervals_log), repeat=True
        )
        self.save(self.name, animation=anim)


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
