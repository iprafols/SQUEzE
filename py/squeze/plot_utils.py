"""
    SQUEzE
    ======

    This file provides useful functions to plot the performance of SQUEzE
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"

import re
import os

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

from squeze.utils import deserialize, load_json

COLOR_LIST = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'
]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["THIS_DIR"] = THIS_DIR
SQUEZE = THIS_DIR.split("py/squeze")[0]
os.environ["SQUEZE"] = SQUEZE
LINES = deserialize(
    load_json(os.path.expandvars("$SQUEZE/data/default_lines.json")))


def compare_performances_plot(stats_dict,
                              names,
                              labels,
                              control_name,
                              plot_f1=False,
                              add_purity=False,
                              add_completeness=False,
                              sharey=True):
    """ Plot the f1-score as a function of magnitude of multiple runs of SQUEzE
    to compare them.

    Arguments
    ---------
    stats_dict: dictionary
    A dictionary with the statistics as a function of magnitude. Keys are the
    names of the different runs, values are the output of functon
    compute_stats_vs_mag (see stats_utils.py)

    names: list of str or str
    The keys in stats_dict that should be plotted. If a string, only one key is
    passed

    labels: list of str or str
    Labels for the runs. Must have the same ordering as names.

    control_name: str
    The name of the baseline run others are compared to. Must be a key in
    stats_dict.

    plot_f1: bool - Default: False
    If True then plot the overall f1-score along with the comparison. Otherwise
    just plot the comparison

    add_purity: bool - Default: False
    If True, then add a dashed line with the purity values

    add_completeness: bool - Default: False
    If True, then add a dotted line with the completeness values

    sharey: bool - Default: True
    If True, the plots at low-z and high-z will share the y axis

    Return
    ------
    fig: matplotlib.pyplot.figure
    The figure with the plot
    """
    if not isinstance(names, list):
        names = [names]
        labels = [labels]

    if len(names) > len(COLOR_LIST):
        print("Too many items to plot. Either add more colors to the list or "
              "else remove some items to plot")
        return

    # plot options
    if plot_f1:
        figsize = (10, 8)
    else:
        figsize = (10, 5)
    fontsize = 14
    labelsize = 13
    ticksize = 8
    tickwidth = 2
    pad = 6
    ncols = 2
    if plot_f1:
        nrows = 3
        height_ratios = [10, 5, 1]
    else:
        nrows = 2
        height_ratios = [10, 1]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, height_ratios=height_ratios)
    gs.update(wspace=0.25,
              hspace=0.4,
              bottom=0.15,
              left=0.1,
              right=0.95,
              top=0.9)
    if plot_f1:
        ax_lowz_f1 = fig.add_subplot(gs[0, 0])
        if sharey:
            ax_highz_f1 = fig.add_subplot(gs[0, 1], sharey=ax_lowz_f1)
        else:
            ax_highz_f1 = fig.add_subplot(gs[0, 1])
        lns_lowz_f1 = []
        lns_highz_f1 = []
    ax_lowz_diff = fig.add_subplot(gs[-2, 0])
    if sharey:
        ax_highz_diff = fig.add_subplot(gs[-2, 1], sharey=ax_lowz_diff)
    else:
        ax_highz_diff = fig.add_subplot(gs[-2, 1])
    lns_lowz_diff = []
    lns_highz_diff = []
    ax_legend = fig.add_subplot(gs[-1, :])

    for index, (name, label) in enumerate(zip(names, labels)):
        if plot_f1:
            lns_lowz_f1 += ax_lowz_f1.plot(
                stats_dict.get(name).get("mag_cuts"),
                stats_dict.get(name).get("f1_score_vs_mag")[:, 0],
                label=label,
                color=COLOR_LIST[index],
                linestyle="solid")
            lns_highz_f1 += ax_highz_f1.plot(
                stats_dict.get(name).get("mag_cuts"),
                stats_dict.get(name).get("f1_score_vs_mag")[:, 1],
                label=label,
                color=COLOR_LIST[index],
                linestyle="solid")
        control = stats_dict.get(control_name).get("f1_score_vs_mag")
        lns_lowz_diff += ax_lowz_diff.plot(
            stats_dict.get(name).get("mag_cuts"),
            stats_dict.get(name).get("f1_score_vs_mag")[:, 0] - control[:, 0],
            label=label,
            color=COLOR_LIST[index],
            linestyle="solid")
        lns_highz_diff += ax_highz_diff.plot(
            stats_dict.get(name).get("mag_cuts"),
            stats_dict.get(name).get("f1_score_vs_mag")[:, 1] - control[:, 1],
            label=label,
            color=COLOR_LIST[index],
            linestyle="solid")

        if add_purity:
            if plot_f1:
                ax_lowz_f1.plot(stats_dict.get(name).get("mag_cuts"),
                                stats_dict.get(name).get("purity_vs_mag")[:, 0],
                                label=label,
                                color=COLOR_LIST[index],
                                linestyle="dashed",
                                alpha=0.5)
                ax_highz_f1.plot(stats_dict.get(name).get("mag_cuts"),
                                 stats_dict.get(name).get("purity_vs_mag")[:,
                                                                           1],
                                 label=label,
                                 color=COLOR_LIST[index],
                                 linestyle="dashed",
                                 alpha=0.5)
            control = stats_dict.get(control_name).get("purity_vs_mag")
            ax_lowz_diff.plot(stats_dict.get(name).get("mag_cuts"),
                              stats_dict.get(name).get("purity_vs_mag")[:, 0] -
                              control[:, 0],
                              label=label,
                              color=COLOR_LIST[index],
                              linestyle="dashed")
            ax_highz_diff.plot(stats_dict.get(name).get("mag_cuts"),
                               stats_dict.get(name).get("purity_vs_mag")[:, 1] -
                               control[:, 1],
                               label=label,
                               color=COLOR_LIST[index],
                               linestyle="dashed",
                               alpha=0.5)

        if add_completeness:
            if plot_f1:
                ax_lowz_f1.plot(
                    stats_dict.get(name).get("mag_cuts"),
                    stats_dict.get(name).get("completeness_vs_mag")[:, 0],
                    label=label,
                    color=COLOR_LIST[index],
                    linestyle="dotted",
                    alpha=0.5)
                ax_highz_f1.plot(
                    stats_dict.get(name).get("mag_cuts"),
                    stats_dict.get(name).get("completeness_vs_mag")[:, 1],
                    label=label,
                    color=COLOR_LIST[index],
                    linestyle="dotted",
                    alpha=0.5)
            control = stats_dict.get(control_name).get("completeness_vs_mag")
            ax_lowz_diff.plot(
                stats_dict.get(name).get("mag_cuts"),
                stats_dict.get(name).get("completeness_vs_mag")[:, 0] -
                control[:, 0],
                label=label,
                color=COLOR_LIST[index],
                linestyle="dotted",
                alpha=0.5)
            ax_highz_diff.plot(
                stats_dict.get(name).get("mag_cuts"),
                stats_dict.get(name).get("completeness_vs_mag")[:, 1] -
                control[:, 1],
                label=label,
                color=COLOR_LIST[index],
                linestyle="dotted",
                alpha=0.5)

    # axis settings, labels
    xlim = (min(stats_dict.get(names[0]).get("mag_cuts")),
            max(stats_dict.get(names[0]).get("mag_cuts")))
    if plot_f1:
        ax_lowz_f1.set_title(r"$z < 2.1$", fontsize=fontsize)
        ax_lowz_f1.set_ylabel(r"$f_{1}$", fontsize=fontsize)
        ax_lowz_f1.yaxis.set_major_locator(MultipleLocator(0.05))
        ax_lowz_f1.tick_params(labelsize=labelsize,
                               size=ticksize,
                               width=tickwidth,
                               pad=pad,
                               left=True,
                               right=False,
                               labelleft=True,
                               labelright=False)
        ax_lowz_f1.set_xlim(xlim)

        ax_highz_f1.set_title(r"$z \geq 2.1$", fontsize=fontsize)
        ax_highz_f1.yaxis.set_major_locator(MultipleLocator(0.05))
        ax_highz_f1.tick_params(labelsize=labelsize,
                                size=ticksize,
                                width=tickwidth,
                                pad=pad,
                                left=True,
                                right=False,
                                labelleft=True,
                                labelright=False)
        ax_highz_f1.set_xlim(xlim)

    else:
        ax_lowz_diff.set_title(r"$z < 2.1$", fontsize=fontsize)
        ax_highz_diff.set_title(r"$z \geq 2.1$", fontsize=fontsize)

    ax_lowz_diff.set_ylabel(r"$f_{1} - f_{1} ({\rm fid})$", fontsize=fontsize)
    ax_lowz_diff.set_xlabel("r mag cut", fontsize=fontsize)
    ax_lowz_diff.tick_params(labelsize=labelsize,
                             size=ticksize,
                             width=tickwidth,
                             pad=pad,
                             left=True,
                             right=False,
                             labelleft=True,
                             labelright=False)
    ax_lowz_diff.set_xlim(xlim)

    ax_highz_diff.set_xlabel("r mag cut", fontsize=fontsize)
    ax_highz_diff.tick_params(labelsize=labelsize,
                              size=ticksize,
                              width=tickwidth,
                              pad=pad,
                              left=True,
                              right=False,
                              labelleft=True,
                              labelright=False)
    ax_highz_diff.set_xlim(xlim)

    # legend
    labels = [lns.get_label() for lns in lns_highz_diff]
    ax_legend.legend(lns_highz_diff, labels, ncol=3, loc=9, fontsize=fontsize)
    ax_legend.axis('off')


def confusion_line_plots(df,
                         rmag_bins,
                         prob_low=0.0,
                         prob_high=0.0,
                         lines=None,
                         exclude_line_pairs=None,
                         delta_z=0.15):
    """ Make a confusion line plot

    Plot only items with probability above a certain threshold.
    High-z quasars (z>=2.1) can be treated differently from low-z quasars.

    Arguments
    ---------
    df: pd.DataFrame
    The dataframe with the classifications

    rmag_bins: array of float
    Limiting magnitudes to split the plot. len(rmag_bins) -1 plot are
    created

    prob_low: float - Default: 0.0
    Probability threshold for low-z quasars (z < 2.1)

    prob_high: float - Default: 0.0
    Probability threshold for high-z quasars (z >= 2.1)

    lines: pd.DataFrame or None - Default: None
    Dataframe with the confusion lines to plot. It must contain column "WAVE".
    Indexs should be the names of the lines. Ignored if None

    exclude_line_pairs: list of (str, str) or None - Default: None
    List containing confusion lines that are not plotted.

    delta_z: float - Default: 0.15
    Maximum redshift error for correctly classified objects
    """
    if exclude_line_pairs is None:
        exclude_line_pairs = []

    ncols = 2
    if len(rmag_bins) % 2 == 0:
        nrows = int((len(rmag_bins) - 1) // 2) + 1
    else:
        nrows = int((len(rmag_bins) - 1) // 2)

    # plot options
    figsize = (8 * nrows, 8 * ncols)
    fontsize = 16
    labelsize = 14
    ticksize = 8
    tickwidth = 2
    markersize = 20
    markersize2 = 30
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols)
    gs.update(wspace=0., hspace=0.2, bottom=0.15, left=0.1, right=0.95, top=0.9)

    row_index = 0
    col_index = 0
    for rmag_min, rmag_max in zip(rmag_bins[:-1], rmag_bins[1:]):

        aux = df[(df["R_MAG"] > rmag_min) & (df["R_MAG"] <= rmag_max)]
        ax = fig.add_subplot(gs[row_index, col_index])
        ax.set_aspect('equal')

        ax.scatter(aux[(aux["IS_CORRECT"])]["Z_TRUE"],
                   aux[aux["IS_CORRECT"]]["Z_TRY"],
                   c='k',
                   label="correct",
                   zorder=4,
                   s=markersize)
        ax.scatter(
            aux[(aux["CLASS_PERSON"] == 3) &
                (((aux["Z_TRY"] >= 2.1) & (aux["PROB"] > prob_high)) |
                 ((aux["Z_TRY"] < 2.1) & (aux["PROB"] > prob_low)))]["Z_TRUE"],
            aux[(aux["CLASS_PERSON"] == 3) &
                (((aux["Z_TRY"] >= 2.1) & (aux["PROB"] > prob_high)) |
                 ((aux["Z_TRY"] < 2.1) & (aux["PROB"] > prob_low)))]["Z_TRY"],
            label="qso",
            zorder=1,
            marker="v",
            s=markersize)
        ax.scatter(
            aux[(aux["CLASS_PERSON"] == 4) &
                (((aux["Z_TRY"] >= 2.1) & (aux["PROB"] > prob_high)) |
                 ((aux["Z_TRY"] < 2.1) & (aux["PROB"] > prob_low)))]["Z_TRUE"],
            aux[(aux["CLASS_PERSON"] == 4) &
                (((aux["Z_TRY"] >= 2.1) & (aux["PROB"] > prob_high)) |
                 ((aux["Z_TRY"] < 2.1) & (aux["PROB"] > prob_low)))]["Z_TRY"],
            label="galaxy",
            zorder=2,
            marker="^",
            s=markersize)
        ax.scatter(
            aux[(aux["CLASS_PERSON"] == 1) &
                (((aux["Z_TRY"] >= 2.1) & (aux["PROB"] > prob_high)) |
                 ((aux["Z_TRY"] < 2.1) & (aux["PROB"] > prob_low)))]["Z_TRUE"],
            aux[(aux["CLASS_PERSON"] == 1) &
                (((aux["Z_TRY"] >= 2.1) & (aux["PROB"] > prob_high)) |
                 ((aux["Z_TRY"] < 2.1) & (aux["PROB"] > prob_low)))]["Z_TRY"],
            label="star",
            zorder=3,
            marker="s",
            s=markersize2)

        if lines is not None:
            z = np.arange(0, 5, 0.5)
            for line1 in lines.index:
                for line2 in lines.index:
                    if line1 == line2 or (line1, line2) in exclude_line_pairs:
                        continue
                    z_line1_as_line2 = (
                        LINES["WAVE"][line1] / LINES["WAVE"][line2] *
                        (1 + z) - 1)
                    ax.plot(z_line1_as_line2,
                            z,
                            label=f"real: {line1}; assumed: {line2}")

        ax.legend(numpoints=1, fontsize=labelsize, loc='lower right')

        xlim = np.array((0, 4))
        ax.fill_between([0.0, 2.1], [0.0, 0.0], [2.1, 2.1],
                        color="k",
                        alpha=0.1,
                        zorder=0)
        ax.fill_between([2.1, xlim[1]], [2.1, 2.1], [xlim[1], xlim[1]],
                        color="k",
                        alpha=0.1,
                        zorder=0)
        ax.plot(xlim, xlim, "r-")
        ax.fill_between(xlim,
                        xlim + delta_z,
                        xlim - delta_z,
                        color="r",
                        alpha=0.2,
                        zorder=0)
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        ax.set_xlabel(r"$z_{\rm true}$", fontsize=fontsize)
        ax.set_ylabel(r"$z_{\rm try}$", fontsize=fontsize)
        ax.set_title(fr"${rmag_min:.1f} < r \leq {rmag_max:.1f}$",
                     fontsize=fontsize)
        ax.tick_params(labelsize=labelsize, size=ticksize, width=tickwidth)

        col_index += 1
        if col_index == ncols:
            col_index = 0
            row_index += 1


def multiline(x_coordinates,
              y_coordinates,
              color_coordinates,
              ax=None,
              **kwargs):
    """Plot lines with different colorings

    Arguments
    ---------
    x_coordinates: 2d array of float
    Array containing x coordinates for each of the lines

    y_coordinates: 2d array of float
    Array containing y coordinates for each of the lines

    color_coordinates: 1d array of float
    Array containing numbers mapped to colormap

    ax: plt.Axes or None - default: None
    Axes to plot on. If None, then create new axes

    **kwargs
    Keyword arguments passed to LineCollection

    Notes
    -----
    len(x_coordinates) == len(y_coordinates) == len(color_coordinates) is the
    number of line segments

    len(x_coordinates[index]) == len(y_coordinates[index]) is the number of
    points for each line (indexed by i)

    Return
    ------
    line_collection: LineCollection
    LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [
        np.column_stack([x, y]) for x, y in zip(x_coordinates, y_coordinates)
    ]
    line_collection = LineCollection(segments, **kwargs)

    # set coloring of line segments
    line_collection.set_array(np.asarray(color_coordinates))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(line_collection)
    ax.autoscale()

    return line_collection


def plot_peaks(spectra,
               peak_finders,
               labels,
               markers,
               offset=1.0,
               ontop=True,
               plot_lines=True,
               emission_lines=LINES,
               add_legend=False):
    """ Plot the peaks found by one or more peak finder instances
    All peak finders accepted by SQUEzE can be passed

    Arguments
    ---------
    spectra: Spectra
    An instance of Spectra with the spectra to plot

    peak_finder: list
    A list of valid peak finder instances.

    labels: list of str
    Labels of the different peak finders. Must have same length as peak_finder

    markers: list of str
    Matplotlib marker strings for the different peak finders

    offset: float or list of float - Default: 1.0
    Offset of the peak markers. Offset is computed by multiplying the flux
    by the value in offset. If a float, use the same value for all peak
    finders. If a list, must have same length as peak_finder. The peaks of each
    peak finder will be offset using the respective offset

    plot_lines: bool - Default: True
    If True, overplot the position of the emision lines in the spectra of
    quasars and galaxies

    emission_lines: pd.DataFrame - Default: LINES
    Emission lines to plot. Format is a datframe with the field "WAVE" with
    the rest-frame wavelength of the lines.

    add_legend: boolean - Default: False
    If True, then plot the plot legend
    """
    if len(peak_finders) > len(COLOR_LIST):
        print("Too many items to plot. Either add more colors to the list or "
              "else remove some items to plot")
        return
    assert len(peak_finders) == len(labels)
    assert len(peak_finders) == len(markers)
    assert isinstance(offset, float) or len(peak_finders) == len(offset)
    if isinstance(offset, float):
        offset = [offset] * len(peak_finders)

    num_spectra = spectra.size()

    # plot options
    fontsize = 14
    labelsize = 12
    ticksize = 8
    tickwidth = 2
    pad = 6
    ncols = 2
    nrows = num_spectra // 2
    if num_spectra % 2 > 0:
        nrows += 1
    figsize = (13, 5 * nrows)
    fig = plt.figure(figsize=figsize)
    if add_legend:
        gs = fig.add_gridspec(nrows=nrows + 1, ncols=ncols)
    else:
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols)
    gs.update(wspace=0.25,
              hspace=0.4,
              bottom=0.15,
              left=0.1,
              right=0.95,
              top=0.9)
    axes = [
        fig.add_subplot(gs[index_row, index_col])
        for index_row in range(nrows)
        for index_col in range(ncols)
    ]
    if add_legend:
        ax_legend = fig.add_subplot(gs[-1, :])

    lines = []
    for index, (ax, spectrum) in enumerate(zip(axes, spectra.spectra_list)):
        specid = spectrum.metadata_by_key("SPECID")
        rmag = spectrum.metadata_by_key("R_MAG")
        z_true = spectrum.metadata_by_key("Z_TRUE")
        class_person = spectrum.metadata_by_key("CLASS_PERSON")

        if index == 0:
            lines += ax.plot(
                spectrum.wave,
                spectrum.flux,
                color="k",
                linestyle="-",
                label="spectrum"
                )
        ax.errorbar(
            spectrum.wave,
            spectrum.flux,
            yerr=1/np.sqrt(spectrum.ivar),
            color="k",
            linestyle="-")
        ax.set_title(f"SPECID: {specid}, R_MAG: {rmag}, Z_TRUE: {z_true:.2f}",
                     fontsize=labelsize)

        for index_pf, peak_finder in enumerate(peak_finders):
            peaks, _ = peak_finder.find_peaks(spectrum)
            if index == 0:
                if ontop:
                    ymax = np.max(spectrum.flux)
                    lines += ax.plot(spectrum.wave[peaks],
                                     [ymax * offset[index_pf]] *
                                     spectrum.wave[peaks].size,
                                     color=COLOR_LIST[index_pf],
                                     linestyle='',
                                     marker=markers[index_pf],
                                     label=f"{labels[index_pf]} peaks")
                else:
                    lines += ax.plot(spectrum.wave[peaks],
                                     spectrum.flux[peaks] * offset[index_pf],
                                     color=COLOR_LIST[index_pf],
                                     linestyle='',
                                     marker=markers[index_pf],
                                     label=f"{labels[index_pf]} peaks")
            else:
                if ontop:
                    ymax = np.max(spectrum.flux)
                    ax.plot(spectrum.wave[peaks], [ymax * offset[index_pf]] *
                            spectrum.wave[peaks].size,
                            color=COLOR_LIST[index_pf],
                            linestyle='',
                            marker=markers[index_pf],
                            label=f"{labels[index_pf]} peaks")
                else:
                    ax.plot(spectrum.wave[peaks],
                            spectrum.flux[peaks] * offset[index_pf],
                            color=COLOR_LIST[index_pf],
                            linestyle='',
                            marker=markers[index_pf])

        if plot_lines and class_person in [3, 4]:
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            emission_lines_observed_frame = emission_lines["WAVE"].values * (
                1 + z_true)
            w = np.where((xlim[0] < emission_lines_observed_frame) &
                         (emission_lines_observed_frame < xlim[1]))
            ax.vlines(emission_lines_observed_frame[w],
                      ylim[0],
                      ylim[1],
                      colors="k",
                      linestyle='--',
                      alpha=0.5)
            ax.set_ylim(ylim)

    # axis settings, labels
    for ax in axes:
        ax.set_ylabel(r"flux", fontsize=fontsize)
        ax.set_xlabel(r"wavelength [${\rm \AA}$]", fontsize=fontsize)
        ax.tick_params(labelsize=labelsize,
                       size=ticksize,
                       width=tickwidth,
                       pad=pad,
                       left=True,
                       right=False,
                       labelleft=True,
                       labelright=False)

    # legend
    if add_legend:
        labels = [lns.get_label() for lns in lines]
        ax_legend.legend(lines, labels, ncol=3, loc=9, fontsize=fontsize)
        ax_legend.axis('off')


def plot_peakfinder_stats_vs_magnitude(mag_cuts,
                                       significance_cut_vs_mag,
                                       completeness_vs_mag,
                                       num_spectra_vs_mag,
                                       num_spectra_qso_vs_mag,
                                       num_entries_vs_mag,
                                       num_correct_entries_vs_mag,
                                       significance_cut_lim=None,
                                       completeness_lim=None,
                                       title=None):
    """
    Plot Peak Finder statistics as a function of magnitude cuts.

    Statistics plotted are the completeness after peak finder and the number
    of trial redshifts per spectrum (all trial redshifts as solid lines and
    correct trial redshfit as dashed lines).

    Format of the arrays should be the same as the outputs from
    compute_peak_finder_completeness_vs_mag (see stats_utils.py)

    Arguments
    ---------
    mag_cuts: list of float
    The used magnitude cuts

    significance_cut: 1d array of float
    The significance cuts used to compute the completeness for each magnitude cut

    completeness_vs_mag: 2d array of float
    Completeness considering the entries that meet the each significance and magnitude cuts

    num_spectra_vs_mag: 1d array of int
    Number of spectra as for each magnitude cut

    num_entries_vs_mag: 2d array of float
    The number of entries in the dataframe that meet the significance and magnitude cuts

    num_correct_entries_vs_mag: 2d array of float
    The number of entries in the dataframe that meet the significance and magnitude cuts
    and corresponds to quasars with the correct redshift

    significance_cut_lim: (float, float) or None - Default: None
    Significance cut range to show in the plot. If None, use the automatic choice

    completeness_lim: (float, float) or None - Default: None
    Completeness range to show in the plot. If None, use the automatic choice

    title: str or None - Default: None
    If not None, add this as plot title
    """

    fontsize = 18
    labelsize = 14
    ticksize = 8
    tickwitdh = 2
    figsize = (12, 5)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, hspace=0., wspace=0.5, width_ratios=[10, 10, 1])

    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    cmap = "viridis"

    line_collection = multiline(significance_cut_vs_mag,
                                completeness_vs_mag,
                                mag_cuts,
                                ax=ax,
                                cmap=cmap)
    multiline(significance_cut_vs_mag,
              num_entries_vs_mag / num_spectra_vs_mag,
              mag_cuts,
              ax=ax2,
              cmap=cmap)
    multiline(significance_cut_vs_mag,
              num_correct_entries_vs_mag /
              num_spectra_qso_vs_mag,
              mag_cuts,
              ax=ax2,
              cmap=cmap,
              linestyle="dashed")

    ax.tick_params(labelsize=labelsize, size=ticksize, width=tickwitdh)
    ax.set_xlabel("min. peak significance", fontsize=fontsize)
    ax.set_ylabel("max. completeness", fontsize=fontsize)

    ax2.set_ylabel("num. trial redshift/spectrum", fontsize=fontsize)
    ax2.set_xlabel("min. peak significance", fontsize=fontsize)
    ax2.tick_params(labelsize=labelsize, size=ticksize, width=tickwitdh)

    fig.colorbar(line_collection, cax=ax3, shrink=0.8)
    ax3.yaxis.set_label_position('left')
    ax3.set_ylabel("magnitude cut", fontsize=fontsize)
    ax3.tick_params(labelsize=labelsize,
                    size=ticksize,
                    width=tickwitdh,
                    left=True,
                    right=False,
                    labelleft=True,
                    labelright=False)

    if completeness_lim is not None:
        ax.set_ylim(completeness_lim)
    if significance_cut_lim is not None:
        ax.set_xlim(significance_cut_lim)
        ax2.set_xlim(significance_cut_lim)
    if title is not None:
        fig.suptitle(title, fontsize=fontsize)


def redshift_precision_histogram(df, mag_bins, title=None,
                                 bins=np.arange(-2e4, 2e4, 750)):
    """ Plot the redshift precision histogram. Also print a table summarising
    the precision.

    Arguments
    ---------
    df: pd.DataFrame
    The catalogue

    mag_bins: list of float
    List of the magnitude limits in each bins. For example [18, 20, 22]
    has two bins: from 18 to 20 and from 20 to 22.

    bins: array of float - Default: np.arange(-2e4, 2e4, 750)
    These are the histogram bins.

    title: str or None - Default: None
    If not None, then add title as the plot title
    """
    # plot options
    figsize = (5, 5)
    fontsize = 16
    labelsize = 12
    ticksize = 8
    tickwidth = 2
    pad = 6
    ncols = 1
    nrows = 1
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols)
    gs.update(wspace=0., hspace=0.2, bottom=0.15, left=0.1, right=0.95, top=0.9)
    ax = fig.add_subplot(gs[0])

    redsfhit_precision = {
        "mag bin": [],
        r"$\overline{\Delta v}$ [km/s]": [],
        r"$\sigma_{\Delta v}$ [km/s]": [],
        r"100$\sigma_{\rm NMAD}$": [],
        "$N$": [],
    }

    if "DELTA_V" not in df.columns:
        df["DELTA_V"] = df["DELTA_Z"] / (1 + df["Z_TRUE"]) * 3e5

    for rmag_min, rmag_max in zip(mag_bins[:-1], mag_bins[1:]):

        aux = df[(df["R_MAG"] > rmag_min) & (df["R_MAG"] <= rmag_max) &
                 (df["IS_CORRECT"])]

        ax.hist(aux["DELTA_V"],
                bins=bins,
                label=fr"${rmag_min:.1f} < r \leq {rmag_max:.1f}$",
                histtype="step",
                density=True,
        )

        redsfhit_precision["mag bin"].append(
            fr"${rmag_min:.1f} < r \leq {rmag_max:.1f}$")
        redsfhit_precision[r"$\overline{\Delta v}$ [km/s]"].append(
            aux['DELTA_V'].mean())
        redsfhit_precision[r"$\sigma_{\Delta v}$ [km/s]"].append(
            aux['DELTA_V'].std())
        redsfhit_precision[r"100$\sigma_{\rm NMAD}$"].append(
            np.fabs(aux['DELTA_V']).median()/3e5*1.48*100)
        redsfhit_precision["$N$"].append(
            aux.shape[0])

    ax.set_xlabel(r"$\Delta v$ [km/s]", fontsize=fontsize)
    ax.set_ylabel(r"normalized counts", fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.tick_params(labelsize=labelsize, pad=pad, size=ticksize, width=tickwidth)

    ax.legend(numpoints=1, fontsize=labelsize, loc="upper left")

    redsfhit_precision_df = pd.DataFrame(redsfhit_precision)
    latex = redsfhit_precision_df.to_latex(
        index=False,
        escape=False,
        column_format="c" * redsfhit_precision_df.shape[0],
        float_format='{:,.2f}'.format,
    )
    latex = re.sub(r"cline{\d-\d}", r"midrule", latex)
    print(latex)
    return redsfhit_precision_df
