"""
    SQUEzE
    ======

    This file provides useful functions to plot the performance of SQUEzE
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

COLOR_LIST = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

def compare_performances_plot(stats_dict, names, labels, control_name, plot_f1=False,
                              add_purity=False, add_completeness=False):
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

    Return
    ------
    fig: matplotlib.pyplot.figure
    The figure with the plot
    """
    if not isinstance(names, list):
        names = [names]
        labels = [labels]

    if len(names) > len(color_list):
        print("Too many items to plot. Either add more colors to the list or else remove some items to plot")
        return

    # plot options
    if plot_f1:
        figsize = (10, 8)
    else:
        figsize = (10, 5)
    fontsize = 14
    labelsize= 13
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
    gs.update(wspace=0.25, hspace=0.4, bottom=0.15, left=0.1, right=0.95, top=0.9)
    if plot_f1:
        ax_lowz_f1 = fig.add_subplot(gs[0, 0])
        ax_highz_f1 = fig.add_subplot(gs[0, 1])
        lns_lowz_f1 = []
        lns_highz_f1 = []
    ax_lowz_diff = fig.add_subplot(gs[-2, 0])
    ax_highz_diff = fig.add_subplot(gs[-2, 1])
    lns_lowz_diff = []
    lns_highz_diff = []
    ax_legend = fig.add_subplot(gs[-1,:])

    for index, (name, label) in enumerate(zip(names, labels)):
        if plot_f1:
            lns_lowz_f1 += ax_lowz_f1.plot(
                stats_dict.get(name).get("mag_cuts"),
                stats_dict.get(name).get("f1_score_vs_mag")[:, 0],
                label=label, color=color_list[index], linestyle="solid")
            lns_highz_f1 += ax_highz_f1.plot(
                stats_dict.get(name).get("mag_cuts"),
                stats_dict.get(name).get("f1_score_vs_mag")[:, 1],
                label=label, color=color_list[index], linestyle="solid")
        control = stats_dict.get(control_name).get("f1_score_vs_mag")
        ax_lowz_diff += ax_lowz_diff.plot(
            stats_dict.get(name).get("mag_cuts"),
            stats_dict.get(name).get("f1_score_vs_mag")[:, 0] - control[:, 0],
            label="fiducial", color=color_list[index], linestyle="solid")
        lns_highz_diff += ax_highz_diff.plot(
            stats_dict.get(name).get("mag_cuts"),
            stats_dict.get(name).get("f1_score_vs_mag")[:, 1] - control[:, 1],
            label="fiducial", color=color_list[index], linestyle="solid")

        if add_purity:
            if plot_f1:
                ax_lowz_f1.plot(
                    stats_dict.get(name).get("mag_cuts"),
                    stats_dict.get(name).get("purity_vs_mag")[:, 0],
                    label=label, color=color_list[index], linestyle="dashed",
                    alpha=0.5)
                ax_highz_f1.plot(
                    stats_dict.get(name).get("mag_cuts"),
                    stats_dict.get(name).get("purity_vs_mag")[:, 1],
                    label=label, color=color_list[index], linestyle="dashed",
                    alpha=0.5)
            control = stats_dict.get(control_name).get("purity_vs_mag")
            ax_lowz_diff.plot(
                stats_dict.get(name).get("mag_cuts"),
                stats_dict.get(name).get("purity_vs_mag")[:, 0] - control[:, 0],
                label="fiducial", color=color_list[index],
                linestyle="dashed")
            ax_highz_diff.plot(
                stats_dict.get(name).get("mag_cuts"),
                stats_dict.get(name).get("purity_vs_mag")[:, 1] - control[:, 1],
                label="fiducial", color=color_list[index],
                linestyle="dashed", alpha=0.5)

        if add_completeness:
            if plot_f1:
                ax_lowz_f1.plot(
                    stats_dict.get(name).get("mag_cuts"),
                    stats_dict.get(name).get("completeness_vs_mag")[:, 0],
                    label=label, color=color_list[index], linestyle="dotted",
                    alpha=0.5)
                ax_highz_f1.plot(
                    stats_dict.get(name).get("mag_cuts"),
                    stats_dict.get(name).get("completeness_vs_mag")[:, 1],
                    label=label, color=color_list[index], linestyle="dotted",
                    alpha=0.5)
            control = stats_dict.get(control_name).get("completeness_vs_mag")
            ax_lowz_diff.plot(
                stats_dict.get(name).get("mag_cuts"),
                stats_dict.get(name).get("completeness_vs_mag")[:, 0] - control[:, 0],
                label="fiducial", color=color_list[index],
                linestyle="dotted", alpha=0.5)
            ax_highz_diff.plot(
                stats_dict.get(name).get("mag_cuts"),
                stats_dict.get(name).get("completeness_vs_mag")[:, 1] - control[:, 1],
                label="fiducial", color=color_list[index],
                linestyle="dotted", alpha=0.5)

    # axis settings, labels
    xlim = (min(stats_dict.get(name).get("mag_cuts")),
            max(stats_dict.get(name).get("mag_cuts")))
    if plot_f1:
        ax_lowz_f1.set_title(case[0], fontsize=fontsize)
        ax_lowz_f1.set_ylabel(r"$f_{1}$", fontsize=fontsize)
        ax_lowz_f1.yaxis.set_major_locator(MultipleLocator(0.05))
        ax_lowz_f1.tick_params(
            labelsize=labelsize, size=ticksize, width=tickwidth,
            left=True, right=False, labelleft=True, labelright=False)
        ax_lowz_f1.set_xlim(xlim)

        ax_highz_f1.set_title(case[1], fontsize=fontsize)
        ax_highz_f1.yaxis.set_major_locator(MultipleLocator(0.05))
        ax_highz_f1.tick_params(
            labelsize=labelsize, size=ticksize, width=tickwidth,
            left=True, right=False, labelleft=True, labelright=False)
        ax_highz_f1.set_xlim(xlim)

    ax_lowz_diff.set_ylabel(r"$f_{1} - f_{1} ({\rm fid})$", fontsize=fontsize)
    ax_lowz_diff.set_xlabel("r mag cut", fontsize=fontsize)
    ax_lowz_diff.tick_params(
        labelsize=labelsize, size=ticksize, width=tickwidth,
        left=True, right=False, labelleft=True, labelright=False)
    ax_lowz_diff.set_xlim(xlim)

    ax_highz_diff.set_xlabel("r mag cut", fontsize=fontsize)
    ax_highz_diff.tick_params(
        labelsize=labelsize, size=ticksize, width=tickwidth,
        left=True, right=False, labelleft=True, labelright=False)
    ax_highz_diff.set_xlim(xlim)

    # legend
    labels = [lns.get_label() for lns in lns_highz_diff]
    ax_legend.legend(lns_highz_diff, labels, ncol=3, loc=9, fontsize=fontsize)
    ax_legend.axis('off')


def redshift_precision_histogram(df, mag_bins, z_try="Z_TRY", title=None):
    """ Plot the redshift precision histogram. Also print a table summarising
    the precision.

    Arguments
    ---------
    df: pd.DataFrame
    The catalogue

    mag_bins: list of float
    List of the magnitude limits in each bins. For example [18, 20, 22]
    has two bins: from 18 to 20 and from 20 to 22.

    z_try: str - Default: "Z_TRY"
    Name of the redshift column to test

    title: str or None
    If not None, then add title as the plot title
    """
    # plot options
    figsize = (5, 5)
    fontsize = 16
    labelsize= 12
    ticksize = 8
    tickwidth = 2
    markersize = 4
    markersize2 = 14
    pad = 6
    ncols = 1
    nrows = 1
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols)
    gs.update(wspace=0., hspace=0.2, bottom=0.15, left=0.1, right=0.95, top=0.9)
    ax = fig.add_subplot(gs[0])

    colors = ["k", "r", "b", "g"]
    bins = np.arange(-2e4, 2e4, 750)
    redsfhit_precision = {"mag bin": [],
                          r"$\overline{\Delta v}$ [km/s]": [],
                          r"$\sigma_{\Delta v}$ [km/s]": [],
                         }

    if "DELTA_V" not in df.columns:
        df["DELTA_V"] = df["DELTA_Z"]/(1+df["Z_TRUE"])*3e5

    for rmag_min, rmag_max, color in zip(mag_bins[:-1], mag_bins[1:], colors):

        aux = df[(df["R_MAG"] > rmag_min) & (df["R_MAG"] <= rmag_max) & (df["IS_CORRECT"])]

        ax.hist(aux["DELTA_V"], bins=bins, label=fr"${rmag_min:.1f} < r \leq {rmag_max:.1f}$",
                histtype="step", density=True, color=color)


        redsfhit_precision["mag bin"].append(fr"${rmag_min:.1f} < r \leq {rmag_max:.1f}$")
        redsfhit_precision[r"$\overline{\Delta v}$ [km/s]"].append(aux['DELTA_V'].mean())
        redsfhit_precision[r"$\sigma_{\Delta v}$ [km/s]"].append(aux['DELTA_V'].std())

    ax.set_xlabel(r"$\Delta v$ [km/s]", fontsize=fontsize)
    ax.set_ylabel(r"normalized counts", fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.tick_params(labelsize=labelsize, size=ticksize, width=tickwidth)

    ax.legend(numpoints=1, fontsize=labelsize, loc="upper left")


    redsfhit_precision_df = pd.DataFrame(redsfhit_precision)
    latex = redsfhit_precision_df.to_latex(index=False,
                                           escape=False,
                                           column_format="c"*redsfhit_precision_df.shape[0],
                                           float_format='{:,.2f}'.format,
                                           )
    latex = re.sub(r"cline{\d-\d}", r"midrule", latex)
    print(latex)
    return redsfhit_precision_df
