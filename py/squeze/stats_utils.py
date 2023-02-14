"""
    SQUEzE
    ======

    This file provides useful functions to test the performance of SQUEzE
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"

import tqdm.notebook as tqdm
import numpy as np
import pandas as pd

from squeze.utils import quietprint, verboseprint


def compute_peak_finder_completeness(df_candidates,
                                     df_truth,
                                     significance_cut=np.arange(0, 10, 0.1)):
    """Compute completeness after the peak finder step

    Arguments
    ---------
    df_candidates: pd.DataFrame
    The candidates DataFrame

    df_truth: pd.DataFrame
    The quasar catalogue

    significance_cut: array of float - default: np.arange(0, 10, 0.1)
    Completeness will be computed with these significance cuts

    Return
    ------
    num_spectra: int
    Number of spectra in the dataframe

    significance_cut: array of float
    The significance cuts used to compute the completeness

    num_entries: array of float
    The number of entries in the dataframe that meet the significance cut

    num_correct_entries: array of float
    The number of entries in the dataframe that meet the significance cut
    and corresponds to quasars with the correct redshift

    completeness: array of float
    Completeness considering the entries that meet the significance cut
    """
    num_spectra = np.unique(df_truth["SPECID"]).size
    num_entries = np.zeros_like(significance_cut)
    num_correct_entries = np.zeros_like(significance_cut)
    completeness = np.zeros_like(significance_cut)

    for index in range(significance_cut.size):
        aux = df_candidates[(df_candidates["PEAK_SIGNIFICANCE"] >=
                             significance_cut[index])]
        num_entries[index] = aux.shape[0]
        aux2 = aux[(aux["IS_LINE"])]
        num_correct_entries[index] = aux2.shape[0]
        completeness[index] = np.unique(aux2["SPECID"]).size / num_spectra

    return num_spectra, significance_cut, num_entries, num_correct_entries, completeness


def compute_peak_finder_completeness_vs_mag(mag_cuts,
                                            df_candidates,
                                            df_truth,
                                            significance_cut=np.arange(
                                                0, 10, 0.1)):
    """Compute completeness after the peak finder step as a function of magnitude

    Arguments
    ---------
    mag_cuts: list of float
    The list of magnitude cuts to explore

    df_candidates: pd.DataFrame
    The candidates DataFrame

    truth: pd.DataFrame
    The quasar catalogue

    significance_cut: array of float - default: np.arange(0, 10, 0.1)
    Completeness will be computed with these significance cuts

    Return
    ------
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
    """
    significance_cut_vs_mag = np.zeros((mag_cuts.size, significance_cut.size),
                                       dtype=float)
    completeness_vs_mag = np.zeros((mag_cuts.size, significance_cut.size),
                                   dtype=float)
    num_spectra_vs_mag = np.zeros((mag_cuts.size, significance_cut.size),
                                  dtype=int)
    num_entries_vs_mag = np.zeros((mag_cuts.size, significance_cut.size),
                                  dtype=int)
    num_correct_entries_vs_mag = np.zeros(
        (mag_cuts.size, significance_cut.size), dtype=int)

    for index, mag_cut in tqdm.tqdm(enumerate(mag_cuts), total=len(mag_cuts)):
        (num_spectra_vs_mag[index], significance_cut_vs_mag[index],
         num_entries_vs_mag[index], num_correct_entries_vs_mag[index],
         completeness_vs_mag[index]) = compute_peak_finder_completeness(
             df_candidates[df_candidates["R_MAG"] <= mag_cut],
             df_truth[df_truth["R_MAG"] <= mag_cut],
             significance_cut=significance_cut)

    return (mag_cuts, significance_cut_vs_mag, completeness_vs_mag,
            num_spectra_vs_mag, num_entries_vs_mag, num_correct_entries_vs_mag)


def compute_stats(df_candidates, df_truth):
    """ Compute summary statistics.
    Two sets of summary statistics are computed. The primary one (without the *)
    requiring the redshift to be correct in correct classifications. The
    alternative one (with the *) relaxing the redshfit requirement to it being
    correct within the high-z (z>=2.1) and low-z (z<2.1) split.

    Arguments
    ---------
    df_candidates: pd.DataFrame
    A dataframe with the predictions

    df_truth: pd.DataFrame
    A dataframe with the truth

    Return
    ------
    stats: pd.DataFrame
    A dataframe with the summary statistics.
    """
    prob_step = .01
    probs = np.arange(1.0, 0.0, -prob_step)
    num_quasars = float(df_truth.shape[0])
    num_quasars_zge2_1 = float(df_truth[df_truth["Z_TRUE"] >= 2.1].shape[0])
    num_quasars_zlt2_1 = float(df_truth[df_truth["Z_TRUE"] < 2.1].shape[0])

    num_candidates = np.zeros_like(probs, dtype=int)
    num_candidates_zge2_1 = np.zeros_like(probs, dtype=int)
    num_candidates_zlt2_1 = np.zeros_like(probs, dtype=int)
    for index_prob, prob in enumerate(probs):
        num_candidates[index_prob] = df_candidates[
            df_candidates["PROB"] > prob].shape[0]
        num_candidates_zge2_1[index_prob] = df_candidates[
            (df_candidates["PROB"] > prob) &
            (df_candidates["Z_TRY"] >= 2.1)].shape[0]
        num_candidates_zlt2_1[index_prob] = df_candidates[
            (df_candidates["PROB"] > prob) &
            (df_candidates["Z_TRY"] < 2.1)].shape[0]

    found = np.zeros_like(probs, dtype=int)
    found_zge2_1 = np.zeros_like(probs, dtype=int)
    found_zlt2_1 = np.zeros_like(probs, dtype=int)
    found_alt = np.zeros_like(probs, dtype=int)
    found_alt_zge2_1 = np.zeros_like(probs, dtype=int)
    found_alt_zlt2_1 = np.zeros_like(probs, dtype=int)
    for index_prob, prob in enumerate(probs):
        df = df_candidates[(df_candidates["PROB"] >= prob) &
                           (~df_candidates["DUPLICATED"])]

        found[index_prob] = df[(df['IS_CORRECT']) &
                               np.isin(df["CLASS_PERSON"], [3, 30])].shape[0]
        found_zge2_1[index_prob] = df[(df["Z_TRUE"] >= 2.1) &
                                      (df['IS_CORRECT']) & np.isin(
                                          df["CLASS_PERSON"], [3, 30])].shape[0]
        found_zlt2_1[index_prob] = df[(df["Z_TRUE"] < 2.1) &
                                      (df['IS_CORRECT']) & np.isin(
                                          df["CLASS_PERSON"], [3, 30])].shape[0]

        correction1 = df[(df["Z_TRUE"] >= 2.1) & (df["Z_TRY"] < 2.1) &
                         (df['IS_CORRECT'])].shape[0]
        num_candidates_zge2_1[index_prob] += correction1
        num_candidates_zlt2_1[index_prob] -= correction1

        correction2 = df[(df["Z_TRUE"] < 2.1) & (df["Z_TRY"] >= 2.1) &
                         (df['IS_CORRECT'])].shape[0]
        num_candidates_zge2_1[index_prob] -= correction2
        num_candidates_zlt2_1[index_prob] += correction2

        found_alt_zge2_1[index_prob] = df[(df["Z_TRUE"] >= 2.1) &
                                          (df["Z_TRY"] >= 2.1) &
                                          np.isin(df["CLASS_PERSON"],
                                                  [3, 30])].shape[0]
        found_alt_zge2_1[index_prob] += correction1
        found_alt_zlt2_1[index_prob] = df[(df["Z_TRUE"] < 2.1) &
                                          (df["Z_TRY"] < 2.1) &
                                          np.isin(df["CLASS_PERSON"],
                                                  [3, 30])].shape[0]
        found_alt_zlt2_1[index_prob] += correction2
        found_alt[index_prob] = found_alt_zge2_1[index_prob] + found_alt_zlt2_1[
            index_prob]

    purity = found.astype(float) / num_candidates.astype(float)
    completeness = found.astype(float) / num_quasars
    f1_score = 2.0 * purity * completeness / (purity + completeness)

    purity_zge2_1 = found_zge2_1.astype(float) / num_candidates_zge2_1.astype(
        float)
    completeness_zge2_1 = found_zge2_1.astype(float) / num_quasars_zge2_1
    f1_score_zge2_1 = 2.0 * purity_zge2_1 * completeness_zge2_1 / (
        purity_zge2_1 + completeness_zge2_1)

    purity_zlt2_1 = found_zlt2_1.astype(float) / num_candidates_zlt2_1.astype(
        float)
    completeness_zlt2_1 = found_zlt2_1.astype(float) / num_quasars_zlt2_1
    f1_score_zlt2_1 = 2.0 * purity_zlt2_1 * completeness_zlt2_1 / (
        purity_zlt2_1 + completeness_zlt2_1)

    purity_alt = found_alt.astype(float) / num_candidates.astype(float)
    completeness_alt = found_alt.astype(float) / num_quasars
    f1_score_alt = 2.0 * purity_alt * completeness_alt / (purity_alt +
                                                          completeness_alt)

    purity_alt_zge2_1 = found_alt_zge2_1.astype(
        float) / num_candidates_zge2_1.astype(float)
    completeness_alt_zge2_1 = found_alt_zge2_1.astype(
        float) / num_quasars_zge2_1
    f1_score_alt_zge2_1 = 2.0 * purity_alt_zge2_1 * completeness_alt_zge2_1 / (
        purity_alt_zge2_1 + completeness_alt_zge2_1)

    purity_alt_zlt2_1 = found_alt_zlt2_1.astype(
        float) / num_candidates_zlt2_1.astype(float)
    completeness_alt_zlt2_1 = found_alt_zlt2_1.astype(
        float) / num_quasars_zlt2_1
    f1_score_alt_zlt2_1 = 2.0 * purity_alt_zlt2_1 * completeness_alt_zlt2_1 / (
        purity_alt_zlt2_1 + completeness_alt_zlt2_1)

    stats = pd.DataFrame({
        "prob": probs,
        "num quasars": num_quasars,
        "num quasars z>=2.1": num_quasars_zge2_1,
        "num quasars z<2.1": num_quasars_zlt2_1,
        "num candidates": num_candidates,
        "num candidates z>=2.1": num_candidates_zge2_1,
        "num candidates z<2.1": num_candidates_zlt2_1,
        "found": found,
        "missed": num_quasars - found,
        "found z>=2.1": found_zge2_1,
        "missed z>=2.1": num_quasars_zge2_1 - found_zge2_1,
        "found z<2.1": found_zlt2_1,
        "missed z<2.1": num_quasars_zlt2_1 - found_zlt2_1,
        "purity": purity,
        "completeness": completeness,
        "f1 score": f1_score,
        "purity z>=2.1": purity_zge2_1,
        "completeness z>=2.1": completeness_zge2_1,
        "f1 score z>=2.1": f1_score_zge2_1,
        "purity z<2.1": purity_zlt2_1,
        "completeness z<2.1": completeness_zlt2_1,
        "f1 score z<2.1": f1_score_zlt2_1,
        "found*": found_alt,
        "missed*": num_quasars - found_alt,
        "found* z>=2.1": found_alt_zge2_1,
        "missed* z>=2.1": num_quasars_zge2_1 - found_alt_zge2_1,
        "found* z<2.1": found_alt_zlt2_1,
        "missed* z<2.1": num_quasars_zlt2_1 - found_alt_zlt2_1,
        "purity*": purity_alt,
        "completeness*": completeness_alt,
        "f1 score*": f1_score_alt,
        "purity* z>=2.1": purity_alt_zge2_1,
        "completeness* z>=2.1": completeness_alt_zge2_1,
        "f1 score* z>=2.1": f1_score_alt_zge2_1,
        "purity* z<2.1": purity_alt_zlt2_1,
        "completeness* z<2.1": completeness_alt_zlt2_1,
        "f1 score* z<2.1": f1_score_alt_zlt2_1,
    })

    return stats


def compute_stats_vs_mag(mag_cuts, df_candidates, df_truth):
    """ Compute the statistics as a function of magnitude.
    Include all the objects up to the cut magnitude. Discard fainter objects.

    Arguments
    ---------
    mag_cuts: list of float
    The list of magnitude cuts to explore

    df_candidates: pd.DataFrame
    A dataframe with the predictions

    df_truth: pd.DataFrame
    A dataframe with the truth

    Return
    ------
    stats_vs_mag: dict
    The statistics as a function of magnitude
    """
    # Compute statistics vs mag
    prob_vs_mag = np.zeros((mag_cuts.size, 3), dtype=float)
    num_candidates_vs_mag = np.zeros((mag_cuts.size, 3), dtype=int)
    purity_vs_mag = np.zeros((mag_cuts.size, 3), dtype=float)
    completeness_vs_mag = np.zeros((mag_cuts.size, 3), dtype=float)
    f1_score_vs_mag = np.zeros((mag_cuts.size, 3), dtype=float)
    prob_alt_vs_mag = np.zeros((mag_cuts.size, 3), dtype=float)
    num_candidates_alt_vs_mag = np.zeros((mag_cuts.size, 3), dtype=int)
    purity_alt_vs_mag = np.zeros((mag_cuts.size, 3), dtype=float)
    completeness_alt_vs_mag = np.zeros((mag_cuts.size, 3), dtype=float)
    f1_score_alt_vs_mag = np.zeros((mag_cuts.size, 3), dtype=float)

    print("Compute stats as a function of magnitude:")
    for index, mag_cut in enumerate(tqdm.tqdm(mag_cuts)):
        stats = compute_stats(
            df_candidates[~(df_candidates["DUPLICATED"]) &
                          (df_candidates["R_MAG"] <= mag_cut)],
            df_truth[df_truth["R_MAG"] <= mag_cut])

        opt_prob = find_prob(stats, do_print=False, opt_f1score=False)
        prob_vs_mag[index] = opt_prob["prob"].values
        num_candidates_vs_mag[index] = opt_prob["num_candidates"].values
        purity_vs_mag[index] = opt_prob["purity"].values
        completeness_vs_mag[index] = opt_prob["completeness"].values
        f1_score_vs_mag[index] = opt_prob["f1_score"].values
        prob_alt_vs_mag[index] = opt_prob["prob_alt"].values
        num_candidates_alt_vs_mag[index] = opt_prob["num_candidates_alt"].values
        purity_alt_vs_mag[index] = opt_prob["purity_alt"].values
        completeness_alt_vs_mag[index] = opt_prob["completeness_alt"].values
        f1_score_alt_vs_mag[index] = opt_prob["f1_score_alt"].values

    print("Compute stats for the entire sample")
    stats = compute_stats(df_candidates[~(df_candidates["DUPLICATED"])],
                          df_truth)

    print("test info:")
    print(
        "num objects last magnitude bin (data):",
        df_candidates[~(df_candidates["DUPLICATED"]) &
                      (df_candidates["R_MAG"] <= mag_cuts[-1])].shape[0])
    print("num objects all (data):",
          df_candidates[~(df_candidates["DUPLICATED"])].shape[0])
    print("num objects last magnitude bin (truth):",
          df_truth[df_truth["R_MAG"] <= mag_cuts[-1]].shape[0])
    print("num objects all (truth):", df_truth.shape[0])

    print("Done")

    stats_vs_mag = {
        "stats_all": stats,
        "mag_cuts": mag_cuts,
        "prob_vs_mag": prob_vs_mag,
        "num_candidates_vs_mag": num_candidates_vs_mag,
        "purity_vs_mag": purity_vs_mag,
        "completeness_vs_mag": completeness_vs_mag,
        "f1_score_vs_mag": f1_score_vs_mag,
        "prob_alt_vs_mag": prob_alt_vs_mag,
        "num_candidates_alt_vs_mag": num_candidates_alt_vs_mag,
        "purity_alt_vs_mag": purity_alt_vs_mag,
        "completeness_alt_vs_mag": completeness_alt_vs_mag,
        "f1_score_alt_vs_mag": f1_score_alt_vs_mag,
    }

    return stats_vs_mag


def find_prob(stats, do_print=True, opt_f1score=True):
    """ Find the optimal probability choice.

    Arguments
    ---------
    stats: pd.DataFrame
    A dataframe with the summary statistics. Must be output from
    compute_statistics or be equally formatted

    do_print: bool - Default: True
    If True, print the results

    opt_f1score: bool - Defaut: True
    If True, the optimal probability is found by maximizing the f-1 score.
    Otherwise, the optimal probability is found by locating the crossing point
    of the purity and the completeness.

    Return
    ------
    opt_prob: pd.DataFrame
    A dataframe with the optimal probability and the summary statstics for that
    proability. This is computed for different settings.
    """
    if do_print:
        userprint = verboseprint
    else:
        userprint = quietprint

    case = []
    prob = []
    num_candidates = []
    completeness = []
    purity = []
    f1_score = []
    prob_alt = []
    num_candidates_alt = []
    completeness_alt = []
    purity_alt = []
    f1_score_alt = []
    for compare_sign in ["<", ">="]:
        case.append(f"z{compare_sign}2.1")
        if do_print:
            userprint(f"Stats for case z{compare_sign}2.1\n")

        if opt_f1score:
            pos = np.argmax(stats[f"f1 score z{compare_sign}2.1"])
        else:
            diff = np.fabs(stats[f"purity z{compare_sign}2.1"] -
                           stats[f"completeness z{compare_sign}2.1"])
            pos = np.argmin(diff)
        prob.append(stats.iloc[pos]["prob"])
        num_candidates.append(
            stats.iloc[pos][f"num candidates z{compare_sign}2.1"])
        purity.append(stats.iloc[pos][f"purity z{compare_sign}2.1"])
        completeness.append(stats.iloc[pos][f"completeness z{compare_sign}2.1"])
        f1_score.append(stats.iloc[pos][f"f1 score z{compare_sign}2.1"])
        cols = [
            col for col in stats.columns
            if "prob" in col or (compare_sign in col and "*" not in col)
        ]
        if do_print:
            userprint(stats.iloc[pos][cols])
            userprint("\n")

        pos_alt = np.argmax(stats[f"f1 score* z{compare_sign}2.1"])
        prob_alt.append(stats.iloc[pos_alt]["prob"])
        num_candidates_alt.append(
            stats.iloc[pos_alt][f"num candidates z{compare_sign}2.1"])
        purity_alt.append(stats.iloc[pos_alt][f"purity* z{compare_sign}2.1"])
        completeness_alt.append(
            stats.iloc[pos_alt][f"completeness* z{compare_sign}2.1"])
        f1_score_alt.append(
            stats.iloc[pos_alt][f"f1 score* z{compare_sign}2.1"])
        cols = [
            col for col in stats.columns
            if "prob" in col or (compare_sign in col and "*" in col)
        ]
        if do_print:
            userprint(stats.iloc[pos_alt][cols])
            userprint("\n")

    case.append("all z")
    if do_print:
        print("Stats for all objects")

    if opt_f1score:
        pos = np.argmax(stats["f1 score"])
    else:
        diff = np.fabs(stats["purity"] - stats["completeness"])
        pos = np.argmin(diff)
    prob.append(stats.iloc[pos]["prob"])
    num_candidates.append(stats.iloc[pos]["num candidates"])
    purity.append(stats.iloc[pos]["purity"])
    completeness.append(stats.iloc[pos]["completeness"])
    f1_score.append(stats.iloc[pos]["f1 score"])
    cols = [
        col for col in stats.columns
        if "prob" in col or ("2.1" not in col and "*" not in col)
    ]
    if do_print:
        userprint(stats.iloc[pos][cols])
        userprint("\n")

    pos_alt = np.argmax(stats["f1 score*"])
    prob_alt.append(stats.iloc[pos_alt]["prob"])
    num_candidates_alt.append(stats.iloc[pos_alt]["num candidates"])
    purity_alt.append(stats.iloc[pos_alt]["purity*"])
    completeness_alt.append(stats.iloc[pos_alt]["completeness*"])
    f1_score_alt.append(stats.iloc[pos_alt]["f1 score*"])
    cols = [
        col for col in stats.columns
        if "prob" in col or ("2.1" not in col and "*" in col)
    ]
    if do_print:
        userprint(stats.iloc[pos_alt][cols])

    opt_prob = pd.DataFrame({
        "case": case,
        "prob": prob,
        "num_candidates": num_candidates,
        "purity": purity,
        "completeness": completeness,
        "f1_score": f1_score,
        "prob_alt": prob_alt,
        "num_candidates_alt": num_candidates_alt,
        "purity_alt": purity_alt,
        "completeness_alt": completeness_alt,
        "f1_score_alt": f1_score_alt
    })

    return opt_prob
