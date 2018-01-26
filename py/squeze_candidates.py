"""
    SQUEzE
    ======

    This file implements the class Candidates, that is used to generate the
    list of quasar candidates, compute the cuts required for the cleaning
    process, and construct the final catalogue
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import tqdm

import numpy as np

import pandas as pd

from scipy import fftpack
from scipy import signal

import matplotlib.pyplot as plt
from matplotlib import gridspec

from squeze_common_functions import save_pkl, load_pkl
from squeze_error import Error
from squeze_defaults import LINES
from squeze_defaults import Z_PRECISION
from squeze_defaults import TRY_LINE
from squeze_spectrum import Spectrum

class Candidates(object):
    """
        Create and manage the candidates catalogue

        CLASS: Candidates
        PURPOSE: Create and manage the candidates catalogue. This include
        creating the list of candidates from a set of Spectrum instances,
        computing cuts to maintain specific level of completeness,
        applying specific cuts to the candidates list, and
        creating a final catalogue.
        """

    def __init__(self, lines_settings=(LINES, TRY_LINE), z_precision=Z_PRECISION,
                 mode="operation", name="SQUEzE_candidates.pkl",
                 weighting_mode="weights"):
        """ Initialize class instance.

            Parameters
            ----------
            lines_settings : (pandas.DataFrame, str) - Default: (LINES, TRY_LINE)
            A tuple with a DataFrame with the information of the lines to check and
            the name of the pkl and log file (without extension) where the
            cuts will be saved. This name must be that of one of the lines present in
            lines.

            z_precision : float - Default: z_precision
            A true candidate is defined as a candidate having an absolute value
            of Delta_z is lower or equal than z_precision. Ignored if mode is
            "operation"

            mode : "training" or "operation" - Default: "operation"
            Running mode. "training" mode assumes that true redshifts are known
            and provide for a series of functions to estimate the cuts to be
            applied to the real sample to increase purity while maintaining
            the completeness. These functions are not available in "operation"
            mode.

            name : string - Default: "SQUEzE_candidates.pkl"
            Name of the candidates sample. The code will save an python-binary
            with the information of the database in a pkl file with this name.
            If load is set to True, then the candidates sample will be loaded
            from this file. Recommended extension is pkl.

            weighting_mode : string - Default: "weights"
            Name of the weighting mode. Can be "weights" if ivar is to be used
            as weights when computing the line ratios, "flags" if ivar is to
            be used as flags when computing the line ratios (pixels with 0 value
            will be ignored, the rest will be averaged without weighting), or
            "none" if weights are to be ignored.

            """
        self.__candidates = None # initialize empty catalogue
        self.__lines = lines_settings[0]
        self.__try_line = lines_settings[1]
        self.__z_precision = z_precision
        self.__mode = mode
        self.__name = name
        self.__weighting_mode = weighting_mode

    def __compute_line_ratio(self, spectrum, index, z_try, smoothed_flux):
        """
            Compute the peak-to-continuum ratio for a specified line.

            Parameters
            ----------
            spectrum : Spectrum
            The spectrum where the ratio is computed.

            index : int
            The index of self.__lines that specifies which line to use.

            z_try : float
            Redshift of the candidate.

            smoothed_flux : np.ndarray
            An array with a smoothed version of the flux

            Returns
            -------
            np.nan if any of the intervals specified in self.__lines
            are outside the scope of the spectrum, or if the sum of the
            values of ivar in one of the regions is zero. Otherwise
            returns the peak-to-continuum ratio.
            """
        wave = spectrum.wave()
        #flux = spectrum.flux()
        ivar = spectrum.ivar()

        # compute intervals
        pix_emiss = np.where((wave >= (1.0+z_try)*self.__lines.ix[index]["start"])
                             & (wave <= (1.0+z_try)*self.__lines.ix[index]["end"]))[0]
        pix_blue = np.where((wave >= (1.0+z_try)*self.__lines.ix[index]["blue_start"])
                            & (wave <= (1.0+z_try)*self.__lines.ix[index]["blue_end"]))[0]
        pix_red = np.where((wave >= (1.0+z_try)*self.__lines.ix[index]["red_start"])
                           & (wave <= (1.0+z_try)*self.__lines.ix[index]["red_end"]))[0]

        # compute ratio
        if (pix_blue.size == 0) or (pix_emiss.size == 0) or (pix_red.size == 0):
            ratio = np.nan
        elif self.__weighting_mode == "none":
            norm_blue = np.average(smoothed_flux[pix_blue])
            norm_emiss = np.average(smoothed_flux[pix_emiss])
            norm_red = np.average(smoothed_flux[pix_red])
            ratio = 2.0*norm_emiss/(norm_blue + norm_red)
        elif self.__weighting_mode == "weights":
            if (ivar[pix_blue].sum() == 0.0
                    or ivar[pix_red].sum() == 0.0
                    or ivar[pix_emiss].sum() == 0.0):
                ratio = np.nan
            else:
                norm_blue = np.average(smoothed_flux[pix_blue], weights=ivar[pix_blue])
                norm_emiss = np.average(smoothed_flux[pix_emiss], weights=ivar[pix_emiss])
                norm_red = np.average(smoothed_flux[pix_red], weights=ivar[pix_red])
                ratio = 2.0*norm_emiss/(norm_blue + norm_red)
        elif self.__weighting_mode == "flags":
            if (ivar[pix_blue].sum() == 0.0
                    or ivar[pix_red].sum() == 0.0
                    or ivar[pix_emiss].sum() == 0.0):
                ratio = np.nan
            else:
                weights = np.ones_like(ivar[pix_blue])
                weights[np.where(ivar[pix_blue] == 0.0)] = 0.0
                norm_blue = np.average(smoothed_flux[pix_blue], weights=weights)

                weights = np.ones_like(ivar[pix_emiss])
                weights[np.where(ivar[pix_emiss] == 0.0)] = 0.0
                norm_emiss = np.average(smoothed_flux[pix_emiss], weights=weights)

                weights = np.ones_like(ivar[pix_red])
                weights[np.where(ivar[pix_red] == 0.0)] = 0.0
                norm_red = np.average(smoothed_flux[pix_red], weights=weights)

                ratio = 2.0*norm_emiss/(norm_blue + norm_red)

        return ratio

    def __find_candidates(self, spectrum):
        """
            Given a Spectrum, locate peaks in the flux. Then assume these peaks
            correspond to the Lyman alpha emission line and compute peak-to-continuum
            ratios for the selected lines.
            For each of the peaks, report a candidate with the redshift estimation,
            the computed peak-to-continuum ratios, and the metadata specified in the
            spectrum.

            Parameters
            ----------

            spectrum : Spectrum
            The spectrum where candidates are looked for.

            Returns
            -------
            A DataFrame with the candidates for the given spectrum.
            """
        if not isinstance(spectrum, Spectrum):
            raise Error("The given spectrum is not of the correct type. It should " +
                        "be an instance of class Spectrum (see squeze_spectrum.py " +
                        "for details).")

        if not (spectrum.flux().size == spectrum.wave().size and
                spectrum.flux().size == spectrum.ivar().size):
            raise Error("The flux, ivar and wave matrixes do not have the same size, but " +
                        "have sizes {flux_size}, ".format(flux_size=spectrum.flux().size,) +
                        "{ivar_size}, and {wave_size}.".format(wave_size=spectrum.wave().size,
                                                               ivar_size=spectrum.ivar().size))

        if self.__mode == "training" and "z_true" not in spectrum.metadata_names():
            raise Error("Mode is set to 'training', but spectrum have does not " +
                        "have the property 'z_true'.")

        if self.__mode == "merge":
            raise Error("Mode 'merge' is not valid for function __find_candidates.")

        # filter small scales fluctuations in the flux
        fft = fftpack.rfft(spectrum.flux()) # compute FFT
        fft[50:] = 0 # filter small scales
        smoothed_flux = fftpack.irfft(fft) # compute filtered inverse FFT

        # find peaks in the smoothed spectrum
        peak_indexs = signal.find_peaks_cwt(smoothed_flux, np.array([50]))

        # find peaks in the spectrum
        candidates = []
        for peak_index in peak_indexs:
            # compute redshift
            z_try = spectrum.wave()[peak_index]/self.__lines["wave"][self.__try_line] - 1.0

            # compute peak ratio for the different lines
            ratios = np.zeros(self.__lines.shape[0], dtype=float)
            for i in range(self.__lines.shape[0]):
                ratios[i] = self.__compute_line_ratio(spectrum, i, z_try, smoothed_flux)

            # add candidate to the list
            candidate_info = spectrum.metadata()
            for ratio in ratios:
                candidate_info.append(ratio)
            candidate_info.append(z_try)
            candidates.append(candidate_info)

        columns_candidates = spectrum.metadata_names()
        for i in range(self.__lines.shape[0]):
            columns_candidates.append("{}_ratio".format(self.__lines.ix[i].name))
        columns_candidates.append("z_try")
        candidates_df = pd.DataFrame(candidates, columns=columns_candidates)

        # add truth table if running in training mode
        if self.__mode == "training":
            candidates_df["Delta_z"] = candidates_df["z_try"] - candidates_df["z_true"]
            def is_correct(row):
                """ Returns True if a candidate is a true quasar and False otherwise.
                    A true candidate is defined as a candidate having an absolute value
                    of Delta_z is lower or equal than self.__z_precision."""
                return bool((row["Delta_z"] <= self.__z_precision)
                            and row["Delta_z"] >= -self.__z_precision)
            if candidates_df.shape[0] > 0:
                candidates_df["is_correct"] = candidates_df.apply(is_correct, axis=1)
            else:
                candidates_df["is_correct"] = pd.Series(dtype=bool)

        return candidates_df

    def __save_candidates(self):
        """ Save the candidates DataFrame. """
        save_pkl(self.__name, self.__candidates)

    def candidates(self):
        """ Access the candidates DataFrame. """
        return self.__candidates

    def lines(self):
        """ Access the lines DataFrame. """
        return self.__lines

    def find_candidates(self, spectra):
        """ Find candidates for a given set of spectra, then integrate them in the
            candidates catalogue and save the new version of the catalogue.

            Parameters
            ----------
            spectra : list of Spectrum
            The spectra in which candidates will be looked for.
            """
        if self.__mode == "merge":
            raise Error("The function find_candidates is not available in " +
                        "merge mode.")

        for spectrum in tqdm.tqdm(spectra):
            # locate candidates in this spectrum
            candidates_df = self.__find_candidates(spectrum)

            # integrate them in the candidates catalogue
            if self.__candidates is None:
                self.__candidates = candidates_df.copy()
            else:
                self.__candidates = self.__candidates.append(candidates_df, ignore_index=True)

        # save the new version of the catalogue
        self.__save_candidates()

    def find_completeness_purity(self, quasars_data_frame, data_frame=None,
                                 quiet=False, get_results=False):
        """
            Given a DataFrame with candidates and another one with the catalogued
            quasars, compute the completeness and the purity. Upon error, return
            np.nan

            Parameters
            ----------
            quasars_data_frame : string
            DataFrame containing the quasar catalogue. The quasars must contain
            the column "specid" to identify the spectrum.

            data_frame : pd.DataFrame - Default: self.__candidates
            DataFrame where the percentile will be computed. Must contain the
            columns "is_correct" and "specid".

            quiet : boolean - Default: False
            If True it will not print the results, otherwise it will print
            the computed stadistics.

            get_results : boolean - Default: False
            If True, return the computed purity, the completeness, and the total number
            of found quasars.
            Otherwise, return None.
            """
        # consistency checks
        if self.__mode != "training":
            raise  Error("The function find_ratio_percentiles is available in the " +
                         "training mode only. Detected mode is {}".format(self.__mode))

        if data_frame is None:
            data_frame = self.__candidates

        if "is_correct" not in data_frame.columns:
            raise Error("find_ratio_percentiles: invalid DataFrame, the column " +
                        "'is_correct' is missing")

        if "specid" not in data_frame.columns:
            raise Error("find_ratio_percentiles: invalid DataFrame, the column " +
                        "'specid' is missing")

        found_quasars = 0
        num_quasars = quasars_data_frame.shape[0]
        for index in tqdm.tqdm(np.arange(num_quasars)):
            specid = quasars_data_frame.ix[quasars_data_frame.index[index]]["specid"]
            if data_frame[(data_frame["specid"] == specid) &
                          (data_frame["is_correct"])].shape[0] > 0:
                found_quasars += 1

        if float(data_frame.shape[0]) > 0.:
            purity = float(data_frame["is_correct"].sum())/float(data_frame.shape[0])
        else:
            purity = np.nan
        if float(num_quasars) > 0.0:
            completeness = float(found_quasars)/float(num_quasars)
        else:
            completeness = np.nan

        if not quiet:
            print "There are {} candidates ".format(data_frame.shape[0]),
            print "for {} catalogued quasars".format(num_quasars)
            print "number of quasars = {}".format(num_quasars)
            print "found quasars = {}".format(found_quasars)
            print "completeness = {:.2%}".format(completeness)
            print "purity = {:.2%}".format(purity)

        if get_results:
            return purity, completeness, found_quasars

    def find_completeness_purity_cuts(self, quasars_data_frame, cuts,
                                      quiet=False, get_results=False):
        """
            Given a DataFrame with the catalogued
            quasars, compute the completeness and the purity. Include cuts in
            magnitude and/or percentile cuts.
            The function first compute a "global analysis" which applies only
            the cuts in magnitude. Then it computes the "cut analysis" which
            applies all the given cuts.
            The completeness on the second step is based on the loss of detections
            on the first step. This means that if we detected 100 quasars in the
            "global analysis", then a completeness of 90% in the "cut analysis"
            means that we detect 90 quasars after the cuts (irrespective of the
            total number of quasars in the sample). To compute the overall
            completeness after the cuts, multiply the completeness of the two steps.

            Parameters
            ----------
            quasars_data_frame : pd.DataFrame
            DataFrame containing the quasar catalogue. The quasars must contain
            the column "specid" to identify the spectrum.

            cuts : list
            A list with tuples (name, value, type). Type can be 'sample-high-cut',
            "sample-low-cut", "percentile", or "min_ratio".
            "sample-high-cut" cuts everything with a value higher or equal than the provided value.
            "sample-low-cut" cuts everything with a value lower than the provided value.
            "percentile" cuts everything with a value lower than the provided percentile.
            "min_ratio" cuts everything with a value lower than the provided value, and should
            only be used with cuts in line ratios.

            quiet : boolean - Default: False
            If True it will not print the results, otherwise it will print
            the computed stadistics.

            get_results : boolean - Default: False
            If True, return a dictionary with the recovered statistics
            num_quasars, num_candidates)
            Otherwise, return None.

            Returns
            -------
            None if get_result is false. Otherwise return a dictionary containing the
            obtained statistics: purity, completeness, overall completeness, number of
            quasars, number of found quasars, and number of candidates
            """
        # consistency checks
        if self.__mode != "training":
            raise  Error("The function find_ratio_percentiles is available in the " +
                         "training mode only. Detected mode is {}".format(self.__mode))

        if "is_correct" not in self.__candidates.columns:
            raise Error("find_ratio_percentiles: invalid DataFrame, the column " +
                        "'is_correct' is missing")

        if "specid" not in self.__candidates.columns:
            raise Error("find_ratio_percentiles: invalid DataFrame, the column " +
                        "'specid' is missing")

        # nested function to filter absolute cuts in DataFrame
        def filter_absolut_cuts(data_frame):
            """ Filter absolute cuts in dataframe.

                Parameters
                ----------
                data_frame : pd.DataFrame
                The dataframe where filters are applied

                Returns
                -------
                The modified DataFrame
                """
            for (name, value, cut_type) in cuts:
                if name in data_frame.columns:
                    if cut_type == "sample-high-cut":
                        data_frame = data_frame[data_frame[name] < value]
                    elif cut_type == "sample-low-cut":
                        data_frame = data_frame[data_frame[name] >= value]
            return data_frame

        # nested function to filter percentiles in DataFrame
        def filter_percentiles(data_frame):
            """ Filter percentiles in dataframe.

                Parameters
                ----------
                data_frame : pd.DataFrame
                The dataframe where filters are applied

                Returns
                -------
                The modified DataFrame
                """
            data_frame_aux = data_frame.copy()
            for (name, value, cut_type) in cuts:
                if cut_type == "percentile":
                    data_frame = data_frame[
                        data_frame[name] > self.find_percentiles(name, value,
                                                                 quiet=quiet,
                                                                 data_frame=data_frame_aux)]
                if cut_type == "min_ratio":
                    data_frame = data_frame[data_frame[name] > value]
            return data_frame

        # nested function to remove missing quasars
        def remove_missing_quasars(quasars_data_frame):
            """ Remove quasars that are not found in the global analysis.

                Parameters
                ----------
                quasars_data_frame : string
                DataFrame containing the quasar catalogue. The quasars must contain
                the column "specid" to identify the spectrum.

                Returns
                -------
                The modified DataFrame.
                """
            first_found = np.zeros(quasars_data_frame.shape[0], dtype=bool)
            for index in tqdm.tqdm(np.arange(quasars_data_frame.shape[0])):
                specid = quasars_data_frame.ix[quasars_data_frame.index[index]]["specid"]
                if data_frame[(data_frame["specid"] == specid) &
                              (data_frame["is_correct"])].shape[0] > 0:
                    first_found[index] = True
            quasars_data_frame = \
                quasars_data_frame.ix[quasars_data_frame.index[np.where(first_found)]]
            return quasars_data_frame

        # filter magnitudes in quasar catalogue
        quasars_data_frame = filter_absolut_cuts(quasars_data_frame)

        # filter magnitudes in dataframe
        data_frame = self.__candidates
        data_frame = filter_absolut_cuts(data_frame)

        # do global analysis
        number_quasars = quasars_data_frame.shape[0]
        if not quiet:
            print "Global analysis (no cuts applied)"
        if get_results:
            global_completeness = \
                self.find_completeness_purity(quasars_data_frame, quiet=quiet,
                                              data_frame=data_frame, get_results=True)[1]
        else:
            self.find_completeness_purity(quasars_data_frame, quiet=quiet,
                                          data_frame=data_frame, get_results=False)

        if not quiet:
            print "Analysis with cuts (completeness based on the loss of detected quasars)"

        # remove quasars that are not found in the global analysis
        quasars_data_frame = remove_missing_quasars(quasars_data_frame)

        # filter ratios in dataframe
        data_frame = filter_percentiles(data_frame)

        # find purity and completeness
        if get_results:
            purity, completeness, found_quasars = \
                self.find_completeness_purity(quasars_data_frame, data_frame=data_frame,
                                              quiet=quiet, get_results=get_results)
            stats = {"purity" : purity,
                     "completeness" : completeness,
                     "overall completeness" : global_completeness*completeness,
                     "number of quasars" : number_quasars,
                     "number of found quasars" : found_quasars,
                     "number of candidates" : data_frame.shape[0]}
            return stats
        else:
            self.find_completeness_purity(quasars_data_frame, data_frame=data_frame,
                                          quiet=quiet, get_results=get_results)


    def find_percentiles(self, column_name, percentile, data_frame=None, quiet=False):
        """
            Find the nth percentile for a column in a DataFrame. NaNs are ignored.

            Parameters
            ----------
            column_name : string
            Name of the column to get the percentile from. Must be in df.columns.

            percentile : float
            Percentile required. Must be in the interval (0, 100].

            data_frame : pd.DataFrame - Default: self.__candidates
            DataFrame where the percentile will be computed. Must contain the
            column "is_correct"

            quiet : boolean - Default: False
            If True it will not print the results, otherwise it will print a
            line with the found percentile.
            """
        # consistency checks
        if self.__mode != "training":
            raise  Error("The function find_ratio_percentiles is available in the " +
                         "training mode only. Detected mode is {}".format(self.__mode))

        if percentile <= 0.0 or percentile > 100.0:
            raise Error("find_ratio_percentiles: cannot compute {}th percentile".format(percentile))

        if column_name not in data_frame.columns:
            raise Error("find_ratio_percentiles: the column name " +
                        "'{}' is not a valid column name".format(column_name))

        if "is_correct" not in data_frame.columns:
            raise Error("find_ratio_percentiles: invalid DataFrame, the column " +
                        "'is_correct' is missing")

        if data_frame is None:
            data_frame = self.__candidates

        col = np.array(data_frame[data_frame["is_correct"]][column_name].dropna())
        if col.size == 0:
            return np.nan
        srtd = np.argsort(col)
        pos = int(col.size*percentile/100)
        ratio = (col[srtd[pos]] + col[srtd[pos-1]])/2.0

        if not quiet:
            print "{}th percentile in {} is {}".format(percentile, column_name, ratio)
        return ratio

    def load_candidates(self, filename=None):
        """ Load the candidates DataFrame

            Parameters
            ----------
            filename : str - Default: None
            Name of the file from where to load existing candidates.
            If None, then load from self.__name
            """
        if filename is None:
            self.__candidates = load_pkl(self.__name)
        else:
            self.__candidates = load_pkl(filename)

    def merge(self, others_list):
        """
            Merge self.__candidates with another candidates object

            Parameters
            ----------
            other : pd.DataFrame
            The other candidates object to merge
            """
        if self.__mode != "merge":
            raise  Error("The function merge is available in the " +
                         "merge mode only. Detected mode is {}".format(self.__mode))

        for candidates_filename in tqdm.tqdm(others_list):
            # load candidates
            other = load_pkl(candidates_filename)

            # append to candidates list
            self.__candidates = self.__candidates.append(other, ignore_index=True)

        self.__save_candidates()

    def plot_histograms(self, plot_col, normed=True):
        """
            Plot the histogram of the specified column
            In training mode, separate the histograms by distinguishing
            contaminants and non-contaminants

            Parameters
            ----------
            plot_col : str
            Name of the column to plot

            normed : bool - Default: True
            If True, then plot the normalized histograms
            """
        # plot settings
        fontsize = 20
        labelsize = 18
        ticksize = 10
        fig = plt.figure(figsize=(10, 6))
        axes_grid = gridspec.GridSpec(1, 1)
        axes_grid.update(hspace=0.4, wspace=0.0)

        # distinguish from contaminants and non-contaminants i necessary
        if self.__mode == "training":
            contaminants_df = self.__candidates[~self.__candidates["is_correct"]]
            correct_df = self.__candidates[self.__candidates["is_correct"]]

        # plot the histograms
        fig_ax = fig.add_subplot(axes_grid[0])

        # plot the contaminants and non-contaminants separately
        if self.__mode == "training":
            contaminants_df[plot_col].hist(ax=fig_ax, bins=100, grid=False, color='r',
                                           alpha=0.5, normed=normed)
            correct_df[plot_col].hist(ax=fig_ax, bins=100, grid=False, color='b',
                                      alpha=0.5, normed=normed)

        # plot the entire sample
        self.__candidates[plot_col].hist(ax=fig_ax, bins=100, grid=False,
                                         histtype='step', color='k', normed=normed)
        # set axis labels
        fig_ax.set_xlabel(plot_col, fontsize=fontsize)
        fig_ax.set_ylabel("counts", fontsize=fontsize)
        fig_ax.tick_params(axis='both', labelsize=labelsize, pad=0, top=True,
                           right=True, length=ticksize, direction="inout")
        return fig

    def plot_line_ratios_histograms(self, normed=True):
        """
            Plot the histogram of the ratios for the different lines.
            In training mode, separate
            the histograms by distinguishing contaminants and
            non-contaminants

            Parameters
            ----------
            normed : bool - Default: True
            If True, then plot the normalized histograms
            """
        # get the number of plots and the names of the columns
        plot_cols = np.array([item for item in self.__candidates.columns if "ratio" in item])
        num_ratios = plot_cols.size

        # plot settings
        fontsize = 20
        labelsize = 18
        ticksize = 10
        fig = plt.figure(figsize=(10, 6*num_ratios))
        axes = []
        axes_grid = gridspec.GridSpec(num_ratios, 1)
        axes_grid.update(hspace=0.4, wspace=0.0)

        # distinguish from contaminants and non-contaminants i necessary
        if self.__mode == "training":
            contaminants_df = self.__candidates[~self.__candidates["is_correct"]]
            correct_df = self.__candidates[self.__candidates["is_correct"]]

        # plot the histograms
        for index, plot_col in enumerate(plot_cols):
            axes.append(fig.add_subplot(axes_grid[index]))
            # plot the contaminants and non-contaminants separately
            if self.__mode == "training":
                contaminants_df[plot_col].hist(ax=axes[index], bins=100, range=(-1, 4),
                                               grid=False, color='r', alpha=0.5, normed=normed)
                correct_df[plot_col].hist(ax=axes[index], bins=100, range=(-1, 4),
                                          grid=False, color='b', alpha=0.5, normed=normed)

            # plot the entire sample
            self.__candidates[plot_col].hist(ax=axes[index], bins=100, range=(-1, 4),
                                             grid=False, histtype='step', color='k', normed=normed)
            # set axis labels
            axes[index].set_xlabel(plot_col, fontsize=fontsize)
            axes[index].set_ylabel("counts", fontsize=fontsize)
            axes[index].tick_params(axis='both', labelsize=labelsize, pad=0, top=True,
                                    right=True, length=ticksize, direction="inout")
            if normed:
                axes[index].set_ylim(0, 4)

        return fig

    def process_cuts(self, cuts, cuts_filename, stats):
        """ Process cuts to use them in training mode and save the results.

            Parameters
            ----------
            cuts : list
            A list with tuples (name, value, type). Type can be 'magnitude',
            "sample-low-cut", "percentile", or "min_ratio".
            "sample-high-cut" cuts everything with a value higher or equal than the provided value.
            "sample-low-cut" cuts everything with a value lower than the provided value.
            "percentile" cuts everything with a value lower than the provided percentile.
            "min_ratio" cuts everything with a value lower than the provided value, and should
            only be used with cuts in line ratios.

            cuts_filename : str
            The name of the file where the cuts information will be stored. Recommended
            extension is pkl

            stats : dict
            A dictionary containing the obtained statistics: purity, completeness, overall
            completeness, number of quasars, number of found quasars, and number of candidates
            """
        # consistency checks
        if self.__mode != "training":
            raise  Error("The function find_ratio_percentiles is available in the " +
                         "training mode only. Detected mode is {}".format(self.__mode))

        # adapt cuts for operation mode
        cuts_operation = []
        for cut in cuts:
            if cut[2] == "percentile":
                cuts_operation.append((cut[0],
                                       self.find_percentiles(cut[0], cut[1], quiet=True,
                                                             data_frame=self.__candidates),
                                       "min_ratio"))
            else:
                cuts_operation.append(cut)
    
        # save cuts for operation mode
        save_pkl("{}.pkl".format(cuts_filename), cuts_operation)

        # write the results in the log
        save_file = open("{}.log".format(cuts_filename), 'w')
        save_file.write("There are {} ".format(stats.get("number of candidates", np.nan)) +
                        "candidates for {} ".format(stats.get("number of quasars", np.nan)) +
                        "catalogued quasars\n")
        save_file.write("number of quasars = {}\n".format(stats.get("number of quasars", np.nan)))
        save_file.write("found quasars = {}\n".format(stats.get("number of found quasars", np.nan)))
        save_file.write("completeness = {:.2%}\n".format(stats.get("completeness", np.nan)))
        save_file.write("overall completeness = {:.2%}\n".format(stats.get("overall completeness",
                                                                           np.nan)))
        save_file.write("purity = {:.2%}\n".format(stats.get("purity", np.nan)))
        save_file.write("\n")
        save_file.write("Applied cuts\n")
        save_file.write("------------\n")
        for cut in cuts_operation:
            save_file.write("{}\n".format(cut))
        save_file.close()

if __name__ == '__main__':
    pass
