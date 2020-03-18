"""
    SQUEzE
    ======

    This file implements the class Candidates, that is used to generate the
    list of quasar candidates, trains or applies the model required for the
    cleaning process, and construct the final catalogue
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import numpy as np

import pandas as pd

import astropy.io.fits as fits

import matplotlib.pyplot as plt
from matplotlib import gridspec

from squeze.common_functions import verboseprint
from squeze.common_functions import save_json, load_json
from squeze.common_functions import deserialize
from squeze.error import Error
from squeze.model import Model
from squeze.peak_finder import PeakFinder
from squeze.defaults import CUTS
from squeze.defaults import LINES
from squeze.defaults import TRY_LINES
from squeze.defaults import RANDOM_FOREST_OPTIONS
from squeze.defaults import RANDOM_STATE
from squeze.defaults import Z_PRECISION
from squeze.defaults import PEAKFIND_WIDTH
from squeze.defaults import PEAKFIND_SIG
from squeze.spectrum import Spectrum

class Candidates(object):
    """ Create and manage the candidates catalogue

        CLASS: Candidates
        PURPOSE: Create and manage the candidates catalogue. This include
        creating the list of candidates from a set of Spectrum instances,
        computing cuts to maintain specific level of completeness,
        training or appliying the model to clean the candidates list, and
        creating a final catalogue.
        """
    # pylint: disable=too-many-instance-attributes
    # 12 is reasonable in this case.
    def __init__(self, lines_settings=(LINES, TRY_LINES), z_precision=Z_PRECISION,
                 mode="operation", name="SQUEzE_candidates.json",
                 weighting_mode="weights", peakfind=(PEAKFIND_WIDTH, PEAKFIND_SIG),
                 model=(None, CUTS), model_opt=(RANDOM_FOREST_OPTIONS, RANDOM_STATE)):
        """ Initialize class instance.

            Parameters
            ----------
            lines_settings : (pandas.DataFrame, list) - Default: (LINES, TRY_LINES)
            A tuple with a DataFrame with the information of the lines to compute the
            ratios and the name of the lines to assume for each of the found peaks.
            This names must be included in the DataFrame. This will be overloaded if
            model is not None.

            z_precision : float - Default: z_precision
            A true candidate is defined as a candidate having an absolute value
            of Delta_z is lower or equal than z_precision. Ignored if mode is
            "operation". This will be overloaded if model is not None.

            mode : "training", "test", "operation", or "merge" - Default: "operation"
            Running mode. "training" mode assumes that true redshifts are known
            and provide a series of functions to train the model.

            name : string - Default: "SQUEzE_candidates.csv"
            Name of the candidates sample. The code will save an python-binary
            with the information of the database in a csv file with this name.
            If load is set to True, then the candidates sample will be loaded
            from this file. Recommended extension is csv.

            weighting_mode : string - Default: "weights"
            Name of the weighting mode. Can be "weights" if ivar is to be used
            as weights when computing the line ratios, "flags" if ivar is to
            be used as flags when computing the line ratios (pixels with 0 value
            will be ignored, the rest will be averaged without weighting), or
            "none" if weights are to be ignored.

            model : (Model or None, tuple)  - Default: (None, CUTS)
            First item is the instance of the Model class defined in
            squeze_model or None. In test and operation mode, it is supposed
            to be the quasar model to construct the catalogue. In training mode,
            it is supposed to be None initially, and the model will be trained
            and given as an output of the code.
            Second item are the hard-code cuts. In training mode they will be
            added to the model and trained using contaminants. These cuts will
            fix the probability of some of the candidates to 1. In testing and
            operation mode this will be ignored.

            model_opt : (dict, int) - Defaut: (RANDOM_FOREST_OPTIONS, RANDOM_STATE)
            The first dictionary sets the options to be passed to the random forest
            cosntructor. If high-low split of the training is desired, the
            dictionary must contain the entries "high" and "low", and the
            corresponding values must be dictionaries with the options for each
            of the classifiers. In training mode, they're passed to the model
            instance before training. Otherwise it's ignored.
            """
        self.__mode = mode
        self.__name = name

        self.__candidates = None # initialize empty catalogue

        # main settings
        self.__lines = lines_settings[0]
        self.__try_lines = lines_settings[1]
        self.__z_precision = z_precision
        self.__weighting_mode = weighting_mode

        # options to be passed to the peak finder
        self.__peakfind_width = peakfind[0]
        self.__peakfind_sig = peakfind[1]

        # model
        if model[0] is None:
            self.__model = None
            self.__cuts = model[1]
        else:
            self.__model = model[0]
            self.__load_model_settings()
        self.__model_opt = model_opt

        # initialize peak finder
        self.__peak_finder = PeakFinder(self.__peakfind_width, self.__peakfind_sig)

    def __compute_line_ratio(self, spectrum, index, z_try):
        """ Compute the peak-to-continuum ratio for a specified line.

            Parameters
            ----------
            spectrum : Spectrum
            The spectrum where the ratio is computed.

            index : int
            The index of self.__lines that specifies which line to use.

            z_try : float
            Redshift of the candidate.

            Returns
            -------
            np.nan for both the ratio and its error if any of the intervals
            specified in self.__lines are outside the scope of the spectrum,
            or if the sum of the values of ivar in one of the regions is zero.
            Otherwise returns the peak-to-continuum ratio and its error.
            """
        wave = spectrum.wave()
        flux = spectrum.flux()
        ivar = spectrum.ivar()

        # compute intervals
        pix_peak = np.where((wave >= (1.0+z_try)*self.__lines.ix[index]["start"])
                            & (wave <= (1.0+z_try)*self.__lines.ix[index]["end"]))[0]
        pix_blue = np.where((wave >= (1.0+z_try)*self.__lines.ix[index]["blue_start"])
                            & (wave <= (1.0+z_try)*self.__lines.ix[index]["blue_end"]))[0]
        pix_red = np.where((wave >= (1.0+z_try)*self.__lines.ix[index]["red_start"])
                           & (wave <= (1.0+z_try)*self.__lines.ix[index]["red_end"]))[0]

        # compute peak and continuum values
        compute_ratio = True
        if ((pix_blue.size == 0) or (pix_peak.size == 0) or (pix_red.size == 0)
                or (pix_blue.size < pix_peak.size//2)
                or (pix_red.size < pix_peak.size//2)):
            compute_ratio = False
        else:
            peak = np.average(flux[pix_peak])
            cont_red = np.average(flux[pix_red])
            cont_blue = np.average(flux[pix_blue])
            cont_red_and_blue = cont_red + cont_blue
            if cont_red_and_blue == 0.0:
                compute_ratio = False
            peak_ivar_sum = ivar[pix_peak].sum()
            if peak_ivar_sum == 0.0:
                peak_err_squared = np.nan
            else:
                peak_err_squared = 1.0/peak_ivar_sum
            blue_ivar_sum = ivar[pix_blue].sum()
            red_ivar_sum = ivar[pix_red].sum()
            if blue_ivar_sum == 0.0 or red_ivar_sum == 0.0:
                cont_err_squared = np.nan
            else:
                cont_err_squared = (1.0/blue_ivar_sum +
                                    1.0/red_ivar_sum)/4.0
        # compute ratios
        if compute_ratio:
            ratio = 2.0*peak/cont_red_and_blue
            ratio2 = np.abs((cont_red - cont_blue)/cont_red_and_blue)
            err_ratio = np.sqrt(4.*peak_err_squared + ratio*ratio*cont_err_squared)/np.abs(cont_red_and_blue)
            ratio_sn = (ratio - 1.0)/err_ratio
        else:
            ratio = np.nan
            ratio2 = np.nan
            ratio_sn = np.nan

        return ratio, ratio_sn, ratio2

    def __get_settings(self):
        """ Pack the settings in a dictionary. Return it """
        return {"lines": self.__lines,
                "try_lines": self.__try_lines,
                "z_precision": self.__z_precision,
                "weighting_mode": self.__weighting_mode,
                "peakfind_width": self.__peakfind_width,
                "peakfind_sig": self.__peakfind_sig,
               }

    def __is_correct(self, row):
        """ Returns True if a candidate is a true quasar and False otherwise.
            A true candidate is defined as a candidate having an absolute value
            of Delta_z is lower or equal than self.__z_precision.
            This function should be called using the .apply method of the candidates
            data frame with the option axis=1

            Parameters
            ----------
            row : pd.Series
            A row in the candidates data frame

            Returns
            -------
            True if a candidate is a true quasar and False otherwise
            """
        return bool((row["Delta_z"] <= self.__z_precision)
                    and row["Delta_z"] >= -self.__z_precision
                    and (np.isin(row["class_person"], [3, 30])))

    def __is_correct_redshift(self, row):
        """ Returns True if a candidate has a correct redshift and False otherwise.
            A candidate is assumed to have a correct redshift if it has an absolute
            value of Delta_z is lower or equal than self.__z_precision.
            If the object is a star (class_person = 1), then return False.
            This function should be called using the .apply method of the candidates
            data frame with the option axis=1

            Parameters
            ----------
            row : pd.Series
            A row in the candidates data frame

            Returns
            -------
            True if a candidate is a true quasar and False otherwise
            """
        correct_redshift = False
        if row["class_person"] != 1:
            correct_redshift = bool((row["Delta_z"] <= self.__z_precision)
                                   and (row["Delta_z"] >= -self.__z_precision))
        return correct_redshift

    def __is_line(self, row):
        """ Returns True if a candidate is a quasar line and False otherwise.
            A quasar line is defined as a candidate where its redshift assuming
            it was any of the specified lines is off the true redshift by at
            most self.__z_precision.
            This function should be called using the .apply method of the candidates
            data frame with the option axis=1

            Parameters
            ----------
            row : pd.Series
            A row in the candidates data frame

            Returns
            -------
            Returns True if a candidate is a quasar line and False otherwise.
            """
        is_line = False
        # correct identification
        if row["is_correct"]:
            is_line = True
        # not a quasar
        elif not np.isin(row["class_person"], [3, 30]):
            pass
        # not a peak
        elif row["assumed_line"] == "none":
            pass
        else:
            for line in self.__lines.index:
                if line == row["assumed_line"]:
                    continue
                z_try_line = (self.__lines["wave"][row["assumed_line"]]/
                              self.__lines["wave"][line])*(1 + row["z_try"]) - 1
                if ((z_try_line - row["z_true"] <= self.__z_precision) and
                        (z_try_line - row["z_true"] >= -self.__z_precision)):
                    is_line = True
        return is_line

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

        if self.__mode == "test" and "z_true" not in spectrum.metadata_names():
            raise Error("Mode is set to 'test', but spectrum have does not " +
                        "have the property 'z_true'.")

        if self.__mode == "merge":
            raise Error("Mode 'merge' is not valid for function __find_candidates.")

        # find peaks
        peak_indexs, significances = self.__peak_finder.find_peaks(spectrum)

        # keep peaks in the spectrum
        candidates = []
        # if there are no peaks, include the spectrum with redshift -1
        # assumed_line='none', significance is set to np.nan
        # and all the metrics set to np.nan
        if peak_indexs.size == 0:
            candidate_info = spectrum.metadata()
            z_try = -1.0
            significance = np.nan
            try_line = 'none'
            ratios = np.zeros(self.__lines.shape[0], dtype=float)
            ratios_sn = np.zeros_like(ratios)
            ratios2 = np.zeros_like(ratios)
            for (ratio, ratio_sn, ratio2) in zip(ratios, ratios_sn, ratios2):
                candidate_info.append(np.nan)
                candidate_info.append(np.nan)
                candidate_info.append(np.nan)
            candidate_info.append(z_try)
            candidate_info.append(significance)
            candidate_info.append(try_line)
            candidates.append(candidate_info)
        # if there are peaks, compute the metrics and keep the info
        else:
            for peak_index, significance in zip(peak_indexs, significances):
                for try_line in self.__try_lines:
                    # compute redshift
                    z_try = spectrum.wave()[peak_index]/self.__lines["wave"][try_line] - 1.0
                    if z_try < 0.0:
                        continue

                    # compute peak ratio for the different lines
                    ratios = np.zeros(self.__lines.shape[0], dtype=float)
                    ratios_sn = np.zeros_like(ratios)
                    ratios2 = np.zeros_like(ratios)
                    for i in range(self.__lines.shape[0]):
                        ratios[i], ratios_sn[i], ratios2[i] = \
                            self.__compute_line_ratio(spectrum, i, z_try)

                    # add candidate to the list
                    candidate_info = spectrum.metadata()
                    for (ratio, ratio_sn, ratio2) in zip(ratios, ratios_sn, ratios2):
                        candidate_info.append(ratio)
                        candidate_info.append(ratio_sn)
                        candidate_info.append(ratio2)
                    candidate_info.append(z_try)
                    candidate_info.append(significance)
                    candidate_info.append(try_line)
                    candidates.append(candidate_info)

        columns_candidates = spectrum.metadata_names()
        for i in range(self.__lines.shape[0]):
            columns_candidates.append("{}_ratio".format(self.__lines.ix[i].name))
            columns_candidates.append("{}_ratio_SN".format(self.__lines.ix[i].name))
            columns_candidates.append("{}_ratio2".format(self.__lines.ix[i].name))
        columns_candidates.append("z_try")
        columns_candidates.append("peak_significance")
        columns_candidates.append("assumed_line")
        candidates_df = pd.DataFrame(candidates, columns=columns_candidates)

        # add truth table if running in training or test modes
        if self.__mode in ["training", "test"]:
            candidates_df["Delta_z"] = candidates_df["z_try"] - candidates_df["z_true"]
            if candidates_df.shape[0] > 0:
                candidates_df["is_correct"] = candidates_df.apply(self.__is_correct, axis=1)
                candidates_df["is_line"] = candidates_df.apply(self.__is_line, axis=1)
                candidates_df["correct_redshift"] = candidates_df.apply(self.__is_correct_redshift, axis=1)
            else:
                candidates_df["is_correct"] = pd.Series(dtype=bool)
                candidates_df["is_line"] = pd.Series(dtype=bool)
                candidates_df["correct_redshift"] = pd.Series(dtype=bool)

        return candidates_df

    def __load_model_settings(self):
        """ Overload the settings with those stored in self.__model """
        settings = self.__model.get_settings()
        self.__lines = deserialize(settings.get("lines"))
        self.__try_lines = settings.get("try_lines")
        self.__z_precision = settings.get("z_precision")
        self.__weighting_mode = settings.get("weighting_mode")
        self.__peakfind_width = settings.get("peakfind_width")
        self.__peakfind_sig = settings.get("peakfind_sig")

    def __save_candidates(self):
        """ Save the candidates DataFrame. """
        save_json(self.__name, self.__candidates)

    def candidates(self):
        """ Access the candidates DataFrame. """
        return self.__candidates

    def lines(self):
        """ Access the lines DataFrame. """
        return self.__lines

    def classify_candidates(self):
        """ Create a model instance and train it. Save the resulting model"""
        # consistency checks
        if self.__mode not in ["test", "operation"]:
            raise  Error("The function classify_candidates is available in the " +
                         "test mode only. Detected mode is {}".format(self.__mode))
        if self.__candidates is None:
            raise  Error("Attempting to run the function classify_candidates " +
                         "but no candidates were found/loaded. Check your " +
                         "formatter")
        self.__candidates = self.__model.compute_probability(self.__candidates)
        self.__save_candidates()

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

        for spectrum in spectra:
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
                                 get_results=False, userprint=verboseprint):
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

            get_results : boolean - Default: False
            If True, return the computed purity, the completeness, and the total number
            of found quasars. Otherwise, return None.

            userprint : function - Default: verboseprint
            Print function to use

            Returns
            -------
            If get_results is True, return the computed purity, the completeness, and
            the total number of found quasars. Otherwise, return None.
            """
        # consistency checks
        if self.__mode not in ["training", "test"]:
            raise  Error("The function find_completeness_purity is available in the " +
                         "training and test modes only. Detected mode is {}".format(self.__mode))

        if data_frame is None:
            data_frame = self.__candidates

        if "is_correct" not in data_frame.columns:
            raise Error("find_completeness_purity: invalid DataFrame, the column " +
                        "'is_correct' is missing")

        if "specid" not in data_frame.columns:
            raise Error("find_completeness_purity: invalid DataFrame, the column " +
                        "'specid' is missing")

        found_quasars = 0
        found_quasars_zge1 = 0
        found_quasars_zge2_1 = 0
        num_quasars = quasars_data_frame.shape[0]
        num_quasars_zge1 = quasars_data_frame[quasars_data_frame["z_true"] >= 1.0].shape[0]
        num_quasars_zge2_1 = quasars_data_frame[quasars_data_frame["z_true"] >= 2.1].shape[0]
        for index in np.arange(num_quasars):
            specid = quasars_data_frame.ix[quasars_data_frame.index[index]]["specid"]
            if data_frame[(data_frame["specid"] == specid) &
                          (data_frame["is_correct"])].shape[0] > 0:
                found_quasars += 1
                if quasars_data_frame.ix[quasars_data_frame.index[index]]["z_true"] >= 2.1:
                    found_quasars_zge2_1 += 1
                    found_quasars_zge1 += 1
                elif quasars_data_frame.ix[quasars_data_frame.index[index]]["z_true"] >= 1:
                    found_quasars_zge1 += 1
        if float(num_quasars) > 0.0:
            completeness = float(found_quasars)/float(num_quasars)
        else:
            completeness = np.nan
        if float(num_quasars_zge1) > 0.0:
            completeness_zge1 = float(found_quasars_zge1)/float(num_quasars_zge1)
        else:
            completeness_zge1 = np.nan
        if float(num_quasars_zge2_1) > 0.0:
            completeness_zge2_1 = float(found_quasars_zge2_1)/float(num_quasars_zge2_1)
        else:
            completeness_zge2_1 = np.nan


        if float(data_frame.shape[0]) > 0.:
            purity = float(data_frame["is_correct"].sum())/float(data_frame.shape[0])
            purity_zge1 = (float(data_frame[data_frame["z_true"] >= 1]["is_correct"].sum())/
                           float(data_frame[data_frame["z_true"] >= 1].shape[0]))
            purity_zge2_1 = (float(data_frame[data_frame["z_true"] >= 2.1]["is_correct"].sum())/
                             float(data_frame[data_frame["z_true"] >= 2.1].shape[0]))
            line_purity = float(data_frame["is_line"].sum())/float(data_frame.shape[0])
            #purity_to_quasars = (float(data_frame[data_frame["specid"] > 0].shape[0])/
            #                     float(data_frame.shape[0]))
            quasar_specids = np.unique(data_frame[data_frame["specid"] > 0]["specid"])
            specids = np.unique(data_frame["specid"])
            #quasar_spectra_fraction = float(quasar_specids.size)/float(specids.size)
        else:
            purity = np.nan
            purity_zge1 = np.nan
            purity_zge2_1 = np.nan
            line_purity = np.nan
            #purity_to_quasars = np.nan
            quasar_spectra_fraction = np.nan

        userprint("There are {} candidates ".format(data_frame.shape[0]),)
        userprint("for {} catalogued quasars".format(num_quasars))
        userprint("number of quasars = {}".format(num_quasars))
        userprint("found quasars = {}".format(found_quasars))
        userprint("completeness = {:.2%}".format(completeness))
        userprint("completeness z>=1 = {:.2%}".format(completeness_zge1))
        userprint("completeness z>=2.1 = {:.2%}".format(completeness_zge2_1))
        userprint("purity = {:.2%}".format(purity))
        userprint("purity z >=1 = {:.2%}".format(purity_zge1))
        userprint("purity z >=2.1 = {:.2%}".format(purity_zge2_1))
        userprint("line purity = {:.2%}".format(line_purity))
        #print "purity to quasars = {:.2%}".format(purity_to_quasars)
        #print "fraction of quasars = {:.2%}".format(quasar_spectra_fraction)
        if get_results:
            return purity, completeness, found_quasars

    def load_candidates(self, filename=None):
        """ Load the candidates DataFrame

            Parameters
            ----------
            filename : str - Default: None
            Name of the file from where to load existing candidates.
            If None, then load from self.__name
            """
        if filename is None:
            json_dict = load_json(self.__name)
        else:
            json_dict = load_json(filename)
        self.__candidates = deserialize(json_dict)

    def merge(self, others_list, userprint=verboseprint):
        """
            Merge self.__candidates with another candidates object

            Parameters
            ----------
            others_list : pd.DataFrame
            The other candidates object to merge

            userprint : function - Default: verboseprint
            Print function to use
            """
        if self.__mode != "merge":
            raise  Error("The function merge is available in the " +
                         "merge mode only. Detected mode is {}".format(self.__mode))

        for index, candidates_filename in enumerate(others_list):
            userprint("Merging... {} of {}".format(index, len(others_list)))

            try:
                # load candidates
                other = deserialize(load_json(candidates_filename))

                # append to candidates list
                self.__candidates = self.__candidates.append(other, ignore_index=True)

            except TypeError:
                userprint("Error occured when loading file {}.".format(candidates_filename))
                userprint("Ignoring file")

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

            Returns
            -------
            The figure object
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
            contaminants_df[plot_col].hist(ax=fig_ax, bins=100, range=(-1, 4),
                                           grid=False, color='r', alpha=0.5, normed=normed)
            correct_df[plot_col].hist(ax=fig_ax, bins=100, range=(-1, 4),
                                      grid=False, color='b', alpha=0.5, normed=normed)

        # plot the entire sample
        self.__candidates[plot_col].hist(ax=fig_ax, bins=100, range=(-1, 4), grid=False,
                                         histtype='step', color='k', normed=normed)
        # set axis labels
        fig_ax.set_xlabel(plot_col, fontsize=fontsize)
        if normed:
            fig_ax.set_ylabel("normalized distribution", fontsize=fontsize)
        else:
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

            Returns
            -------
            The figure object
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

    def train_model(self):
        """ Create a model instance and train it. Save the resulting model"""
        # consistency checks
        if self.__mode != "training":
            raise  Error("The function train_model is available in the " +
                         "training mode only. Detected mode is {}".format(self.__mode))

        selected_cols = [col for col in self.__candidates.columns if col.endswith("ratio_SN")]
        selected_cols += [col for col in self.__candidates.columns if col.endswith("ratio2")]
        selected_cols += [col for col in self.__candidates.columns if col.endswith("ratio")]
        selected_cols += ["peak_significance"]

        # add columns to compute the class in training
        selected_cols += ['class_person', 'correct_redshift']

        self.__model = Model("{}_model.json".format(self.__name[:self.__name.rfind(".")]),
                             selected_cols, self.__get_settings(),
                             model_opt=self.__model_opt,
                             cuts=self.__cuts)
        self.__model.train(self.__candidates)
        self.__model.save_model()

    def to_fits(self, filename, data_frame=None):
        """Save the DataFrame as a fits file. String columns with length greater than 15
            characters might be truncated

            Parameters
            ----------
            filename : str
            Name of the fits file the dataframe is going to be saved to

            data_frame : pd.DataFrame - Default: self.__candidates
            DataFrame to save
        """
        if data_frame is None:
            data_frame = self.__candidates

        if filename is None:
            filename = self.__name.replace("json", "fits")

        def convert_dtype(dtype):
             if dtype == "O":
                 return "15A"
             else:
                 return dtype

        hdu = fits.BinTableHDU.from_columns([fits.Column(name=col,
                                                         format=convert_dtype(dtype),
                                                         array=data_frame[col])
                                             for col, dtype in zip(data_frame.columns,
                                                                   data_frame.dtypes)])
        hdu.writeto(filename, overwrite=True)

if __name__ == '__main__':
    pass
