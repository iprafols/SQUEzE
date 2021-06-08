"""
    SQUEzE
    ======

    This file implements the class Candidates, that is used to generate the
    list of quasar candidates, trains or applies the model required for the
    cleaning process, and construct the final catalogue
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

from math import sqrt
import numpy as np

import pandas as pd

import astropy.io.fits as fits
from astropy.table import Table

from squeze.common_functions import verboseprint
from squeze.common_functions import deserialize
from squeze.error import Error
from squeze.model import Model
from squeze.peak_finder import PeakFinder
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
                 mode="operation", name="SQUEzE_candidates.fits.gz",
                 peakfind=(PEAKFIND_WIDTH, PEAKFIND_SIG),
                 model=None, model_opt=(RANDOM_FOREST_OPTIONS, RANDOM_STATE)):
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

            mode : "training", "test", "operation", "candidates", or "merge"
            - Default: "operation"
            Running mode. "training" mode assumes that true redshifts are known
            and provide a series of functions to train the model.

            name : string - Default: "SQUEzE_candidates.fits.gz"
            Name of the candidates sample. The code will save an python-binary
            with the information of the database in a csv file with this name.
            If load is set to True, then the candidates sample will be loaded
            from this file. Recommended extension is fits.gz.

            model : Model or None  - Default: None
            Instance of the Model class defined in squeze_model or None.
            In test and operation mode, it is supposed
            to be the quasar model to construct the catalogue. In training mode,
            it is supposed to be None initially, and the model will be trained
            and given as an output of the code.

            model_opt : (dict, int) - Defaut: (RANDOM_FOREST_OPTIONS, RANDOM_STATE)
            The first dictionary sets the options to be passed to the random forest
            cosntructor. If high-low split of the training is desired, the
            dictionary must contain the entries "high" and "low", and the
            corresponding values must be dictionaries with the options for each
            of the classifiers. In training mode, they're passed to the model
            instance before training. Otherwise it's ignored.
            """
        if mode in ["training", "test", "operation", "candidates", "merge"]:
            self.__mode = mode
        else:
            raise Error("Invalid mode")

        if name.endswith(".fits.gz") or name.endswith(".fits"):
            self.__name = name
        else:
            message = "Candidates name should have .fits or .fits.gz extensions."
            message += "Given name was {}".format(name)
            raise Error(message)

        # initialize empty catalogue
        self.__candidates_list = []
        self.__candidates = None

        # main settings
        self.__lines = lines_settings[0]
        self.__try_lines = lines_settings[1]
        self.__z_precision = z_precision

        # options to be passed to the peak finder
        self.__peakfind_width = peakfind[0]
        self.__peakfind_sig = peakfind[1]

        # model
        if model is None:
            self.__model = None
        else:
            self.__model = model
            self.__load_model_settings()
        self.__model_opt = model_opt

        # initialize peak finder
        self.__peak_finder = PeakFinder(self.__peakfind_width, self.__peakfind_sig)

    def __compute_line_ratio(self, spectrum, index, oneplusz):
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
        pix_peak = np.where((wave >= oneplusz*self.__lines.iloc[index]["START"])
                            & (wave <= oneplusz*self.__lines.iloc[index]["END"]))[0]
        pix_blue = np.where((wave >= oneplusz*self.__lines.iloc[index]["BLUE_START"])
                            & (wave <= oneplusz*self.__lines.iloc[index]["BLUE_END"]))[0]
        pix_red = np.where((wave >= oneplusz*self.__lines.iloc[index]["RED_START"])
                           & (wave <= oneplusz*self.__lines.iloc[index]["RED_END"]))[0]

        # compute peak and continuum values
        compute_ratio = True
        if ((pix_blue.size == 0) or (pix_peak.size == 0) or (pix_red.size == 0)
                or (pix_blue.size < pix_peak.size//2)
                or (pix_red.size < pix_peak.size//2)):
            compute_ratio = False
        else:
            peak = np.mean(flux[pix_peak])
            cont_red = np.mean(flux[pix_red])
            cont_blue = np.mean(flux[pix_blue])
            cont_red_and_blue = cont_red + cont_blue
            if (cont_red_and_blue == 0.0 or
                    isinstance(cont_red, np.ma.core.MaskedConstant) or
                    isinstance(cont_blue, np.ma.core.MaskedConstant) or
                    isinstance(peak, np.ma.core.MaskedConstant)):
                compute_ratio = False
            else:
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
            ratio2 = abs((cont_red - cont_blue)/cont_red_and_blue)
            err_ratio = sqrt(4.*peak_err_squared + ratio*ratio*cont_err_squared)/abs(cont_red_and_blue)
            ratio_sn = (ratio - 1.0)/err_ratio
        else:
            ratio = np.nan
            ratio2 = np.nan
            ratio_sn = np.nan

        return ratio, ratio_sn, ratio2

    def __get_settings(self):
        """ Pack the settings in a dictionary. Return it """
        return {"LINES": self.__lines,
                "TRY_LINES": self.__try_lines,
                "Z_PRECISION": self.__z_precision,
                "PEAKFIND_WIDTH": self.__peakfind_width,
                "PEAKFIND_SIG": self.__peakfind_sig,
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
        return bool((row["DELTA_Z"] <= self.__z_precision)
                    and row["DELTA_Z"] >= -self.__z_precision
                    and (np.isin(row["CLASS_PERSON"], [3, 30])))

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
        if row["CLASS_PERSON"] != 1:
            correct_redshift = bool((row["DELTA_Z"] <= self.__z_precision)
                                   and (row["DELTA_Z"] >= -self.__z_precision))
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
        if row["IS_CORRECT"]:
            is_line = True
        # not a quasar
        elif not np.isin(row["CLASS_PERSON"], [3, 30]):
            pass
        # not a peak
        elif row["ASSUMED_LINE"] == "none":
            pass
        else:
            for line in self.__lines.index:
                if line == row["ASSUMED_LINE"]:
                    continue
                z_try_line = (self.__lines["WAVE"][row["ASSUMED_LINE"]]/
                              self.__lines["WAVE"][line])*(1 + row["Z_TRY"]) - 1
                if ((z_try_line - row["Z_TRUE"] <= self.__z_precision) and
                        (z_try_line - row["Z_TRUE"] >= -self.__z_precision)):
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
            A list with the candidates for the given spectrum.
            """
        # find peaks
        peak_indexs, significances = self.__peak_finder.find_peaks(spectrum)

        # keep peaks in the spectrum
        # if there are no peaks, include the spectrum with redshift np.nan
        # assumed_line='none', significance is set to np.nan
        # and all the metrics set to np.nan
        if peak_indexs.size == 0:
            candidate_info = spectrum.metadata()
            z_try = np.nan
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
            self.__candidates_list.append(candidate_info)
        # if there are peaks, compute the metrics and keep the info
        else:
            for peak_index, significance in zip(peak_indexs, significances):
                for try_line in self.__try_lines:
                    # compute redshift
                    z_try = spectrum.wave()[peak_index]/self.__lines["WAVE"][try_line] - 1.0
                    if z_try < 0.0:
                        continue
                    oneplusz = (1.0 + z_try)

                    candidate_info = spectrum.metadata()

                    # compute peak ratio for the different lines
                    for i in range(self.__lines.shape[0]):
                        ratio, ratio_sn, ratio2 = \
                            self.__compute_line_ratio(spectrum, i, oneplusz)
                        candidate_info.append(ratio)
                        candidate_info.append(ratio_sn)
                        candidate_info.append(ratio2)

                    candidate_info.append(z_try)
                    candidate_info.append(significance)
                    candidate_info.append(try_line)

                    # add candidate to the list
                    self.__candidates_list.append(candidate_info)

    def __load_model_settings(self):
        """ Overload the settings with those stored in self.__model """
        settings = self.__model.get_settings()
        self.__lines = settings.get("LINES")
        self.__try_lines = settings.get("TRY_LINES")
        self.__z_precision = settings.get("Z_PRECISION")
        self.__peakfind_width = settings.get("PEAKFIND_WIDTH")
        self.__peakfind_sig = settings.get("PEAKFIND_SIG")

    def save_candidates(self):
        """ Save the candidates DataFrame. """
        def convert_dtype(dtype):
             if dtype == "O":
                 return "15A"
             else:
                 return dtype

        hdu = fits.BinTableHDU.from_columns([fits.Column(name=col,
                                                         format=convert_dtype(dtype),
                                                         array=self.__candidates[col])
                                             for col, dtype in zip(self.__candidates.columns,
                                                                   self.__candidates.dtypes)])
        hdu.writeto(self.__name, overwrite=True)

    def candidates(self):
        """ Access the candidates DataFrame. """
        return self.__candidates

    def lines(self):
        """ Access the lines DataFrame. """
        return self.__lines

    def set_mode(self, mode):
        """ Allow user to change the running mode

        Parameters
        ----------
        mode : "training", "test", "candidates", "operation", or "merge"
               - Default: "operation"
        Running mode. "training" mode assumes that true redshifts are known
        and provide a series of functions to train the model.
        """
        if mode in ["training", "test", "candidates", "operation", "merge"]:
            self.__mode = mode
        else:
            raise Error("Invalid mode")

        return

    def candidates_list_to_dataframe(self, columns_candidates, save=True):
        """ Format existing candidates list into a dataframe

            Parameters
            ----------
            columns_candidates : list of str
            The column names of the spectral metadata

            save : bool - default: True
            If True, then save the catalogue file after candidates are found
            """
        if len(self.__candidates_list) == 0:
            return

        for i in range(self.__lines.shape[0]):
            columns_candidates.append("{}_RATIO".format(self.__lines.iloc[i].name.upper()))
            columns_candidates.append("{}_RATIO_SN".format(self.__lines.iloc[i].name.upper()))
            columns_candidates.append("{}_RATIO2".format(self.__lines.iloc[i].name.upper()))
        columns_candidates.append("Z_TRY")
        columns_candidates.append("PEAK_SIGNIFICANCE")
        columns_candidates.append("ASSUMED_LINE")

        if self.__candidates is None:
            self.__candidates = pd.DataFrame(self.__candidates_list, columns=columns_candidates)
        else:
            aux = pd.DataFrame(self.__candidates_list, columns=columns_candidates)
            self.__candidates = pd.concat([self.__candidates, aux], ignore_index=True)

        # add truth table if running in training or test modes
        if (self.__mode in ["training", "test"] or
            (self.__mode == "candidates" and
            "Z_TRUE" in self.__candidates.columns)):
            self.__candidates["DELTA_Z"] = self.__candidates["Z_TRY"] - self.__candidates["Z_TRUE"]
            if self.__candidates.shape[0] > 0:
                self.__candidates["IS_CORRECT"] = self.__candidates.apply(self.__is_correct, axis=1)
                self.__candidates["IS_LINE"] = self.__candidates.apply(self.__is_line, axis=1)
                self.__candidates["CORRECT_REDSHIFT"] = self.__candidates.apply(self.__is_correct_redshift, axis=1)
            else:
                self.__candidates["IS_CORRECT"] = pd.Series(dtype=bool)
                self.__candidates["IS_LINE"] = pd.Series(dtype=bool)
                self.__candidates["CORRECT_REDSHIFT"] = pd.Series(dtype=bool)

        self.__candidates_list = []

        # save the new version of the catalogue
        if save:
            self.save_candidates()

    def classify_candidates(self, save=True):
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
        if save:
            self.save_candidates()

    def find_candidates(self, spectra):
        """ Find candidates for a given set of spectra, then integrate them in the
            candidates catalogue and save the new version of the catalogue.

            Parameters
            ----------
            spectra : list of Spectrum
            The spectra in which candidates will be looked for.
            """
        if self.__mode == "training" and "Z_TRUE" not in spectra[0].metadata_names():
            raise Error("Mode is set to 'training', but spectra do not " +
                        "have the property 'Z_TRUE'.")

        elif self.__mode == "test" and "Z_TRUE" not in spectra[0].metadata_names():
            raise Error("Mode is set to 'test', but spectra do not " +
                        "have the property 'Z_TRUE'.")

        elif self.__mode == "merge":
            raise Error("The function find_candidates is not available in " +
                        "merge mode.")

        for spectrum in spectra:
            # locate candidates in this spectrum
            # candidates are appended to self.__candidates_list
            self.__find_candidates(spectrum)


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

        if "IS_CORRECT" not in data_frame.columns:
            raise Error("find_completeness_purity: invalid DataFrame, the column " +
                        "'IS_CORRECT' is missing")

        if "SPECID" not in data_frame.columns:
            raise Error("find_completeness_purity: invalid DataFrame, the column " +
                        "'SPECID' is missing")

        found_quasars = 0
        found_quasars_zge1 = 0
        found_quasars_zge2_1 = 0
        num_quasars = quasars_data_frame.shape[0]
        num_quasars_zge1 = quasars_data_frame[quasars_data_frame["Z_TRUE"] >= 1.0].shape[0]
        num_quasars_zge2_1 = quasars_data_frame[quasars_data_frame["Z_TRUE"] >= 2.1].shape[0]
        for index in np.arange(num_quasars):
            specid = quasars_data_frame.iloc[quasars_data_frame.index[index]]["SPECID"]
            if data_frame[(data_frame["SPECID"] == specid) &
                          (data_frame["IS_CORRECT"])].shape[0] > 0:
                found_quasars += 1
                if quasars_data_frame.iloc[quasars_data_frame.index[index]]["Z_TRUE"] >= 2.1:
                    found_quasars_zge2_1 += 1
                    found_quasars_zge1 += 1
                elif quasars_data_frame.iloc[quasars_data_frame.index[index]]["Z_TRUE"] >= 1:
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
            purity = float(data_frame["IS_CORRECT"].sum())/float(data_frame.shape[0])
            purity_zge1 = (float(data_frame[data_frame["Z_TRUE"] >= 1]["IS_CORRECT"].sum())/
                           float(data_frame[data_frame["Z_TRUE"] >= 1].shape[0]))
            purity_zge2_1 = (float(data_frame[data_frame["Z_TRUE"] >= 2.1]["IS_CORRECT"].sum())/
                             float(data_frame[data_frame["Z_TRUE"] >= 2.1].shape[0]))
            line_purity = float(data_frame["IS_LINE"].sum())/float(data_frame.shape[0])
            #purity_to_quasars = (float(data_frame[data_frame["specid"] > 0].shape[0])/
            #                     float(data_frame.shape[0]))
            quasar_specids = np.unique(data_frame[data_frame["SPECID"] > 0]["SPECID"])
            specids = np.unique(data_frame["SPECID"])
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
            filename = self.__name

        data = Table.read(filename, format='fits')
        candidates = data.to_pandas()
        candidates.columns = candidates.columns.str.upper()
        self.__candidates = candidates
        del data, candidates

    def merge(self, others_list, userprint=verboseprint, save=True):
        """
            Merge self.__candidates with another candidates object

            Parameters
            ----------
            others_list : pd.DataFrame
            The other candidates object to merge

            userprint : function - Default: verboseprint
            Print function to use

            save : bool - Defaut: True
            If True, save candidates before exiting
            """
        if self.__mode != "merge":
            raise  Error("The function merge is available in the " +
                         "merge mode only. Detected mode is {}".format(self.__mode))

        for index, candidates_filename in enumerate(others_list):
            userprint("Merging... {} of {}".format(index, len(others_list)))

            try:
                # load candidates
                data = Table.read(candidates_filename, format='fits')
                other = data.to_pandas()
                del data

                # append to candidates list
                self.__candidates = self.__candidates.append(other, ignore_index=True)

            except TypeError:
                userprint("Error occured when loading file {}.".format(candidates_filename))
                userprint("Ignoring file")

        if save:
            self.save_candidates()

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
        # extra imports for this function
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        # plot settings
        fontsize = 20
        labelsize = 18
        ticksize = 10
        fig = plt.figure(figsize=(10, 6))
        axes_grid = gridspec.GridSpec(1, 1)
        axes_grid.update(hspace=0.4, wspace=0.0)

        # distinguish from contaminants and non-contaminants i necessary
        if self.__mode == "training":
            contaminants_df = self.__candidates[~self.__candidates["IS_CORRECT"]]
            correct_df = self.__candidates[self.__candidates["IS_CORRECT"]]

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
        # extra imports for this function
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        # get the number of plots and the names of the columns
        plot_cols = np.array([item for item in self.__candidates.columns if "RATIO" in item])
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
            contaminants_df = self.__candidates[~self.__candidates["IS_CORRECT"]]
            correct_df = self.__candidates[self.__candidates["IS_CORRECT"]]

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

    def train_model(self, model_fits):
        """ Create a model instance and train it. Save the resulting model

            Parameters
            ----------
            model_fits : bool
            If True, save the model as a fits file. Otherwise, save it as a
            json file.
            """
        # consistency checks
        if self.__mode != "training":
            raise  Error("The function train_model is available in the " +
                         "training mode only. Detected mode is {}".format(self.__mode))

        selected_cols = [col.upper() for col in self.__candidates.columns if col.endswith("RATIO_SN")]
        selected_cols += [col.upper() for col in self.__candidates.columns if col.endswith("RATIO2")]
        selected_cols += [col.upper() for col in self.__candidates.columns if col.endswith("RATIO")]
        selected_cols += ["PEAK_SIGNIFICANCE"]

        # add columns to compute the class in training
        selected_cols += ['CLASS_PERSON', 'CORRECT_REDSHIFT']

        if self.__name.endswith(".fits"):
            if model_fits:
                model_name = self.__name.replace(".fits", "_model.fits.gz")
            else:
                model_name = self.__name.replace(".fits", "_model.json")
        elif self.__name.endswith(".fits.gz"):
            if model_fits:
                model_name = self.__name.replace(".fits.gz", "_model.fits.gz")
            else:
                model_name = self.__name.replace(".fits.gz", "_model.json")
        else:
            raise Error("Invalid model name")
        self.__model = Model(model_name, selected_cols, self.__get_settings(),
                             model_opt=self.__model_opt)
        self.__model.train(self.__candidates)
        self.__model.save_model()

    def save_catalogue(self, filename, prob_cut):
        """ Save the final catalogue as a fits file. Only non-duplicated
            candidates with probability greater or equal to prob_cut will
            be included in this catalogue.
            String columns with length greater than 15
            characters might be truncated

            Parameters
            ----------
            filename : str or None
            Name of the fits file the final catalogue is going to be saved to.
            If it is None, then we will use self.__candidates with '_catalogue'
            appended to it before the extension.

            prob_cut : float
            Probability cut to be applied to the candidates. Only candidates
            with greater probability will be saved
        """
        if filename is None:
            filename = self.__name.replace(".fits", "_catalogue.fits")

        def convert_dtype(dtype):
             if dtype == "O":
                 return "15A"
             else:
                 return dtype

        # filter data DataFrame
        data_frame = self.__candidates[(~self.__candidates["DUPLICATED"]) &
                                       (self.__candidates["PROB"] >= prob_cut)]

        hdu = fits.BinTableHDU.from_columns([fits.Column(name=col,
                                                         format=convert_dtype(dtype),
                                                         array=data_frame[col])
                                             for col, dtype in zip(data_frame.columns,
                                                                   data_frame.dtypes)])
        hdu.writeto(filename, overwrite=True)

if __name__ == '__main__':
    pass
