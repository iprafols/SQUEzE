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
import time
import json

import numpy as np
from numba import prange, jit, vectorize
import pandas as pd
import fitsio
from astropy.table import Table

from squeze.candidates_utils import (
    compute_line_ratios, compute_pixel_metrics,
    compute_is_correct, compute_is_correct_redshift, compute_is_line)
from squeze.error import Error
from squeze.model import Model
from squeze.utils import (
    verboseprint, function_from_string, deserialize, load_json)


# extra imports for plotting function
PLOTTING_ERROR = None
try:
    import matplotlib.pyplot as plt
except ImportError as error:
    PLOTTING_ERROR = error

MODES = ["training", "test", "operation", "candidates", "merge"]

# This variable sets the maximum number of candidates allowed before a partial
# conversion to dataframe is executed
MAX_CANDIDATES_TO_CONVERT = 100000000

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
    def __init__(self, config):
        """ Initialize class instance.

        Arguments
        ---------
        config: Config
        A configuration instance
        """
        self.config = config
        general_config = self.config.get_section("general")

        # printing function
        userprint = general_config.get("userprint")
        if userprint is None:
            message = "Expected printing function, found None"
            raise Error(message)
        try:
            self.userprint = function_from_string(userprint, "squeze.utils")
        except ImportError as error:
            raise Error(
                f"Error loading class {peak_finder_name}, "
                f"module {module_name} could not be loaded") from error
        except AttributeError as error:
            raise Error(
                f"Error loading class {peak_finder_name}, "
                f"module {module_name} did not contain requested class"
            ) from error

        # mode
        self.mode = general_config.get("mode")
        if self.mode not in MODES:
            message = (
                f"Invalid mode. Expected one of {', '.join(MODES)}. Found"
                f"{self.mode}")
            raise Error(message)

        # output info
        self.name = general_config.get("output")
        if not (self.name.endswith(".fits.gz") or
                self.name.endswith(".fits")):
            message = (
                "Candidates name should have .fits or .fits.gz extensions."
                f"Given name was {self.name}")
            raise Error(message)

        # initialize empty catalogue
        self.candidates_list = []
        self.candidates = None

        # main settings
        self.lines = None
        self.num_pixels = None
        self.pixels_as_metrics = None
        self.try_lines = None
        self.try_lines_indexs = None
        self.try_lines_dict = None
        self.z_precision = None
        self.__initialize_main_settings()

        # model
        model_config = self.config.get_section("model")
        model_filename = model_config.get("filename")
        if model_filename is None:
            self.model = None
        else:
            self.userprint("Loading model")
            t0 = time.time()
            if model_filename.endswith(".json"):
                self.model = Model.from_json(load_json(model_filename))
            else:
                self.model = Model.from_fits(model_filename)
            t1 = time.time()
            self.userprint(f"INFO: time elapsed to load model: {(t1-t0)/60.0} minutes")
            self.__load_model_settings()

        # model options
        random_state = model_config.getint("random state")
        random_forest_options = model_config.get("random forest options")
        if random_forest_options is None:
            self.model_options = ({}, random_state)
        else:
            self.model_options = (load_json(random_forest_options), random_state)

        # initialize peak finder
        self.__initialize_peak_finder()

    def __initialize_main_settings(self):
        """ Initialize main settings"""
        settings = self.config.get_section("candidates")

        # line metrics
        lines = settings.get("lines")
        if lines is None:
            message = "In section [candidates], variable 'lines' is required"
            raise Error(message)
        self.lines = deserialize(load_json(lines))
        if not isinstance(self.lines, pd.DataFrame):
            message = ("Expected a DataFrame with the line information. "
                       f"Found: {type(self.lines)}\n    lines: {lines}\n"
                       f"self.lines: {self.lines}")
            raise Error(message)

        try_lines = settings.get("try lines")
        self.try_lines = try_lines.split()
        if not isinstance(self.try_lines, list):
            message = ("Expected a list with the try lines names. "
                       f"Found: {self.try_lines}")
            raise Error(message)

        self.z_precision = settings.getfloat("z precision")
        if self.z_precision is None or self.z_precision <= 0:
            message = ("z precision must be greater than 0. "
                       f"Found {self.z_precision}")
            raise Error(message)

        self.pixels_as_metrics = settings.getboolean("pixels as metrics")

        self.num_pixels = settings.getint("num pixels")
        if self.num_pixels is None or self.num_pixels <= 0:
            message = ("num pixels must be greater than 0. "
                       f"Found {self.num_pixels}")
            raise Error(message)

        # make sure fields in self.lines are properly sorted
        self.lines = self.lines[[
            'WAVE', 'START', 'END', 'BLUE_START', 'BLUE_END', 'RED_START',
            'RED_END'
        ]]

        # compute convert try_lines strings to indexs in self.lines array
        self.try_lines_indexs = np.array([
            np.where(self.lines.index == item)[0][0]
            for item in self.try_lines
        ])
        self.try_lines_dict = dict(
            zip(self.try_lines, self.try_lines_indexs))
        self.try_lines_dict["none"] = -1

    def __initialize_peak_finder(self):
        """Initialize the peak finder"""
        # figure out which Peak Finder to load
        PeakFinder, arguments = self.config.get_peak_finder()

        # initialize peak finder
        self.peak_finder = PeakFinder(arguments)

        # some variables that will be removed later on but that for now are
        # required for the model to work
        self.peakfind_width = arguments.getfloat("width")
        self.peakfind_sig = arguments.getfloat("min significance")

    def __get_settings(self):
        """ Pack the settings in a dictionary. Return it """
        return {
            "LINES": self.lines,
            "TRY_LINES": self.try_lines,
            "Z_PRECISION": self.z_precision,
            "PEAKFIND_WIDTH": self.peakfind_width,
            "PEAKFIND_SIG": self.peakfind_sig,
            "PIXELS_AS_METRICS": self.pixels_as_metrics,
            "NUM_PIXELS": self.num_pixels,
        }

    def __find_candidates(self, spectrum):
        """ Find the candidates in a spectrum.

        Given a Spectrum, locate peaks in the flux. Then assume these peaks
        correspond to the emission lines in self.try_lines and compute
        peak-to-continuum ratios for the selected lines (see Perez-Rafols et al.
        2020 for details).
        For each of the peaks, report a list of candidates with their redshift
        estimation, the computed peak-to-continuum ratios, and the metadata
        specified in the spectrum.

        Arguments
        ---------
        spectrum : Spectrum
        The spectrum where candidates are looked for.

        Return
        ------
        A list with the candidates for the given spectrum.
        """
        # find peaks
        peak_indexs, significances = self.peak_finder.find_peaks(spectrum)

        # keep peaks in the spectrum
        # if there are no peaks, include the spectrum with redshift np.nan
        # assumed_line='none', significance is set to np.nan
        # and all the metrics set to np.nan
        if peak_indexs.size == 0:
            candidate_info = spectrum.metadata()
            z_try = np.nan
            significance = np.nan
            try_line = 'none'
            ratios = np.zeros(self.lines.shape[0], dtype=float)
            ratios_sn = np.zeros_like(ratios)
            ratios2 = np.zeros_like(ratios)
            for _ in zip(ratios, ratios_sn, ratios2):
                candidate_info.append(np.nan)
                candidate_info.append(np.nan)
                candidate_info.append(np.nan)
            candidate_info.append(z_try)
            candidate_info.append(significance)
            candidate_info.append(try_line)
            if self.pixels_as_metrics:
                for _ in range(-self.num_pixels, 0):
                    candidate_info.append(np.nan)
                for _ in range(0, self.num_pixels):
                    candidate_info.append(np.nan)
            self.candidates_list.append(candidate_info)
        # if there are peaks, compute the metrics and keep the info
        else:
            wave = spectrum.wave()
            flux = spectrum.flux()
            ivar = spectrum.ivar()
            metadata = spectrum.metadata()

            new_candidates = compute_line_ratios(wave, flux, ivar, peak_indexs,
                                                 significances,
                                                 self.try_lines_indexs,
                                                 self.lines.values)

            new_candidates = [
                metadata + item[:-1] + [self.try_lines[int(item[-1])]]
                for item in new_candidates
            ]

            if self.pixels_as_metrics:
                pixel_metrics = compute_pixel_metrics(wave, flux, ivar,
                                                      peak_indexs,
                                                      self.num_pixels,
                                                      self.try_lines_indexs,
                                                      self.lines.values)
                new_candidates = [
                    item + candidate_pixel_metrics
                    for item, candidate_pixel_metrics in zip(
                        new_candidates, pixel_metrics)
                ]

            self.candidates_list += new_candidates

    def __load_model_settings(self):
        """ Overload the settings with those stored in self.model """
        settings = self.model.get_settings()
        self.lines = settings.get("LINES")
        self.try_lines = settings.get("TRY_LINES")
        self.z_precision = settings.get("Z_PRECISION")
        self.peakfind_width = settings.get("PEAKFIND_WIDTH")
        self.peakfind_sig = settings.get("PEAKFIND_SIG")
        self.pixels_as_metrics = settings.get("PIXELS_AS_METRICS")
        self.num_pixels = settings.get("NUM_PIXELS")

    def save_candidates(self):
        """ Save the candidates DataFrame. """
        results = fitsio.FITS(self.name, 'rw', clobber=True)
        names = list(self.candidates.columns)
        cols = [
            np.array(self.candidates[col].values, dtype=str)
            if self.candidates[col].dtype == "object" else
            self.candidates[col].values for col in self.candidates.columns
        ]
        results.write(cols, names=names, extname="CANDIDATES")
        results.close()

    def candidates_list_to_dataframe(self, columns_candidates, save=True):
        """ Format existing candidates list into a dataframe

        Arguments
        ---------
        columns_candidates : list of str
        The column names of the spectral metadata

        save : bool - default: True
        If True, then save the catalogue file after candidates are found
        """
        if len(self.candidates_list) == 0:
            return

        if "Z_TRY" not in columns_candidates:
            for index1 in range(self.lines.shape[0]):
                columns_candidates.append(
                    f"{self.lines.iloc[index1].name.upper()}_RATIO")
                columns_candidates.append(
                    f"{self.lines.iloc[index1].name.upper()}_RATIO_SN")
                columns_candidates.append(
                    f"{self.lines.iloc[index1].name.upper()}_RATIO2")
            columns_candidates.append("Z_TRY")
            columns_candidates.append("PEAK_SIGNIFICANCE")
            columns_candidates.append("ASSUMED_LINE")
            if self.pixels_as_metrics:
                for index in range(-self.num_pixels, 0):
                    columns_candidates.append(f"FLUX_{index}")
                    columns_candidates.append(f"IVAR_{index}")
                for index in range(0, self.num_pixels):
                    columns_candidates.append(f"FLUX_{index}")
                    columns_candidates.append(f"IVAR_{index}")

        # create dataframe
        aux = pd.DataFrame(self.candidates_list, columns=columns_candidates)

        # add truth table if running in training or test modes
        if (self.mode in ["training", "test"] or
            (self.mode == "candidates" and "Z_TRUE" in aux.columns)):
            self.userprint("Adding control variables from truth table")
            aux["DELTA_Z"] = aux["Z_TRY"] - aux["Z_TRUE"]
            if aux.shape[0] > 0:
                self.userprint("    is_correct_redshift")
                aux["CORRECT_REDSHIFT"] = compute_is_correct_redshift(
                    aux["DELTA_Z"].values, aux["CLASS_PERSON"].values,
                    self.z_precision)
                self.userprint("    is_correct")
                aux["IS_CORRECT"] = compute_is_correct(
                    aux["CORRECT_REDSHIFT"].values, aux["CLASS_PERSON"].values)
                self.userprint("    is_line")
                aux["IS_LINE"] = compute_is_line(
                    aux["IS_CORRECT"].values, aux["CLASS_PERSON"].values,
                    np.array([
                        self.try_lines_dict.get(assumed_line)
                        for assumed_line in aux["ASSUMED_LINE"]
                    ]), aux["Z_TRUE"].values, aux["Z_TRY"].values,
                    self.z_precision,
                    self.lines.iloc[self.try_lines_indexs].values)
            else:
                self.userprint("    is_correct_redshift")
                aux["CORRECT_REDSHIFT"] = pd.Series(dtype=bool)
                self.userprint("    is_correct")
                aux["IS_CORRECT"] = pd.Series(dtype=bool)
                self.userprint("    is_line")
                aux["IS_LINE"] = pd.Series(dtype=bool)
            self.userprint("Done")

        # keep the results
        if self.candidates is None:
            self.candidates = aux
        else:
            self.userprint(
                "Concatenating dataframe with previouly exisiting candidates")
            self.candidates = pd.concat([self.candidates, aux],
                                          ignore_index=True)
            self.userprint("Done")
        self.candidates_list = []

        # save the new version of the catalogue
        if save:
            self.userprint("Saving candidates")
            self.save_candidates()
            self.userprint("Done")

    def classify_candidates(self, save=True):
        """ Create a model instance and train it. Save the resulting model

        Arguments
        ---------
        save : bool - default: True
        If True, then save the catalogue file after predictions are made
        """
        # consistency checks
        if self.mode not in ["test", "operation"]:
            raise Error(
                "The function classify_candidates is available in the " +
                f"test mode only. Detected mode is {self.mode}")
        if self.candidates is None:
            raise Error("Attempting to run the function classify_candidates " +
                        "but no candidates were found/loaded. Check your " +
                        "formatter")
        self.candidates = self.model.compute_probability(self.candidates)
        if save:
            self.save_candidates()

    def find_candidates(self, spectra, columns_candidates):
        """ Find candidates for a given set of spectra

        Integrate them in the candidates catalogue.

        Arguments
        ---------
        spectra : list of Spectrum
        The spectra in which candidates will be looked for.

        columns_candidates : list of str
        The column names of the spectral metadata
        """
        if self.mode == "training" and "Z_TRUE" not in spectra[
                0].metadata_names():
            raise Error("Mode is set to 'training', but spectra do not " +
                        "have the property 'Z_TRUE'.")

        if self.mode == "test" and "Z_TRUE" not in spectra[0].metadata_names(
        ):
            raise Error("Mode is set to 'test', but spectra do not " +
                        "have the property 'Z_TRUE'.")

        if self.mode == "merge":
            raise Error("The function find_candidates is not available in " +
                        "merge mode.")

        for spectrum in spectra:
            # locate candidates in this spectrum
            # candidates are appended to self.candidates_list
            self.__find_candidates(spectrum)

            if len(self.candidates_list) > MAX_CANDIDATES_TO_CONVERT:
                self.userprint("Converting candidates to dataframe")
                time0 = time.time()
                self.candidates_list_to_dataframe(columns_candidates,
                                                  save=False)
                time1 = time.time()
                self.userprint(
                    "INFO: time elapsed to convert candidates to dataframe: "
                    f"{(time0-time1)/60.0} minutes")

    def find_completeness_purity(self, quasars_data_frame, data_frame=None):
        """ Find purity and completeness

        Given a DataFrame with candidates and another one with the catalogued
        quasars, compute the completeness and the purity. Upon error, return
        np.nan

        Arguments
        ---------
        quasars_data_frame : string
        DataFrame containing the quasar catalogue. The quasars must contain
        the column "specid" to identify the spectrum.

        data_frame : pd.DataFrame - Default: self.candidates
        DataFrame where the percentile will be computed. Must contain the
        columns "is_correct" and "specid".

        Return
        ------
        purity : float
        The computed purity

        completeness: float
        The computed completeness

        found_quasars : int
        The total number of found quasars.
        """
        # consistency checks
        if self.mode not in ["training", "test"]:
            raise Error(
                "The function find_completeness_purity is available in the " +
                f"training and test modes only. Detected mode is {self.mode}")

        if data_frame is None:
            data_frame = self.candidates

        if "IS_CORRECT" not in data_frame.columns:
            raise Error(
                "find_completeness_purity: invalid DataFrame, the column " +
                "'IS_CORRECT' is missing")

        if "SPECID" not in data_frame.columns:
            raise Error(
                "find_completeness_purity: invalid DataFrame, the column " +
                "'SPECID' is missing")

        found_quasars = 0
        found_quasars_zge1 = 0
        found_quasars_zge2_1 = 0
        num_quasars = quasars_data_frame.shape[0]
        num_quasars_zge1 = quasars_data_frame[
            quasars_data_frame["Z_TRUE"] >= 1.0].shape[0]
        num_quasars_zge2_1 = quasars_data_frame[
            quasars_data_frame["Z_TRUE"] >= 2.1].shape[0]
        for index in np.arange(num_quasars):
            specid = quasars_data_frame.iloc[
                quasars_data_frame.index[index]]["SPECID"]
            if data_frame[(data_frame["SPECID"] == specid) &
                          (data_frame["IS_CORRECT"])].shape[0] > 0:
                found_quasars += 1
                if quasars_data_frame.iloc[
                        quasars_data_frame.index[index]]["Z_TRUE"] >= 2.1:
                    found_quasars_zge2_1 += 1
                    found_quasars_zge1 += 1
                elif quasars_data_frame.iloc[
                        quasars_data_frame.index[index]]["Z_TRUE"] >= 1:
                    found_quasars_zge1 += 1
        if float(num_quasars) > 0.0:
            completeness = float(found_quasars) / float(num_quasars)
        else:
            completeness = np.nan
        if float(num_quasars_zge1) > 0.0:
            completeness_zge1 = float(found_quasars_zge1) / float(
                num_quasars_zge1)
        else:
            completeness_zge1 = np.nan
        if float(num_quasars_zge2_1) > 0.0:
            completeness_zge2_1 = float(found_quasars_zge2_1) / float(
                num_quasars_zge2_1)
        else:
            completeness_zge2_1 = np.nan

        if float(data_frame.shape[0]) > 0.:
            purity = float(data_frame["IS_CORRECT"].sum()) / float(
                data_frame.shape[0])
            purity_zge1 = (
                float(data_frame[data_frame["Z_TRUE"] >= 1]["IS_CORRECT"].sum())
                / float(data_frame[data_frame["Z_TRUE"] >= 1].shape[0]))
            purity_zge2_1 = (
                float(
                    data_frame[data_frame["Z_TRUE"] >= 2.1]["IS_CORRECT"].sum())
                / float(data_frame[data_frame["Z_TRUE"] >= 2.1].shape[0]))
            line_purity = float(data_frame["IS_LINE"].sum()) / float(
                data_frame.shape[0])
        else:
            purity = np.nan
            purity_zge1 = np.nan
            purity_zge2_1 = np.nan
            line_purity = np.nan

        self.userprint(f"There are {data_frame.shape[0]} candidates ",)
        self.userprint(f"for {num_quasars} catalogued quasars")
        self.userprint(f"number of quasars = {num_quasars}")
        self.userprint(f"found quasars = {found_quasars}")
        self.userprint(f"completeness = {completeness:.2%}")
        self.userprint(f"completeness z>=1 = {completeness_zge1:.2%}")
        self.userprint(f"completeness z>=2.1 = {completeness_zge2_1:.2%}")
        self.userprint(f"purity = {purity:.2%}")
        self.userprint(f"purity z >=1 = {purity_zge1:.2%}")
        self.userprint(f"purity z >=2.1 = {purity_zge2_1:.2%}")
        self.userprint(f"line purity = {line_purity:.2%}")

        return purity, completeness, found_quasars

    def load_candidates(self, filename=None):
        """ Load the candidates DataFrame

        Parameter
        ---------
        filename : str - Default: None
        Name of the file from where to load existing candidates.
        If None, then load from self.name
        """
        if filename is None:
            filename = self.name

        data = Table.read(filename, format='fits')
        candidates = data.to_pandas()
        candidates.columns = candidates.columns.str.upper()
        self.candidates = candidates
        del data, candidates

    def merge(self, others_list, save=True):
        """ Merge self.candidates with another candidates object

        Parameter
        ---------
        others_list : pd.DataFrame
        The other candidates object to merge

        save : bool - Defaut: True
        If True, save candidates before exiting
        """
        if self.mode != "merge":
            raise Error("The function merge is available in the " +
                        f"merge mode only. Detected mode is {self.mode}")

        for index, candidates_filename in enumerate(others_list):
            self.userprint(f"Merging... {index} of {len(others_list)}")

            try:
                # load candidates
                data = Table.read(candidates_filename, format='fits')
                other = data.to_pandas()
                del data

                # append to candidates list
                self.candidates = self.candidates.append(other,
                                                             ignore_index=True)

            except TypeError:
                self.userprint(
                    f"Error occured when loading file {candidates_filename}.")
                self.userprint("Ignoring file")

        if save:
            self.save_candidates()

    def plot_histograms(self, plot_col, normed=True):
        """ Plot the histogram of the specified column

        In training mode, separate the histograms by distinguishing
        contaminants and non-contaminants

        Arguments
        ---------
        plot_col : str
        Name of the column to plot

        normed : bool - Default: True
        If True, then plot the normalized histograms

        Return
        ------
        The figure object
        """
        if PLOTTING_ERROR is not None:
            raise PLOTTING_ERROR

        # plot settings
        fontsize = 20
        labelsize = 18
        ticksize = 10
        fig = plt.figure(figsize=(10, 6))
        axes_grid = fig.add_gridspec.GridSpec(norws=1, ncols=1)
        axes_grid.update(hspace=0.4, wspace=0.0)

        # distinguish from contaminants and non-contaminants i necessary
        if self.mode == "training":
            contaminants_df = self.candidates[~self.
                                                __candidates["IS_CORRECT"]]
            correct_df = self.candidates[self.candidates["IS_CORRECT"]]

        # plot the histograms
        fig_ax = fig.add_subplot(axes_grid[0])

        # plot the contaminants and non-contaminants separately
        if self.mode == "training":
            contaminants_df[plot_col].hist(ax=fig_ax,
                                           bins=100,
                                           range=(-1, 4),
                                           grid=False,
                                           color='r',
                                           alpha=0.5,
                                           normed=normed)
            correct_df[plot_col].hist(ax=fig_ax,
                                      bins=100,
                                      range=(-1, 4),
                                      grid=False,
                                      color='b',
                                      alpha=0.5,
                                      normed=normed)

        # plot the entire sample
        self.candidates[plot_col].hist(ax=fig_ax,
                                         bins=100,
                                         range=(-1, 4),
                                         grid=False,
                                         histtype='step',
                                         color='k',
                                         normed=normed)
        # set axis labels
        fig_ax.set_xlabel(plot_col, fontsize=fontsize)
        if normed:
            fig_ax.set_ylabel("normalized distribution", fontsize=fontsize)
        else:
            fig_ax.set_ylabel("counts", fontsize=fontsize)
        fig_ax.tick_params(axis='both',
                           labelsize=labelsize,
                           pad=0,
                           top=True,
                           right=True,
                           length=ticksize,
                           direction="inout")
        return fig

    def plot_line_ratios_histograms(self, normed=True):
        """ Plot the histogram of the ratios for the different lines.

        In training mode, separate
        the histograms by distinguishing contaminants and
        non-contaminants

        Arguments
        ---------
        normed : bool - Default: True
        If True, then plot the normalized histograms

        Return
        ------
        The figure object
        """
        if PLOTTING_ERROR is not None:
            raise PLOTTING_ERROR

        # get the number of plots and the names of the columns
        plot_cols = np.array(
            [item for item in self.candidates.columns if "RATIO" in item])
        num_ratios = plot_cols.size

        # plot settings
        fontsize = 20
        labelsize = 18
        ticksize = 10
        fig = plt.figure(figsize=(10, 6 * num_ratios))
        axes = []
        axes_grid = fig.add_gridspec(nrows=num_ratios, ncols=1)
        axes_grid.update(hspace=0.4, wspace=0.0)

        # distinguish from contaminants and non-contaminants i necessary
        if self.mode == "training":
            contaminants_df = self.candidates[~self.
                                                __candidates["IS_CORRECT"]]
            correct_df = self.candidates[self.candidates["IS_CORRECT"]]

        # plot the histograms
        for index, plot_col in enumerate(plot_cols):
            axes.append(fig.add_subplot(axes_grid[index]))
            # plot the contaminants and non-contaminants separately
            if self.mode == "training":
                contaminants_df[plot_col].hist(ax=axes[index],
                                               bins=100,
                                               range=(-1, 4),
                                               grid=False,
                                               color='r',
                                               alpha=0.5,
                                               normed=normed)
                correct_df[plot_col].hist(ax=axes[index],
                                          bins=100,
                                          range=(-1, 4),
                                          grid=False,
                                          color='b',
                                          alpha=0.5,
                                          normed=normed)

            # plot the entire sample
            self.candidates[plot_col].hist(ax=axes[index],
                                             bins=100,
                                             range=(-1, 4),
                                             grid=False,
                                             histtype='step',
                                             color='k',
                                             normed=normed)
            # set axis labels
            axes[index].set_xlabel(plot_col, fontsize=fontsize)
            axes[index].set_ylabel("counts", fontsize=fontsize)
            axes[index].tick_params(axis='both',
                                    labelsize=labelsize,
                                    pad=0,
                                    top=True,
                                    right=True,
                                    length=ticksize,
                                    direction="inout")
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
        if self.mode != "training":
            raise Error("The function train_model is available in the " +
                        f"training mode only. Detected mode is {self.mode}")

        selected_cols = [
            col.upper()
            for col in self.candidates.columns
            if col.endswith("RATIO_SN")
        ]
        selected_cols += [
            col.upper()
            for col in self.candidates.columns
            if col.endswith("RATIO2")
        ]
        selected_cols += [
            col.upper()
            for col in self.candidates.columns
            if col.endswith("RATIO")
        ]
        selected_cols += [
            col.upper()
            for col in self.candidates.columns
            if col.startswith("FLUX_")
        ]
        selected_cols += [
            col.upper()
            for col in self.candidates.columns
            if col.startswith("IVAR_")
        ]
        selected_cols += ["PEAK_SIGNIFICANCE"]

        # add extra columns
        if len(self.model_options
              ) == 3 and self.model_options[2] is not None:
            selected_cols += [item.upper() for item in self.model_options[2]]

        # add columns to compute the class in training
        selected_cols += ['CLASS_PERSON', 'CORRECT_REDSHIFT']

        if self.name.endswith(".fits"):
            if model_fits:
                model_name = self.name.replace(".fits", "_model.fits.gz")
            else:
                model_name = self.name.replace(".fits", "_model.json")
        elif self.name.endswith(".fits.gz"):
            if model_fits:
                model_name = self.name.replace(".fits.gz", "_model.fits.gz")
            else:
                model_name = self.name.replace(".fits.gz", "_model.json")
        else:
            raise Error("Invalid model name")
        self.model = Model(model_name,
                             selected_cols,
                             self.__get_settings(),
                             model_options=self.model_options)
        self.model.train(self.candidates)
        self.model.save_model()

    def save_catalogue(self, filename, prob_cut):
        """ Save the final catalogue as a fits file.

        Only non-duplicated candidates with probability greater or equal
        to prob_cut will be included in this catalogue.
        String columns with length greater than 15 characters might be truncated

        Arguments
        ---------
        filename : str or None
        Name of the fits file the final catalogue is going to be saved to.
        If it is None, then we will use self.candidates with '_catalogue'
        appended to it before the extension.

        prob_cut : float
        Probability cut to be applied to the candidates. Only candidates
        with greater probability will be saved
        """
        if filename is None:
            filename = self.name.replace(".fits", "_catalogue.fits")

        # filter data DataFrame
        data_frame = self.candidates[(~self.candidates["DUPLICATED"]) &
                                       (self.candidates["PROB"] >= prob_cut)]

        results = fitsio.FITS(filename, 'rw', clobber=True)
        names = list(data_frame.columns)
        cols = [
            np.array(data_frame[col].values, dtype=str)
            if data_frame[col].dtype == "object" else data_frame[col].values
            for col in data_frame.columns
        ]
        results.write(cols, names=names, extname="CANDIDATES")
        results.close()


if __name__ == '__main__':
    pass
