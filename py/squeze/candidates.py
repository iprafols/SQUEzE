"""
    SQUEzE
    ======

    This file implements the class Candidates, that is used to generate the
    list of quasar candidates, train or apply the model required for the
    cleaning process, and construct the final catalogue
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

from math import sqrt
import time
import os

import numpy as np
from numba import prange, jit, vectorize
import pandas as pd
import fitsio

from squeze.candidates_utils import (
    compute_line_ratios, compute_pixel_metrics, compute_is_correct,
    compute_is_correct_redshift, compute_is_line, load_df)
from squeze.config import Config
from squeze.error import Error
from squeze.model import Model
from squeze.quasar_catalogue import QuasarCatalogue
from squeze.spectra import Spectra
from squeze.utils import (
    verboseprint, deserialize, load_json)


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
        self.userprint = self.config.userprint

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

        # main settings
        self.input_spectra = None
        self.lines = None
        self.num_pixels = None
        self.pixels_as_metrics = None
        self.prob_cut = None
        self.save_catalogue_flag = None
        self.try_lines = None
        self.try_lines_indexs = None
        self.try_lines_dict = None
        self.z_precision = None
        self.__initialize_main_settings()

        # model
        self.__initialize_model()

        # initialize peak finder
        self.__initialize_peak_finder()

        # initialize candidates list
        self.candidates_list = []
        self.candidates = None
        self.load_candidates()

    def __initialize_main_settings(self):
        """ Initialize main settings"""
        settings = self.config.get_section("candidates")

        # input spectra
        input_spectra = settings.get("input spectra")
        if input_spectra is None:
            self.input_spectra = input_spectra
        else:
            self.input_spectra = input_spectra.split()

        # line metrics
        lines = settings.get("lines")
        if lines is None:
            message = "In section [candidates], variable 'lines' is required"
            raise Error(message)
        self.lines = deserialize(load_json(os.path.expandvars(lines)))
        if not isinstance(self.lines, pd.DataFrame):
            message = ("Expected a DataFrame with the line information. "
                       f"Found: {type(self.lines)}\n    lines: {lines}\n"
                       f"self.lines: {self.lines}")
            raise Error(message)

        self.num_pixels = settings.getint("num pixels")
        if self.num_pixels is None or self.num_pixels <= 0:
            message = ("num pixels must be greater than 0. "
                       f"Found {self.num_pixels}")
            raise Error(message)

        self.pixels_as_metrics = settings.getboolean("pixels as metrics")

        self.prob_cut = settings.getfloat("prob cut")
        if self.prob_cut is None or self.prob_cut < 0.0:
            message = ("prob cut must be greater than or equal to 0.0. "
                       f"Found {self.prob_cut}")
            raise Error(message)

        self.save_catalogue_flag = settings.getboolean("save catalogue")
        if self.save_catalogue is None:
            message = (
                "In section [candidates], variable 'save catalogue' is required")
            raise Error(message)

        # try lines
        try_lines = settings.get("try lines")
        self.try_lines = try_lines.split()
        if not isinstance(self.try_lines, list):
            message = ("Expected a list with the try lines names. "
                       f"Found: {self.try_lines}")
            raise Error(message)

        # redsift precision
        self.z_precision = settings.getfloat("z precision")
        if self.z_precision is None or self.z_precision <= 0:
            message = ("z precision must be greater than 0. "
                       f"Found {self.z_precision}")
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

    def __initialize_model(self):
        """Initialize the model"""
        model_config = self.config.get_section("model")
        model_filename = model_config.get("filename")

        # create model from scratch
        if self.mode == "training" or model_filename is None:
            # update selected cols
            selected_cols = []
            selected_cols += [f"{line.upper()}_RATIO_SN" for line in self.lines.index]
            selected_cols += [f"{line.upper()}_RATIO2" for line in self.lines.index]
            selected_cols += [f"{line.upper()}_RATIO" for line in self.lines.index]
            if self.pixels_as_metrics:
                selected_cols += [f"FLUX_{index}" for index in range(-self.num_pixels, 0)]
                selected_cols += [f"FLUX_{index}" for index in range(0, self.num_pixels)]
            selected_cols += ["PEAK_SIGNIFICANCE"]
            # add extra columns
            pass_cols_to_random_forest = model_config.get("pass cols to random forest")
            if pass_cols_to_random_forest is not None:
                selected_cols += [item.upper()
                                  for item in pass_cols_to_random_forest.split()]
            # add columns to compute the class in training
            selected_cols += ['CLASS_PERSON', 'CORRECT_REDSHIFT']
            self.config.set_option("model", "selected cols", " ".join(selected_cols))

            model_fits = model_config.getboolean("fits file")
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
            self.config.set_option("model", "filename", model_name)

            self.model = Model(self.config)
        # read trained model from file
        else:
            self.userprint("Loading model")
            if not os.path.exists(os.path.expandvars(model_filename)):
                message = (
                    "Could not read model file "
                    f"{os.path.expandvars(model_filename)} in original form "
                    f"{model_filename}")
                raise Error(message)
            t0 = time.time()
            if model_filename.endswith(".json"):
                model_config_file = model_filename.replace(".json", ".ini")
            else:
                model_config_file = model_filename.replace(".fits.gz", ".ini")
            print("#######################")
            print(model_config_file)
            print(os.path.expandvars(model_config_file))
            print(os.path.exists(os.path.expandvars(model_config_file)))
            print("#######################")
            model_config = Config(model_config_file)
            self.model = Model.from_file(model_config, model_filename)
            t1 = time.time()
            self.userprint(f"INFO: time elapsed to load model: {(t1-t0)/60.0} minutes")
            self.config.update_from_model(self.model.config)

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
            candidate_info = spectrum.metadata_values()
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
            wave = spectrum.wave
            flux = spectrum.flux
            ivar = spectrum.ivar
            metadata = spectrum.metadata_values()

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
        settings = self.model.settings
        self.lines = settings.get("LINES")
        self.try_lines = settings.get("TRY_LINES")
        self.z_precision = settings.get("Z_PRECISION")
        self.peakfind_width = settings.get("PEAKFIND_WIDTH")
        self.peakfind_sig = settings.get("PEAKFIND_SIG")
        self.pixels_as_metrics = settings.get("PIXELS_AS_METRICS")
        self.num_pixels = settings.get("NUM_PIXELS")

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

    def check_statistics(self):
        """ Check the statistics on the candidates classification """
        if self.mode != "test":
            raise Error("The function check_statistics is available in the "
                        f"test mode only. Detected mode is {self.mode}")
        stats_settings = self.config.get_section("stats")
        run_stats = stats_settings.getboolean("run stats")
        if run_stats is None:
            message = "In section [stats], variable 'run stats' is required"
            raise Error(message)
        if not run_stats:
            return
        # load truth table
        t0 = time.time()
        self.userprint("Loading quasar catalogue")
        quasar_catalogue_settings = self.config.get_section("quasar catalogue")
        qso_filename = quasar_catalogue_settings.get("filename")
        if qso_filename is not None:
            quasar_catalogue = deserialize(load_json(qso_filename))
            quasar_catalogue["LOADED"] = True
        else:
            quasar_catalogue = QuasarCatalogue(quasar_catalogue_settings).quasar_catalogue
            quasar_catalogue["LOADED"] = False
        t1 = time.time()
        self.userprint(f"INFO: time elapsed to load quasar catalogue: {(t1-t0)/60.0} minutes")

        # do the actual check
        t0 = time.time()
        self.userprint("Check statistics")
        probs_str = stats_settings.get("check probs")
        if probs_str is None:
            message = "In section [stats], variable 'check probs' is required"
            raise Error(message)
        probs = [float(item) for item in probs_str.split()]
        df = self.candidates
        self.userprint("\n---------------")
        self.userprint("step 1")
        self.find_completeness_purity(quasar_catalogue.reset_index(), df)
        for prob in probs:
            self.userprint("\n---------------")
            self.userprint("proba > {}".format(prob))
            self.find_completeness_purity(
                quasar_catalogue.reset_index(),
                df[(df["PROB"] > prob) & ~(df["DUPLICATED"]) &
                   (df["Z_CONF_PERSON"] == 3)],
            )
        t1 = time.time()
        self.userprint(f"INFO: time elapsed to check statistics: {(t1-t0)/60.0} minutes")

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
        t0 = time.time()
        self.userprint("Computing probabilities")
        self.candidates = self.model.compute_probability(self.candidates)
        if save:
            self.save_candidates()

        t1 = time.time()
        self.userprint(f"INFO: time elapsed to classify candidates: {(t1-t0)/60.0} minutes")

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
        settings = self.config.get_section("candidates")
        load_candidates = settings.getboolean("load candidates")
        if load_candidates is None:
            message = (
                "In section [candidates], variable 'load candidates' "
                "is required")
            raise Error(message)
        if load_candidates:
            self.userprint("Loading existing candidates")
            t0 = time.time()
            input_candidates = settings.get("input candidates")
            if input_candidates is None:
                input_candidates = self.name
            input_candidates_list = input_candidates.split()
            self.userprint(
                f"Found {len(input_candidates_list)} file from which to "
                "read candidates")
            self.userprint("Loading first candidate object")
            self.candidates = load_df(input_candidates_list[0])
            if len(input_candidates_list) > 1:
                self.userprint("Merging with the other candidate objects")
                self.merge(input_candidates_list[1:])
            t1 = time.time()
            self.userprint(f"INFO: time elapsed to load candidates: {(t1-t0)/60.0} minutes")

    def load_spectra(self):
        """ Load spectra and find candidates out of them"""
        if self.input_spectra is None:
            self.userprint(
                f"There are no files with spectra to be loaded")
            return
        self.userprint("Loading spectra")
        t0 = time.time()
        columns_candidates = []
        self.userprint(
            f"There are {len(self.input_spectra)} files with spectra to be loaded")
        for index, spectra_filename in enumerate(self.input_spectra):
            self.userprint(
                f"Loading spectra from {spectra_filename} "
                f"({index}/{len(self.input_spectra)})")
            t10 = time.time()
            spectra = Spectra.from_json(load_json(os.path.expandvars(spectra_filename)))
            if not isinstance(spectra, Spectra):
                raise Error("Invalid list of spectra")

            if index == 0:
                columns_candidates += spectra.spectra_list[0].metadata_names()

            # look for candidates
            self.userprint("Looking for candidates")
            self.find_candidates(spectra.spectra_list, columns_candidates)
            t11 = time.time()
            self.userprint(
                f"INFO: time elapsed to find candidates from {spectra_filename}:"
                f" {(t11-t10)/60.0} minutes")

        t1 = time.time()
        self.userprint(
            f"INFO: time elapsed to find candidates: {(t1-t0)/60.0} minutes")

        # convert to dataframe
        self.userprint("Converting candidates to dataframe")
        t0 = time.time()
        self.candidates_list_to_dataframe(columns_candidates)
        t1 = time.time()
        self.userprint(
            "INFO: time elapsed to convert candidates to dataframe: "
            f"{(t1-t0)/60.0} minutes")

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
                other = load_df(candidates_filename)

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

    def save_catalogue(self):
        """ Save the final catalogue as a fits file.

        Only non-duplicated candidates with probability greater or equal
        to self.prob_cut will be included in this catalogue.
        String columns with length greater than 15 characters might be truncated
        """
        if not self.save_catalogue_flag:
            return

        filename = self.name.replace(".fits", "_catalogue.fits")

        # filter data DataFrame
        data_frame = self.candidates[(~self.candidates["DUPLICATED"]) &
                                       (self.candidates["PROB"] >= self.prob_cut)]

        results = fitsio.FITS(filename, 'rw', clobber=True)
        names = list(data_frame.columns)
        cols = [
            np.array(data_frame[col].values, dtype=str)
            if data_frame[col].dtype == "object" else data_frame[col].values
            for col in data_frame.columns
        ]
        results.write(cols, names=names, extname="CANDIDATES")
        results.close()

    def train_model(self):
        """ Create a model instance and train it. Save the resulting model"""
        # consistency checks
        if self.mode != "training":
            raise Error("The function train_model is available in the " +
                        f"training mode only. Detected mode is {self.mode}")

        t0 = time.time()
        self.model.train(self.candidates)
        self.model.save_model()
        t1 = time.time()
        self.userprint(f"INFO: time elapsed to train model: {(t1-t0)/60.0} minutes")


if __name__ == '__main__':
    pass
