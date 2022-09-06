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

import numpy as np
from numba import prange, jit, vectorize
import pandas as pd
import fitsio
from astropy.table import Table

from squeze.common_functions import verboseprint
from squeze.error import Error
from squeze.model import Model
from squeze.peak_finder import PeakFinder
from squeze.defaults import MAX_CANDIDATES_TO_CONVERT
from squeze.defaults import LINES
from squeze.defaults import TRY_LINES
from squeze.defaults import RANDOM_FOREST_OPTIONS
from squeze.defaults import RANDOM_STATE
from squeze.defaults import PASS_COLS_TO_RF
from squeze.defaults import Z_PRECISION
from squeze.defaults import PEAKFIND_WIDTH
from squeze.defaults import PEAKFIND_SIG

# extra imports for plotting function
PLOTTING_ERROR = None
try:
    import matplotlib.pyplot as plt
except ImportError as error:
    PLOTTING_ERROR = error


@jit(nopython=True)
def compute_line_ratios(wave, flux, ivar, peak_indexs, significances, try_lines,
                        lines):
    """Compute the line ratios for the specified lines for a given spectrum

        See equations 1 to 3 of Perez-Rafols et al. 2020 for a detailed
        description of the metrics

        Parameters
        ----------
        wave : array of float
        The spectrum wavelength

        flux : array of float
        The spectrum flux

        ivar : array of float
        The spectrum inverse variance

        peak indexs : array of int
        The indexes on the wave array where peaks are found

        significances : array of float
        The significances of the peaks

        try_lines : array of ints
        Indexes of the lines to be considered as originators of the peaks

        lines : array of arrays of floats
        Information of the lines where the ratios need to be computed. Each
        of the arrays must be organised as follows:
        0 - wavelength of the line
        1 - wavelength of the start of the peak interval
        2 - wavelength of the end of the peak interval
        3 - wavelength of the start of the blue interval
        4 - wavelength of the end of the blue interval
        5 - wavelength of the start of the red interval
        6 - wavelength of the end of the red interval

        Returns
        -------
        new_candidates : list
        Each element of the list contains the ratios of the lines, the trial
        redshift, the significance of the line and the index of the line
        assumed to be originating the emission peak

    """
    new_candidates = []
    # pylint: disable=not-an-iterable
    # prange is the numba equivalent to range
    for index1 in prange(peak_indexs.size):
        for index2 in prange(len(try_lines)):
            # compute redshift
            # 0=WAVE
            z_try = wave[peak_indexs[index1]]
            z_try = z_try / lines[try_lines[index2], 0]
            z_try = z_try - 1.0
            if z_try < 0.0:
                continue
            oneplusz = (1.0 + z_try)

            candidate_info = []

            # compute peak ratio for the different lines
            for index3 in prange(lines.shape[0]):
                # compute intervals
                # 1=START, 2=END
                pix_peak = np.where((wave >= oneplusz * lines[index3, 1]) &
                                    (wave <= oneplusz * lines[index3, 2]))[0]
                # 3=BLUE_START, 4=BLUE_END
                pix_blue = np.where((wave >= oneplusz * lines[index3, 3]) &
                                    (wave <= oneplusz * lines[index3, 4]))[0]
                # 5=RED_START, 6=RED_END
                pix_red = np.where((wave >= oneplusz * lines[index3, 5]) &
                                   (wave <= oneplusz * lines[index3, 6]))[0]

                # compute peak and continuum values
                compute_ratio = True
                if ((pix_blue.size == 0) or (pix_peak.size == 0) or
                    (pix_red.size == 0) or
                    (pix_blue.size < pix_peak.size // 2) or
                    (pix_red.size < pix_peak.size // 2)):
                    compute_ratio = False
                else:
                    peak = np.mean(flux[pix_peak])
                    cont_red = np.mean(flux[pix_red])
                    cont_blue = np.mean(flux[pix_blue])
                    cont_red_and_blue = cont_red + cont_blue
                    if cont_red_and_blue == 0.0:
                        compute_ratio = False

                # compute ratios
                if compute_ratio:
                    peak_ivar_sum = ivar[pix_peak].sum()
                    if peak_ivar_sum == 0.0:
                        peak_err_squared = np.nan
                    else:
                        peak_err_squared = 1.0 / peak_ivar_sum
                    blue_ivar_sum = ivar[pix_blue].sum()
                    red_ivar_sum = ivar[pix_red].sum()
                    if blue_ivar_sum == 0.0 or red_ivar_sum == 0.0:
                        cont_err_squared = np.nan
                    else:
                        cont_err_squared = (1.0 / blue_ivar_sum +
                                            1.0 / red_ivar_sum) / 4.0

                    ratio = 2.0 * peak / cont_red_and_blue
                    ratio2 = abs((cont_red - cont_blue) / cont_red_and_blue)
                    err_ratio = sqrt(4. * peak_err_squared + ratio * ratio *
                                     cont_err_squared) / abs(cont_red_and_blue)
                    ratio_sn = (ratio - 1.0) / err_ratio
                else:
                    ratio = np.nan
                    ratio2 = np.nan
                    ratio_sn = np.nan

                candidate_info.append(ratio)
                candidate_info.append(ratio_sn)
                candidate_info.append(ratio2)

            candidate_info.append(z_try)
            candidate_info.append(significances[index1])
            candidate_info.append(float(index2))

            # add candidate to the list
            new_candidates.append(candidate_info)

    return new_candidates


@jit(nopython=True)
def compute_pixel_metrics(wave, flux, ivar, peak_indexs, num_pixels, try_lines,
                          lines):
    """Compute pixel metrics.

        Basically keep the pixel close to each peak as a new set of metrics.
        For each pixel, keep the flux and the ivar. Use NaN for no coverage.

        Parameters
        ----------
        wave : array of float
        The spectrum wavelength

        flux : array of float
        The spectrum flux

        ivar : array of float
        The spectrum inverse variance

        peak indexs : array of int
        The indexes on the wave array where peaks are found

        num_pixels : int
        The number of pixels to keep to each side of the peak

        try_lines : array of ints
        Indexes of the lines to be considered as originators of the peaks

        lines : array of arrays of floats
        Information of the lines where the ratios need to be computed. Each
        of the arrays must be organised as follows:
        0 - wavelength of the line
        1 - wavelength of the start of the peak interval
        2 - wavelength of the end of the peak interval
        3 - wavelength of the start of the blue interval
        4 - wavelength of the end of the blue interval
        5 - wavelength of the start of the red interval
        6 - wavelength of the end of the red interval

        Returns
        -------
        pixel_metrics : list
        Each element of the list contains the pixel metrics associated to each
        peak.

    """
    pixel_metrics = []
    # pylint: disable=not-an-iterable
    # prange is the numba equivalent to range
    for index1 in prange(peak_indexs.size):
        #for peak_index, significance in zip(peak_indexs, significances):

        candidate_info = [float(x) for x in range(0)]

        # compute pixel metrics
        peak_index = peak_indexs[index1]
        candidate_info = []
        for index2 in prange(-num_pixels, 0):
            if peak_index + index2 < 0:
                candidate_info.append(np.nan)
                candidate_info.append(np.nan)
            else:
                candidate_info.append(flux[peak_index + index2])
                candidate_info.append(ivar[peak_index + index2])
        for index2 in prange(0, num_pixels):
            if peak_index + index2 >= flux.size:
                candidate_info.append(np.nan)
                candidate_info.append(np.nan)
            else:
                candidate_info.append(flux[peak_index + index2])
                candidate_info.append(ivar[peak_index + index2])

        # pylint: disable=not-an-iterable
        # prange is the numba equivalent to range
        for index2 in prange(len(try_lines)):
            #for try_line in try_lines:

            # compute redshift
            # 0=WAVE
            z_try = wave[peak_indexs[index1]]
            z_try = z_try / lines[try_lines[index2], 0]
            z_try = z_try - 1.0
            if z_try < 0.0:
                continue

            # add pixel metrics to the list once per each candidate
            pixel_metrics.append(candidate_info)

    return pixel_metrics


def convert_dtype(dtype):
    """Convert datatype "O" to "15" to save in fits file.
    Other types are ignored return as they are"""
    if dtype == "O":
        return "15A"
    return dtype


@vectorize
def compute_is_correct(correct_redshift, class_person):
    """ Returns True if a candidate is a true quasar and False otherwise.
        A true candidate is defined as a candidate having an absolute value
        of Delta_z is lower or equal than self.z_precision.

        Parameters
        ----------
        correct_redshift : array of bool
        Array specifying if the redhsift is correct

        class_person : array of int
        Array specifying the actual classification. 3 and 30 stand for quasars
        and BAL quasars respectively. 1 stands for stars and 4 stands for galaxies.

        Returns
        -------
        correct : array of bool
        For each element in the arrays, returns True if the candidate is a
        quasar and has the correct redshift assign to it
        """
    correct = bool(correct_redshift and class_person in [3, 30])
    return correct


@vectorize
def compute_is_correct_redshift(delta_z, class_person, z_precision):
    """ Returns True if a candidate has a correct redshift and False otherwise.
        A candidate is assumed to have a correct redshift if it has an absolute
        value of Delta_z lower than or equal to z_precision.
        If the object is a star (class_person = 1), then return False.

        Parameters
        ----------
        delta_z : array of float
        Differences between the trial redshift (Z_TRY) and the true redshift
        (Z_TRUE)

        class_person : array of int
        Array specifying the actual classification. 3 and 30 stand for quasars
        and BAL quasars respectively. 1 stands for stars and 4 stands for galaxies.

        z_precision : float
        Tolerance with which two redshifts are considerd equal

        Returns
        -------
        correct_redshift : array of bool
        For each element, returns True if the candidate is not a star and the
        trial redshift is equal to the true redshift.
        """
    correct_redshift = bool((class_person != 1) and
                            (-z_precision <= delta_z <= z_precision))
    return correct_redshift


@jit(nopython=True)
def compute_is_line(is_correct, class_person, assumed_line_index, z_true, z_try,
                    z_precision, lines):
    """ Returns True if the candidates corresponds to a valid quasar emission
        line and False otherwise. A quasar line is defined as a candidate whose
        trial redshift is correct or where any of the redshifts obtained
        assuming that the emission line is generated by any of the lines is
        equal to the true redshift

        Parameters
        ----------
        is_correct : array of bool
        Array specifying if the candidates with a correct trial redshift

        class_person : array of int
        Array specifying the actual classification. 3 and 30 stand for quasars
        and BAL quasars respectively. 1 stands for stars and 4 stands for galaxies.

        assumed_line_index : int
        Index of the line considered as originating the peaks

        z_true : float
        True redshift

        z_try : float
        Trial redshift

        z_precision : float
        Tolerance with which two redshifts are considerd equal

        lines : array of arrays of floats
        Information of the lines that can be originating the emission line peaks.
        Each of the arrays must be organised as follows:
        0 - wavelength of the line

        Returns
        -------
        is_line : array of bool
        For each element, returns True if thr candidate is a quasar line and
        False otherwise.
        """
    is_line = np.zeros_like(is_correct)

    # correct identification
    # pylint: disable=not-an-iterable
    # prange is the numba equivalent to range
    for index1 in prange(is_correct.size):
        if is_correct[index1]:
            is_line[index1] = True
        # not a quasar
        elif not ((class_person[index1] == 3) or (class_person[index1] == 30)):
            continue
        # not a peak
        elif assumed_line_index[index1] == -1:
            continue
        else:
            for index2 in prange(lines.shape[0]):
                #for line in self.lines.index:
                if index2 == assumed_line_index[index1]:
                    continue
                # 0=WAVE
                z_try_line = (lines[assumed_line_index[index1]][0] /
                              lines[index2][0]) * (1 + z_try[index1]) - 1
                if ((z_try_line - z_true[index1] <= z_precision) and
                    (z_try_line - z_true[index1] >= -z_precision)):
                    is_line[index1] = True
    return is_line


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
    def __init__(self,
                 lines_settings=(LINES, TRY_LINES),
                 z_precision=Z_PRECISION,
                 mode="operation",
                 name="SQUEzE_candidates.fits.gz",
                 peakfind=(PEAKFIND_WIDTH, PEAKFIND_SIG),
                 pixel_as_metrics=(False, 0),
                 model=None,
                 model_options=(RANDOM_FOREST_OPTIONS, RANDOM_STATE,
                                PASS_COLS_TO_RF),
                 userprint=verboseprint):
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

            pixel_as_metrics: (bool, int) - Default: (False, 0)
            The first boolean specifies whether to keep pixel information as
            metrics and the second int specifies the number of pixels to each
            side of the peak to keep. If set, for each pixel, keep the flux and
            ivar as metrics.

            model : Model or None  - Default: None
            Instance of the Model class defined in squeze_model or None.
            In test and operation mode, it is supposed
            to be the quasar model to construct the catalogue. In training mode,
            it is supposed to be None initially, and the model will be trained
            and given as an output of the code.

            model_options : (dict, int, list or None)
                            - Defaut: (RANDOM_FOREST_OPTIONS, RANDOM_STATE, None)
            The first dictionary sets the options to be passed to the random forest
            cosntructor. If high-low split of the training is desired, the
            dictionary must contain the entries "high" and "low", and the
            corresponding values must be dictionaries with the options for each
            of the classifiers. The second int is the random state passed to the
            random forest classifiers. The third list contains columns to be passed
            to the random forest classifiers (None for no columns). In training
            mode, they're passed to the model instance before training.
            Otherwise it's ignored.

            userprint : function - Default: verboseprint
            Print function to use
            """
        if mode in ["training", "test", "operation", "candidates", "merge"]:
            self.mode = mode
        else:
            raise Error("Invalid mode")

        if name.endswith(".fits.gz") or name.endswith(".fits"):
            self.name = name
        else:
            message = (
                "Candidates name should have .fits or .fits.gz extensions."
                f"Given name was {name}")
            raise Error(message)

        # printing function
        self.userprint = userprint

        # initialize empty catalogue
        self.candidates_list = []
        self.candidates = None

        # main settings
        self.lines = lines_settings[0]
        self.try_lines = lines_settings[1]
        self.z_precision = z_precision

        # options to be passed to the peak finder
        self.peakfind_width = peakfind[0]
        self.peakfind_sig = peakfind[1]

        # pixel metrics
        self.pixels_as_metrics = pixel_as_metrics[0]
        self.num_pixels = pixel_as_metrics[1]

        # model
        if model is None:
            self.model = None
        else:
            self.model = model
            self.__load_model_settings()
        self.model_options = model_options

        # initialize peak finder
        self.peak_finder = PeakFinder(self.peakfind_width,
                                        self.peakfind_sig)

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
            self.mode = mode
        else:
            raise Error("Invalid mode")

    def candidates_list_to_dataframe(self, columns_candidates, save=True):
        """ Format existing candidates list into a dataframe

            Parameters
            ----------
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
        """ Create a model instance and train it. Save the resulting model"""
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
        """ Find candidates for a given set of spectra, then integrate them in the
            candidates catalogue and save the new version of the catalogue.

            Parameters
            ----------
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
        """
            Given a DataFrame with candidates and another one with the catalogued
            quasars, compute the completeness and the purity. Upon error, return
            np.nan

            Parameters
            ----------
            quasars_data_frame : string
            DataFrame containing the quasar catalogue. The quasars must contain
            the column "specid" to identify the spectrum.

            data_frame : pd.DataFrame - Default: self.candidates
            DataFrame where the percentile will be computed. Must contain the
            columns "is_correct" and "specid".

            Returns
            -------
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

            Parameters
            ----------
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
        """
            Merge self.candidates with another candidates object

            Parameters
            ----------
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
        """ Save the final catalogue as a fits file. Only non-duplicated
            candidates with probability greater or equal to prob_cut will
            be included in this catalogue.
            String columns with length greater than 15
            characters might be truncated

            Parameters
            ----------
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
