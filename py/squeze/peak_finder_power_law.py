"""
    SQUEzE
    ======

    This file provides a peak finder to be used by SQUEzE
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"

import itertools

from numba import njit
import numpy as np
from scipy import odr

from squeze.utils import quietprint, verboseprint

accepted_options = ["min significance"]

defaults = {
    # This variable sets the minimum sigmas of an outlier.
    "min significance": 2.0,
}


class PeakFinderPowerLaw:
    """ Create and manage the peak finder used by SQUEzE

    CLASS: PeakFinder
    PURPOSE: Create and manage the peak finder used by SQUEzE. This
    peak finder looks for peaks by fiting a power law to the continum of the
    spectra and locating the outliers. It also computes the significance of
    the peaks and filters the results according to their significances.
    """

    def __init__(self, config):
        """ Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        self.min_significance = config.getfloat("min significance")

    def find_peaks(self, spectrum):
        """ Find significant peaks in a given spectrum.

        Arguments
        ---------
        spectrum : Spectrum
        The spectrum where peaks are looked for

        Return
        ------
        An array with the position of the peaks
        """
        wavelength = spectrum.wave
        flux = spectrum.flux
        ivar = spectrum.ivar
        outliers_mask = np.ones_like(flux, dtype=bool)
        significances = np.zeros_like(flux)
        best_fit = np.array((0.0, 0.0))

        userprint=quietprint
        do_fit = True
        index = 0
        while do_fit:
            userprint(
                f"Running iteration {index}, {outliers_mask.sum()} pixels are "
                f"considered in the fit. Best fit is {best_fit}")
            index += 1
            if index > 100:
                userprint = verboseprint
            # fit power law
            new_outliers_mask, new_significances, new_best_fit = fit_power_law(
                wavelength,
                flux,
                ivar,
                outliers_mask,
                self.min_significance)

            if np.allclose(best_fit, new_best_fit, equal_nan=True):
                do_fit = False
            else:
                outliers_mask = new_outliers_mask
                significances = new_significances
                best_fit = new_best_fit

        userprint(
            f"Fit converged after {index} iterations. {outliers_mask.sum()} "
            f"pixels are considered in the fit. Best fit is {best_fit}")

        # if fit did not converge, return
        if any(best_fit == np.nan):
            return np.array([]), np.array([])

        # select only peaks
        peaks = select_peaks(
            wavelength, flux, outliers_mask, best_fit)

        # compress neighbouring pixels into a single pixel
        peak_indexs, peak_significances = compress(peaks, significances)

        # return
        return peak_indexs, peak_significances

def compress(peaks, significances):
    """Compress the neighbouring peak indexs into a single peak

    Arguments
    ---------
    peaks: array of bool
    Pixels mask, only pixels with peaks=True are considered to be peaks

    significances: array of float
    Significance of the outlier detection, computed as
    abs(flux - bestfit_flux)*np.sqrt(ivar)

    Return
    ------
    compressed_peak_indexs: array of int
    Array containing the indexs of the compressed peaks. Contiguous pixels
    are compressed by performing a weighted average according to their
    significance

    compressed_significances: array of float
    Significance of the compressed peaks. Computed by adding the significances
    of the relevant detections
    """
    #find peak indexs
    peak_indexs = np.array([index for index, peak in enumerate(peaks) if peak])

    # compress
    groups = list(group_contiguous(peak_indexs))
    compressed_peak_indexs = np.zeros(len(groups), dtype=int)
    compressed_significances = np.zeros_like(compressed_peak_indexs, dtype=float)
    for index, group in enumerate(groups):
        # single pixel
        if group[1] == group[0]:
            compressed_peak_indexs[index] = group[0]
            compressed_significances[index] = significances[group[0]]
        # grouped pixels
        else:
            aux = np.arange(group[0], group[1]+1, dtype=int)
            compressed_peak_indexs[index] = int(round(np.average(
                aux, weights=significances[aux]), 0))
            compressed_significances[index] = significances[aux].sum()


    return compressed_peak_indexs, compressed_significances

def fit_power_law(wavelength, flux, ivar, outliers_mask, min_significance):
    """ Perform a power-law fit, then compute the outliers

    Arguments
    ---------
    wavelength: array of float
    The wavelength

    flux: array of float
    The flux to fit

    ivar: array of float
    The inverse variance of the flux

    outliers_mask: array of bool
    Outliers mask, only pixels with outliers_mask=True are used in the fit

    min_significance: float
    The minimum significance to select outliers

    Return
    ------
    new_outliers_mask: array of bool
    Outliers mask, only pixels with outliers_mask=True are used in the fit

    significances: array of float
    Significance of the outlier detection, computed as
    abs(flux - bestfit_flux)*np.sqrt(ivar)

    best_fit: (float, float)
    The best fit power law amplitude and index
    """
    # do the actual fit
    data = odr.Data(wavelength, flux, we=ivar)
    odr_instance = odr.ODR(
        data,
        POWER_LAW_MODEL,
        beta0=(flux.mean(), 0.0))
    odr_instance.set_job(fit_type=0)
    fit_output = odr_instance.run()

    # figure out the outliers
    bestfit_flux = power_law(fit_output.beta, wavelength)
    significances = np.abs(flux-bestfit_flux)*np.sqrt(ivar)
    new_outliers_mask = np.zeros_like(outliers_mask)
    new_outliers_mask[np.where(
        significances > min_significance
    )] = True
    new_outliers_mask &= outliers_mask

    return new_outliers_mask, significances, fit_output.beta

def group_contiguous(data):
    """Group continuous elements together

    For example for data = [0, 1, 2, 3, 7, 8, 9] yield (0, 3) and (7, 9).
    To generate a list call is as
    `groups=list(group_contiguous(data))`

    Arguments
    ---------
    data: array of int
    Sorted sequence of integers to group
    """
    for _, group in itertools.groupby(enumerate(data), lambda x: x[1] - x[0]):
        group = list(group)
        yield group[0][1], group[-1][1]

@njit()
def power_law(parameters, x_data):
    """Power law function

    Arguments
    ---------
    parameters: (float, float)
    The amplitude and power law index

    x_data: array of float
    The points where to compute the power law
    """
    amplitude, power_law_index = parameters
    return amplitude*x_data**(-power_law_index)

@njit()
def select_peaks(wavelength, flux, outliers_mask, power_law_params):
    """ Select which of the outliers are peaks

    Arguments
    ---------
    wavelength: array of float
    The wavelength

    flux: array of float
    The flux to fit

    outliers_mask: array of bool
    Outliers mask, only pixels with outliers_mask=True are used in the fit

    power_law_params: (float, float)
    The power law amplitude and index

    Return
    ------
    peaks: array of bool
    Pixels mask, only pixels with peaks=True are considered to be peaks
    """
    # figure out the peaks
    bestfit_flux = power_law(power_law_params, wavelength)
    peaks = outliers_mask & (flux > bestfit_flux)

    return peaks

POWER_LAW_MODEL = odr.Model(power_law)
