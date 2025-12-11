"""
    SQUEzE
    ======

    This file provides a peak finder to be used by SQUEzE
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"

import itertools

import numpy as np

from squeze.numba_utils import njit
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
        peak_indices: array of int
        An array with the position of the peaks

        peak_significances: array of float
        An array with the significance of the peaks

        best_fit: array of float
        The best fit parameters: the amplitude and power law index
        """
        wavelength = spectrum.wave
        flux = spectrum.flux
        ivar = spectrum.ivar
        outliers_mask = np.ones_like(flux, dtype=bool)
        significances = np.zeros_like(flux)
        best_fit = np.array((0.0, 0.0))

        userprint = quietprint
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
                wavelength, flux, ivar, outliers_mask, self.min_significance)

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
            return np.array([]), np.array([]), best_fit

        # select only peaks
        peaks = select_peaks(wavelength, flux, outliers_mask, best_fit)

        # compress neighbouring pixels into a single pixel
        peak_indices, peak_significances = compress(peaks, significances)

        # return
        return peak_indices, peak_significances, best_fit

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
    peak_indices = np.array([index for index, peak in enumerate(peaks) if peak])

    # compress
    groups = list(group_contiguous(peak_indices))
    compressed_peak_indexs = np.zeros(len(groups), dtype=int)
    compressed_significances = np.zeros_like(compressed_peak_indexs,
                                             dtype=float)
    for index, group in enumerate(groups):
        # single pixel
        if group[1] == group[0]:
            compressed_peak_indexs[index] = group[0]
            compressed_significances[index] = significances[group[0]]
        # grouped pixels
        else:
            aux = np.arange(group[0], group[1] + 1, dtype=int)
            compressed_peak_indexs[index] = int(
                round(np.average(aux, weights=significances[aux]), 0))
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
    # Perform log-linear fit
    best_fit = fit_power_law_log_linear(wavelength, flux, ivar, outliers_mask)

    # Check if fit succeeded
    if np.isnan(best_fit).any() or best_fit[0] <= 0:
        # Return default values if fit failed
        return outliers_mask, np.zeros_like(flux), np.array([flux.mean(), 0.0])

    # figure out the outliers
    bestfit_flux = power_law(best_fit, wavelength)
    significances = np.abs(flux - bestfit_flux) * np.sqrt(ivar)
    new_outliers_mask = np.zeros_like(outliers_mask, dtype=bool)
    new_outliers_mask[np.where(significances < min_significance)] = True
    new_outliers_mask &= outliers_mask

    return new_outliers_mask, significances, best_fit


def fit_power_law_log_linear(wavelength, flux, ivar, outliers_mask):
    """
    Fit power law using log-linear regression.
    This is more robust than ODR fitting.
    
    Arguments
    ---------
    wavelength: array of float
    The wavelength
    
    flux: array of float
    The flux to fit
    
    ivar: array of float
    The inverse variance of the flux
    
    outliers_mask: array of bool
    Mask indicating which pixels to include in the fit
    
    Return
    ------
    best_fit: array of float
    The best fit parameters [amplitude, power_index]
    """
    # Apply outliers mask and remove invalid values
    mask = outliers_mask & (flux > 0) & (
        ivar > 0) & np.isfinite(flux) & np.isfinite(ivar)

    if np.sum(mask) < 3:  # Need at least 3 points for a good fit
        return np.array([flux.mean() if len(flux) > 0 else 1.0, 0.0])

    log_wave = np.log(wavelength[mask])
    log_flux = np.log(flux[mask])
    weights = ivar[mask]

    try:
        # Perform weighted linear regression: log(flux) = log(A) - alpha * log(wave)
        # Use weighted least squares
        fit_x = np.column_stack([np.ones(len(log_wave)), -log_wave
                                ]) * weights[:, np.newaxis]
        fit_y = log_flux * weights

        # Solve weighted least squares
        coeffs = np.linalg.lstsq(fit_x, fit_y, rcond=None)[0]
        log_amplitude = coeffs[0]
        power_index = coeffs[1]

        # Convert back from log space
        amplitude = np.exp(log_amplitude)

        # Sanity check the results
        if not np.isfinite(amplitude) or not np.isfinite(
                power_index) or amplitude <= 0:
            return np.array([flux[mask].mean(), 0.0])

        return np.array([amplitude, power_index])

    except Exception:
        # If anything goes wrong, return a reasonable default
        return np.array(
            [flux[mask].mean() if len(flux[mask]) > 0 else 1.0, 0.0])


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
    return amplitude * x_data**(-power_law_index)


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
    peaks = ~outliers_mask & (flux > bestfit_flux)

    return peaks
