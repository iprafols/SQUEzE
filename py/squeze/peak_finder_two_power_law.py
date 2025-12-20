"""
    SQUEzE
    ======

    This file provides a peak finder to be used by SQUEzE
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"

import numpy as np
from scipy import odr

from squeze.peak_finder_power_law import (PeakFinderPowerLaw, compress)
from squeze.utils import quietprint, verboseprint
from squeze.numba_utils import njit

accepted_options = ["min significance"]

defaults = {
    # This variable sets the minimum sigmas of an outlier.
    "min significance": 2.0,
}


class PeakFinderTwoPowerLaw(PeakFinderPowerLaw):
    """ Create and manage the peak finder used by SQUEzE

    CLASS: PeakFinderTwoPowerLaw
    PURPOSE: Create and manage the peak finder used by SQUEzE. This
    peak finder looks for peaks by fitting a power law to the continuum of the
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
        self.min_significance = None
        super().__init__(config)

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
        The best fit parameters as a 4-element array containing:
        amplitude, power_law_index1, power_law_index2, and break_point.
        """
        wavelength = spectrum.wave
        flux = spectrum.flux
        ivar = spectrum.ivar
        outliers_mask = ivar != 0
        outliers_mask &= flux >= 0

        # initial guess
        initial_guess = np.array([
            flux.mean(),
            0.0,
            0.0,
            wavelength.mean(),
        ])

        # initial fit
        outliers_mask, differences, best_fit = fit_two_power_law(
            wavelength,
            flux,
            ivar,
            outliers_mask,
            self.min_significance,
            beta0=initial_guess)
        best_fit_chi2 = np.nansum(
            (flux[outliers_mask] -
             two_power_law(best_fit, wavelength[outliers_mask]))**2 *
            ivar[outliers_mask])

        # repeat ignoring masked pixels until convergence
        userprint = quietprint
        do_fit = True
        index = 0
        while do_fit:
            if index % 100 == 0 and index > 0:
                userprint = verboseprint
            else:
                userprint = quietprint
            userprint(
                f"Running iteration {index}, {outliers_mask.sum()} pixels are "
                f"considered in the fit. Best fit is {best_fit}")
            userprint(outliers_mask)
            userprint(f"Current chi2 = {best_fit_chi2}")

            # fit power law
            new_outliers_mask, new_differences, new_best_fit = fit_two_power_law(
                wavelength,
                flux,
                ivar,
                outliers_mask,
                self.min_significance,
                beta0=best_fit)
            new_best_fit_chi2 = np.nansum(
                (flux[new_outliers_mask] - two_power_law(
                    new_best_fit, wavelength[new_outliers_mask]))**2 *
                ivar[new_outliers_mask])

            userprint(
                f"Fit performed, {new_outliers_mask.sum()} pixels remaining. "
                f"New best fit is {new_best_fit}")
            userprint(f"New chi2 {new_best_fit_chi2}")
            userprint(f"New outliers mask {new_outliers_mask}")

            if new_best_fit_chi2 < best_fit_chi2:
                outliers_mask = new_outliers_mask
                differences = new_differences
                best_fit = new_best_fit
                best_fit_chi2 = new_best_fit_chi2
            else:
                do_fit = False

            index += 1

        userprint(
            f"Fit converged after {index} iterations. {outliers_mask.sum()} "
            f"pixels are considered in the fit. Best fit is {best_fit}")
        userprint(outliers_mask)
        userprint(f"Current chi2 = {best_fit_chi2}")

        # if fit did not converge, return
        if np.isnan(best_fit).any():
            return np.array([]), np.array([]), best_fit

        significances = np.abs(differences) * np.sqrt(ivar)

        # select only peaks
        peaks = select_peaks(wavelength, flux, outliers_mask, best_fit)

        # compress neighbouring pixels into a single pixel
        peak_indices, peak_significances = compress(peaks, significances)

        # return
        return peak_indices, peak_significances, best_fit


def fit_two_power_law(wavelength,
                      flux,
                      ivar,
                      outliers_mask,
                      min_significance,
                      beta0=None):
    """ Perform a two power-law fit, then compute the outliers

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

    best_fit: (float, float, float, float)
    The best fit parameters: the amplitude and power law indices for the
    first and second power laws, and the breaking point
    """
    # do the actual fit
    data = odr.Data(wavelength[outliers_mask],
                    flux[outliers_mask],
                    we=ivar[outliers_mask])
    if beta0 is None:
        beta0 = (
            flux[outliers_mask].mean(),
            0.0,
            0.0,
            wavelength[outliers_mask].mean(),
        )
    odr_instance = odr.ODR(data, TWO_POWER_LAW_MODEL, beta0=beta0)
    odr_instance.set_job(fit_type=0)
    fit_output = odr_instance.run()

    # figure out the outliers
    model = two_power_law(fit_output.beta, wavelength)
    differences = np.abs(flux - model) * np.sqrt(ivar)
    new_outliers_mask = np.zeros_like(outliers_mask, dtype=bool)
    new_outliers_mask[np.where(differences < min_significance)] = True
    new_outliers_mask &= outliers_mask

    return new_outliers_mask, differences, fit_output.beta


@njit()
def two_power_law(parameters, x_data):
    """Two power-laws function

    Arguments
    ---------
    parameters: (float, float, float, float)
    The amplitude, power law indices for the first and second power laws, and the breaking point

    x_data: array of float
    The points where to compute the power law

    Return
    ------
    result: array of float
    The two power-laws function
    """
    amplitude, power_law_index1, power_law_index2, break_point = parameters

    result = np.zeros_like(x_data)
    pos = x_data < break_point
    result[pos] = amplitude * (x_data[pos] / break_point)**(-power_law_index1)
    result[~pos] = amplitude * (x_data[~pos] / break_point)**(-power_law_index2)

    return result


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

    power_law_params: (float, float, float, float)
    The amplitude, power law indices for the first and second power laws, and the breaking point

    Return
    ------
    peaks: array of bool
    Pixels mask, only pixels with peaks=True are considered to be peaks
    """
    # figure out the peaks
    bestfit_flux = two_power_law(power_law_params, wavelength)
    peaks = ~outliers_mask & (flux > bestfit_flux)

    return peaks


TWO_POWER_LAW_MODEL = odr.Model(two_power_law)
