"""
    SQUEzE
    ======

    This file implements the class DesiSpectrum, a specialization of Spectrum
    oriented to load spectra from DESI
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import numpy as np

from squeze.spectrum import Spectrum

# extra imports to read DESI data
try:
    from desispec.interpolation import resample_flux
except ImportError as error:
    print("Make sure you have 'desispec' installed before running on DESI data")
    raise error


def combine_bands(flux_dict, wave_dict, ivar_dict):
    """ Combine the different bands together

    Parameters
    ----------
    flux_dict : dict
    A dictionary with the flux arrays of the different reobserbations.
    Each key will contain an array with the fluxes in a given band.

    wave_dict : dict
    A dictionary with the wavalegth array.
    Each key will contain an array with the fluxes in a given band.

    ivar_dict : dict
    A dictionary with the ivar arrays of the different reobservations.
    Each key will contain an array with the ivars in a given band

    Returns
    -------
    flux : np.array
    Array containing the flux

    wave : np.array
    Array containing the wavelength

    ivar : np.array
    Array containing the inverse variance
    """
    # create empty combined arrays
    min_wave = np.min(np.array([np.min(flux) for flux in wave_dict.values()]))
    max_wave = np.max(np.array([np.max(flux) for flux in wave_dict.values()]))
    wave = np.linspace(min_wave, max_wave, 4000)
    ivar = np.zeros_like(wave, dtype=float)
    flux = np.zeros_like(wave, dtype=float)

    # populate arrays
    for band in flux_dict:
        ivar += resample_flux(wave, wave_dict[band], ivar_dict[band])
        flux += resample_flux(wave, wave_dict[band],
                              ivar_dict[band] * flux_dict[band])
    flux = flux / (ivar + (ivar == 0))

    return flux, wave, ivar


def combine_reobservations(flux_dict, ivar_dict):
    """ Combine the different reobservations into a single one

    Parameters
    ----------
    flux_dict: dict
    Dictionary containing the fluxes for all reobserbations

    ivar_dict: dict
    Dictionary containing the inverse variances for all reobserbations

    Returns
    -------
    flux_dict: dict
    Dictionary containing the flux with combined reobserbations

    ivar_dict: dict
    Dictionary containing the inverse variance with combined reobserbations
    """
    # loop over bands
    for band in flux_dict:
        flux = flux_dict[band]
        ivar = ivar_dict[band]

        # do weighted sum, masked elements are set to have 0 ivar
        sivar = ivar.sum(axis=0)
        flux = np.sum(flux * ivar, axis=0) / (sivar + (sivar == 0))
        ivar = sivar

        # update arrays
        flux_dict[band] = flux
        ivar_dict[band] = ivar

    return flux_dict, ivar_dict


def select_first_reobservation(flux_dict, ivar_dict):
    """ Discard all reobservations except for the first one

        Parameters
        ----------
        flux_dict: dict
        Dictionary containing the fluxes for all reobserbations

        ivar_dict: dict
        Dictionary containing the inverse variances for all reobserbations

        Returns
        -------
        flux_dict: dict
        Dictionary containing the flux for the first reobserbations

        ivar_dict: dict
        Dictionary containing the inverse variance for the first reobserbations
    """
    # loop over bands
    for band in flux_dict:
        flux_dict[band] = flux_dict[band][0, :]
        ivar_dict[band] = ivar_dict[band][0, :]
    return flux_dict, ivar_dict


class DesiSpectrum(Spectrum):
    """
        Load and format a DESI  spectrum to be digested by SQUEzE

        CLASS: DesiSpectrum
        PURPOSE: Load and format a DESI spectrum to be digested by
        SQUEzE
        """

    def __init__(self,
                 flux_dict,
                 wave_dict,
                 ivar_dict,
                 mask_dict,
                 metadata,
                 single_exp=False):
        """ Initialize class instance

            Parameters
            ----------
            flux_dict : dict
            A dictionary with the flux arrays of the different reobserbations.
            Each key will contain an array with the fluxes in a given band.

            wave_dict : dict
            A dictionary with the wavalegth array.
            Each key will contain an array with the fluxes in a given band.

            ivar_dict : dict
            A dictionary with the ivar arrays of the different reobservations.
            Each key will contain an array with the ivars in a given band

            mask_dict : dict
            A dictionary with the mask arrays of the different reobservations.
            Each key will contain an array with the mask in a given band.

            metadata : dict
            A dictionary with the spectral properties to be added in the
            catalogue. Must contain the key "specid".

            single_exp : bool
            If True, loads only the first reobservation. Otherwise combine them.
            """
        # mask inverse variance arrays
        for key in ivar_dict:
            ivar_dict[key][mask_dict.get(key)] = 0.0

        if single_exp:
            # keep only the first reobservation
            flux_dict, ivar_dict = select_first_reobservation(
                flux_dict, ivar_dict)
        else:
            # combine reobservations
            flux_dict, ivar_dict = combine_reobservations(flux_dict, ivar_dict)

        # combine bands
        flux, wave, ivar = combine_bands(flux_dict, wave_dict, ivar_dict)

        super().__init__(flux, ivar, wave, metadata)
