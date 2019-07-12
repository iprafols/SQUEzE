"""
    SQUEzE
    ======

    This file implements the class DesiSpectrum, that is used to make sure that the
    passed spectrum have the required properties for SQUEzE to be able to run
    properly.
    To run SQUEzE on a given dataset, first create a class to load the spectra.
    This class should inherit from Spectrum. See a "fill the spots" example in
    squeze_my_spectrum.py.
    Alternatively, for simple cases, the SimpleSpectrum class defined in
    squeze_simple_spectrum.py may be used.
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import numpy as np

from desispec.interpolation import resample_flux

from squeze_error import Error
from squeze_spectrum import Spectrum

class DesiSpectrum(Spectrum):
    """
        Load and format a DESI  spectrum to be digested by SQUEzE

        CLASS: DesiSpectrum
        PURPOSE: Load and format a DESI spectrum to be digested by
        SQUEzE
        """
    def __init__(self, flux, wave, ivar, mask, metadata):
        """ Initialize class instance

            Parameters
            ----------
            flux : dict
            A dictionary with the flux arrays of the different reobserbations.
            Each key will contain an array with the fluxes in a given band.

            wave : dict
            A dictionary with the wavalegth array.
            Each key will contain an array with the fluxes in a given band. 

            ivar : dict
            A dictionary with the ivar arrays of the different reobservations.
            Each key will contain an array with the ivars in a given band

            mask : dict
            A dictionary with the mask arrays of the different reobservations.
            Each key will contain an array with the mask in a given band.

            metadata : dict
            A dictionary with the spectral properties to be added in the 
            catalogue. Must contain the key "specid".
            """

        # variables to store the information initially they are dictionaries
        # but they will be np.ndarrays by the end of __init__
        self._flux = flux
        self._wave = wave
        self._ivar = ivar
        self._mask = mask # temporary variable
        # keep metadata
        self._metadata = metadata
        
        # combine reobservations
        self.__combine_reobservations()
        del self._mask

        # combine bands
        self.__combine_bands()

    def __combine_bands(self):
        """ Combine the different bands together"""
        # create empty combined arrays
        min_wave = np.min(np.array([np.min(flux) for flux in self._wave.values()]))
        max_wave = np.max(np.array([np.max(flux) for flux in self._wave.values()]))
        wave = np.linspace(min_wave,max_wave,4000)
        ivar = np.zeros_like(wave, dtype=float)
        flux = np.zeros_like(wave, dtype=float)

        # populate arrays
        for band in self._flux.keys():
             ivar += resample_flux(wave,self._wave[band],self._ivar[band])
             flux += resample_flux(wave,self._wave[band],
                                   self._ivar[band]*self._flux[band])
        flux = flux/(ivar+(ivar==0))

        # update arrays
        self._wave = wave
        self._flux = flux
        self._ivar = ivar


    def __combine_reobservations(self):
        """ Combine the different reobservations into a single one"""
        # loop over bands
        for band in self._flux.keys():
            flux = self._flux[band]
            ivar = self._ivar[band]
            mask = self._mask[band]

            # do weighted sum, masked elements are set to have 0 ivar
            sivar = (ivar*(mask==0)).sum(axis=0)
            flux = np.sum(flux*ivar*(mask==0),axis=0)/(sivar+(sivar==0))
            ivar = sivar

            # update arrays
            self._flux[band] = flux
            self._ivar[band] = ivar