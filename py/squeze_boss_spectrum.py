"""
    SQUEzE
    ======

    This file implements the class Spectrum, that is used to make sure that the
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
from numpy.random import randn
from scipy.signal import medfilt
import astropy.io.fits as fits

from squeze_error import Error
from squeze_spectrum import Spectrum

class BossSpectrum(Spectrum):
    """
        Load and format a BOSS spectrum to be digested by SQUEzE

        CLASS: BossSpectrum
        PURPOSE: Load and format a BOSS spectrum to be digested by
        SQUEzE
        """
    def __init__(self, spectrum_file, metadata, mask, smoothing=0, noise_increase=1):
        """ Initialize class instance

            Parameters
            ----------
            spectrum_file : str
            Name of the fits files containing the spectrum

            metadata : dict
            A dictionary with the metadata. Keys should be strings
            
            mask : (np.array, float)
            A tuple containing the array of the wavelengths to mask and the margin
            used in the masking. Wavelengths separated to wavelength given in the array
            by less than the margin will be masked

            smoothing : int - Default: 0
            Number of pixels in the smoothing kernel. Negative values are ignored

            noise_increase : int, >0 - Default: 1
            Adds noise to the spectrum by adding a gaussian random number of width
            equal to the (noise_amount-1) times the given variance. Then increase the
            variance by a factor of sqrt(noise_amount)
            """
        # check that "specid" is present in metadata
        if "specid" not in metadata.keys():
            raise Error("""The property "specid" must be present in metadata""")

        # open fits file
        spectrum_hdu = fits.open(spectrum_file)
        
        # compute sky mask
        self._wave = 10**spectrum_hdu[1].data["loglam"].copy()
        masklambda = mask[0]
        margin = mask[1]
        self.__find_skymask(masklambda, margin)
        
        # store the wavelength, flux and inverse variance as masked arrays
        self._wave = np.ma.array(self._wave, mask=self.__skymask)
        self._flux = np.ma.array(spectrum_hdu[1].data["flux"].copy(),
                                 mask=self.__skymask)
        self._ivar = np.ma.array(spectrum_hdu[1].data["ivar"].copy(),
                                 mask=self.__skymask)
        self._metadata = metadata
        if noise_increase > 1:
            self.__add_noise(noise_increase)
        if smoothing > 0:
            self._flux = self.smooth(smoothing)
            self._ivar = self.smooth_ivar(smoothing)
        del spectrum_hdu[1].data
        spectrum_hdu.close()

    def __add_noise(self, noise_amount):
        """ Adds noise to the spectrum by adding a gaussian random number of width
            equal to the (noise_amount-1) times the given variance. Then increase the
            variance by a factor of sqrt(noise_amount)
            """
        var = 1./self._ivar
        var[np.where(var == np.inf)] = 0.
        self._ivar = self._ivar/np.sqrt(noise_amount)
        self._flux = self._flux + (noise_amount - 1.)*var*randn(self._flux.size)

    def __find_skymask(self, masklambda, margin):
        """ Compute the sky mask according to a set of wavelengths and a margin.
            Keep pixels in each spectrum which meet the following requirement
            fabs(1e4*log10(lambda/maskLambda)) > margin
            Sky mask is 0 if pixel doesn't have to be masked and 1 otherwise
            """
        self.__skymask = np.zeros_like(self._wave)
        for wave in masklambda:
            self.__skymask[np.where(np.abs(np.log10(self._wave/wave)) <= margin)] = 1

if __name__ == "__main__":
    pass
