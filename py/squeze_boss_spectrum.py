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
    def __init__(self, spectrum_file, metadata, smoothing=0, double_noise=False):
        """ Initialize class instance

            Parameters
            ----------
            spectrum_file : str
            Name of the fits files containing the spectrum

            metadata : dict
            A dictionary with the metadata. Keys should be strings

            smoothing : int - Default: 0
            Number of pixels in the smoothing kernel. Negative values are ignored

            double_noise : bool - Default: False
            Doubles the noise of the spectra
            """
        # check that "specid" is present in metadata
        if "specid" not in metadata.keys():
            raise Error("""The property "specid" must be present in metadata""")

        spectrum_hdu = fits.open(spectrum_file)
        self._flux = spectrum_hdu[1].data["flux"].copy()
        self._wave = 10**spectrum_hdu[1].data["loglam"].copy()
        self._ivar = spectrum_hdu[1].data["ivar"].copy()
        self._metadata = metadata
        if smoothing > 0:
            self.__smooth(smoothing)
        if double_noise:
            self.__double_noise()
        del spectrum_hdu[1].data
        spectrum_hdu.close()

    def __double_noise(self):
        """ Doubles the noise of the spectrum by adding a gaussian random number of width
            equal to the given variance. Then increase the variance by a factor of sqrt(2)
            """
        var = 1/self.ivar()
        var[np.where(var == np.inf)] = 0
        self._flux = self._flux + var*randn(self._flux.size)
        self._ivar = self._ivar/np.sqrt(2)

    def __smooth(self, smoothing):
        """ Smooth the flux of the spectrum

            Parameters
            ----------
            smoothing : int
            Number of pixels in the smoothing kernel
            """
        # if smoothing is even, add 1
        if smoothing % 2 == 0:
            smoothing += 1

        self._flux = medfilt(self._flux, smoothing)
        self._ivar = medfilt(self._ivar, smoothing)



if __name__ == "__main__":
    pass
