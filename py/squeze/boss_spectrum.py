"""
    SQUEzE
    ======

    This file implements the class BossSpectrum, a specialization of Spectrum
    oriented to load spectra from BOSS
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import numpy as np
from numpy.random import randn
import fitsio

from squeze.error import Error
from squeze.spectrum import Spectrum

class BossSpectrum(Spectrum):
    """
        Load and format a BOSS spectrum to be digested by SQUEzE

        CLASS: BossSpectrum
        PURPOSE: Load and format a BOSS spectrum to be digested by
        SQUEzE
        """
    def __init__(self, spectrum_file, metadata, sky_mask, mask_jpas=False,
                 mask_jpas_alt=False, rebin_pixels_width=0, extend_pixels=0,
                 noise_increase=1, forbidden_wavelenghts=None):
        """ Initialize class instance

            Parameters
            ----------
            spectrum_file : str
            Name of the fits files containing the spectrum

            metadata : dict
            A dictionary with the metadata. Keys should be strings

            sky_mask : (np.array, float)
            A tuple containing the array of the wavelengths to mask and the margin
            used in the masking. Wavelengths separated to wavelength given in the array
            by less than the margin will be masked

            mask_jpas : bool - Default: False
            If set, mask pixels corresponding to filters in trays T3 and T4. Only works if
            the bin size is 100 Angstroms

            mask_jpas_alt : bool - Default: False
            If set, mask pixels corresponding to filters in trays T3* and T4. Only works if
            the bin size is 100 Angstroms

            rebin_pixels_width : float, >0 - Default: 0
            Width of the new pixel (in Angstroms)

            extend_pixels : float, >0 - Default: 0
            Pixel overlap region (in Angstroms)

            noise_increase : int, >0 - Default: 1
            Adds noise to the spectrum by adding a gaussian random number of width
            equal to the (noise_amount-1) times the given variance. Then increase the
            variance by a factor of sqrt(noise_amount)

            forbidden_wavelengths : list of tuples or None - Default: None
            If not None, a list containing tuples specifying ranges of wavelengths that will
            be masked (both ends included). Each tuple must contain the initial and final range
            of wavelenghts. This is intended to be complementary to the sky mask to limit the
            wavelength coverage, and hard cuts will be applied
            """
        # check that "specid" is present in metadata
        if "SPECID" not in metadata.keys():
            raise Error("""The property "SPECID" must be present in metadata""")

        # open fits file
        spectrum_hdul = fitsio.FITS(spectrum_file)

        # compute sky mask
        self._wave = 10**spectrum_hdul[1]["LOGLAM"][:]
        masklambda = sky_mask[0]
        margin = sky_mask[1]
        self.__skymask = None
        self.__find_skymask(masklambda, margin)

        # mask forbidden lines
        if forbidden_wavelenghts is not None:
            self.__filter_wavelengths(forbidden_wavelenghts)

        # store the wavelength, flux and inverse variance as arrays
        # The 1.0 mulitplying is added to change type from >4f to np.float
        # this is required by numba later on
        self._flux = 1.0*spectrum_hdul[1]["FLUX"][:]
        self._ivar = 1.0*spectrum_hdul[1]["IVAR"][:]
        # mask pixels
        self._ivar[self.__skymask] = 0.0
        self._metadata = metadata
        if noise_increase > 1:
            self.__add_noise(noise_increase)
        if rebin_pixels_width > 0:
            self._flux, self._ivar, self._wave = self.rebin(rebin_pixels_width,
                                                            extend_pixels=extend_pixels)

        # JPAS mask
        if mask_jpas:
            pos = np.where(~((np.isin(self._wave, [3900, 4000, 4300, 4400, 4700, 4800, 5100,
                                                 5200])) | (self._wave >= 7300)))
            self._wave = self._wave[pos].copy()
            self._ivar = self._ivar[pos].copy()
            self._flux = self._flux[pos].copy()
        elif mask_jpas_alt:
            pos = np.where(~((np.isin(self._wave, [3800, 4000, 4200, 4400, 4600, 4800, 5000,
                                                   5200])) | (self._wave >= 7300)))
            self._wave = self._wave[pos].copy()
            self._ivar = self._ivar[pos].copy()
            self._flux = self._flux[pos].copy()

        spectrum_hdul.close()

    def __add_noise(self, noise_amount):
        """ Adds noise to the spectrum by adding a gaussian random number of width
            equal to the (noise_amount-1) times the given variance. Then increase the
            variance by a factor noise_amount
            """
        sigma = 1./np.sqrt(self._ivar)
        sigma[np.where(sigma == np.inf)] = 0.
        self._ivar = self._ivar/noise_amount
        self._flux = self._flux + (noise_amount - 1.)*sigma*randn(self._flux.size)

    def __find_skymask(self, masklambda, margin):
        """ Compute the sky mask according to a set of wavelengths and a margin.
            Keep pixels in each spectrum which meet the following requirement
            fabs(1e4*log10(lambda/maskLambda)) > margin
            Sky mask is 0 if pixel doesn't have to be masked and 1 otherwise
            """
        self.__skymask = np.zeros_like(self._wave, dtype=bool)
        for wave in masklambda:
            self.__skymask[np.where(np.abs(np.log10(self._wave/wave)) <= margin)] = True

    def __filter_wavelengths(self, forbidden_wavelenghts):
        """ Mask the wavelengths in the ranges specified by the tuples in
            forbidden_wavelenghts
            """
        for item in forbidden_wavelenghts:
            self.__skymask[np.where((self._wave >= item[0]) & (self._wave <= item[1]))] = True


if __name__ == "__main__":
    pass
