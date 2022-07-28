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

from astropy.convolution import convolve, Gaussian1DKernel


class Spectrum(object):
    """
        Manage the spectrum data

        CLASS: Spectrum
        TYPE: Abstract
        PURPOSE: Define the required properties of a Spectrum for SQUEzE
        to be able to run. Child classes must save the flux in an np.ndarray
        named self._flux, the inverse variance in a np.ndarray named
        self._ivar, the wavelength in a np.ndarry named self._wave, and the
        metadata in a dictionary where the keys are the names of the properties
        and have type str.
        Otherwise, the methods flux, ivar, wave, metadata, metadata_by_key,
        and metadata_names, must be overwritten
        """

    def __init__(self, flux, ivar, wave, metadata, metadata_dtype):
        """ Initialize class instance

            Parameters
            ----------
            flux : np.array
            Array containing the flux

            ivar : np.array
            Array containing the inverse variance

            wave : np.array
            Array containing the wavelength

            metadata : dict
            A dictionary where the keys are the names of the properties
            and have type str.

            metadata : dtype
            A dictionary where the keys are the names of the properties
            and have type str.
            """
        self._flux = flux
        self._ivar = ivar
        self._wave = wave
        self._metadata = metadata
        self._metadata_dtype = metadata_dtype

    def flux(self):
        """ Returns the flux as a numpy.ndarray.
            Must have the same size as ivar and wavelength."""
        return self._flux

    def ivar(self):
        """ Returns the inverse variance as a numpy.ndarray.
            Must have the same size as flux and wavelength."""
        return self._ivar

    def metadata(self):
        """ Returns metadata to be included in the catalogue.
            Format must be a list of properties.
            The names of the properties should be listed in
            metadata_names.
            In training mode, this must include the true redshift
            of the spectrum in a property named "z_true". Its value
            must be np.nan if the spectrum is not a quasar.
            In training mode, the spectra must be identifiable
            via a property named "specid"
            """
        return list(self._metadata.values())

    def metadata_by_key(self, key):
        """ Access one of the elements in self._metadata by name. Return
            np.nan if not found.
            """
        return self._metadata.get(key, np.nan)

    def metadata_dtype(self):
        """ Returns metadata to be included in the catalogue.
            Format must be a list of properties.
            The names of the properties should be listed in
            metadata_names.
            In training mode, this must include the true redshift
            of the spectrum in a property named "z_true". Its value
            must be np.nan if the spectrum is not a quasar.
            In training mode, the spectra must be identifiable
            via a property named "specid"
            """
        return list(self._metadata_dtype.values())

    def metadata_dtype_by_key(self, key):
        """ Access one of the elements in self._metadata_dtype by name. Return
            np.nan if not found.
            """
        return self._metadata_dtype.get(key, np.nan)

    def metadata_names(self):
        """ Returns the names of the properties that are returned
            by metadata.
            Format must be a list of strings.
            In training mode, this must include the true redshift
            of the spectrum in a property named "z_true". Its value
            must be np.nan if the spectrum is not a quasar.
            In training mode, the spectra must be identifiable
            via a property named "specid"
            """
        return list(self._metadata)

    def rebin(self, pixel_width, extend_pixels=0):
        """ Returns a rebinned version of the flux, inverse variance and wavelength.
            New bins are centered around 4000 Angstroms and have a width specified by
            pixel_width. The rebinning is made by combining all the bins within
            +- half the pixel width of the new pixel centers.

            The flux of the new bin is computed by averaging the fluxes of the
            original array. The inverse variance of the new bin is computed by summing the
            inverse variances of the original array. The wavelength of the new bin
            is computed by averaging the wavelength of the original array.

            Parameters
            ----------
            pixel_width : float
            Width of the new pixel (in Angstroms)

            extend_pixels : float, >0 - Default: 0
            Pixel overlap region (in Angstroms)
            """
        # define matrixes
        start_wave = 4000  # Angstroms
        half_width = pixel_width / 2.0
        rebinned_wave = np.append(
            np.arange(start_wave,
                      self._wave.min() - pixel_width, -pixel_width)[::-1],
            np.arange(start_wave,
                      self._wave.max() + pixel_width, pixel_width))
        rebinned_ivar = np.zeros_like(rebinned_wave)
        rebinned_flux = np.zeros_like(rebinned_wave)
        mask = np.zeros_like(rebinned_wave, dtype=bool)

        # rebin
        for index, wave in enumerate(rebinned_wave):
            pos = np.where((self._wave >= wave - half_width - extend_pixels) &
                           (self._wave < wave + half_width + extend_pixels))
            rebinned_flux[index] = self._flux[pos].mean()
            rebinned_ivar[index] = self._ivar[pos].sum()

        mask[np.where((np.isnan(rebinned_flux)) |
                      (np.isnan(rebinned_ivar)))] = True
        rebinned_flux = np.ma.array(rebinned_flux, mask=mask)
        rebinned_ivar = np.ma.array(rebinned_ivar, mask=mask)

        # return flux, error and wavelength
        return rebinned_flux, rebinned_ivar, rebinned_wave

    def smooth(self, width):
        """ Returns a smoothed version of the flux. The smoothing is computed
            by convolving the flux with a Gaussian kernel of the specified width

            Parameters
            ----------
            width : int
            Width of the Gaussian to be used as smoothing kernel (in number of pixels)
            """
        if width > 0:
            gauss_kernel = Gaussian1DKernel(width)
            return convolve(self._flux, gauss_kernel)
        return self._flux

    def wave(self):
        """ Returns the wavelength as a numpy.ndarray
            Units may be different but Angstroms are suggested
            The user is responsible to make sure all wavelength
            are passed with the same units.
            Must have the same size as flux and ivar."""
        return self._wave


if __name__ == "__main__":
    pass
