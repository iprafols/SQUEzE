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

import numpy as np

from astropy.convolution import convolve, Gaussian1DKernel


class Spectrum:
    """ Manage the spectrum data

    CLASS: Spectrum
    TYPE: Abstract
    PURPOSE: Define the required properties of a Spectrum for SQUEzE
    to be able to run. Child classes must save the flux in an np.ndarray
    named self.flux, the inverse variance in a np.ndarray named
    self.ivar, the wavelength in a np.ndarry named self.wave, and the
    metadata in a dictionary where the keys are the names of the properties
    and have type str.
    Otherwise, the methods flux, ivar, wave, metadata, metadata_by_key,
    and metadata_names, must be overwritten
    """

    def __init__(self, flux, ivar, wave, metadata):
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
            """
        self.flux = flux
        self.ivar = ivar
        self.wave = wave
        self.metadata = metadata

    def metadata_values(self):
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
        return list(self.metadata.values())

    def metadata_by_key(self, key):
        """ Access one of the elements in self.metadata by name.
        Return np.nan if not found.
        """
        return self.metadata.get(key, np.nan)

    def metadata_names(self):
        """ Returns the names of the properties that are returned by
        metadata_values.

        Format must be a list of strings.
        In training mode, this must include the true redshift
        of the spectrum in a property named "z_true". Its value
        must be np.nan if the spectrum is not a quasar.
        In training mode, the spectra must be identifiable
        via a property named "specid"
        """
        return list(self.metadata)

    def rebin(self, pixel_width, extend_pixels=0):
        """ Returns a rebinned version of the flux, inverse variance and wavelength.

        New bins are centered around 4000 Angstroms and have a width specified by
        pixel_width. The rebinning is made by combining all the bins within
        +- half the pixel width of the new pixel centers.

        The flux of the new bin is computed by averaging the fluxes of the
        original array. The inverse variance of the new bin is computed by summing the
        inverse variances of the original array. The wavelength of the new bin
        is computed by averaging the wavelength of the original array.

        Arguments
        ---------
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
                      self.wave.min() - pixel_width, -pixel_width)[::-1],
            np.arange(start_wave,
                      self.wave.max() + pixel_width, pixel_width))
        rebinned_ivar = np.zeros_like(rebinned_wave)
        rebinned_flux = np.zeros_like(rebinned_wave)
        mask = np.zeros_like(rebinned_wave, dtype=bool)

        # rebin
        for index, wave in enumerate(rebinned_wave):
            pos = np.where((self.wave >= wave - half_width - extend_pixels) &
                           (self.wave < wave + half_width + extend_pixels))
            rebinned_flux[index] = self.flux[pos].mean()
            rebinned_ivar[index] = self.ivar[pos].sum()

        mask[np.where((np.isnan(rebinned_flux)) |
                      (np.isnan(rebinned_ivar)))] = True
        rebinned_flux[mask] = 0.0
        rebinned_ivar[mask] = 0.0

        # return flux, error and wavelength
        return rebinned_flux, rebinned_ivar, rebinned_wave

    def smooth(self, width):
        """ Returns a smoothed version of the flux.

        The smoothing is computed
        by convolving the flux with a Gaussian kernel of the specified width

        Arguments
        ---------
        width : int
        Width of the Gaussian to be used as smoothing kernel (in number of pixels)
        """
        if width > 0:
            gauss_kernel = Gaussian1DKernel(width)
            return convolve(self.flux, gauss_kernel)
        return self.flux


if __name__ == "__main__":
    pass
