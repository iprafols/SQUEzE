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
    def flux(self):
        """ Returns the flux as a numpy.ndarray.
            Must have the same size as ivar and wavelength."""
        # member must be declared in child class ... pylint: disable=no-member
        return self._flux

    def ivar(self):
        """ Returns the inverse variance as a numpy.ndarray.
            Must have the same size as flux and wavelength."""
        # member must be declared in child class ... pylint: disable=no-member
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
        # member must be declared in child class ... pylint: disable=no-member
        return self._metadata.values()

    def metadata_by_key(self, key):
        """ Access one of the elements in self._metadata by name. Return
            np.nan if not found.
            """
        # member must be declared in child class ... pylint: disable=no-member
        return self._metadata.get(key, np.nan)

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
        # member must be declared in child class ... pylint: disable=no-member
        return self._metadata.keys()
    
    def rebin(self, num_pixels):
        """ Returns a rebinned version of the flux, inverse variance and wavelength.
            The rebinning is done by combining num_pixels pixels. If the total number
            of pixels is not multiple of num_pixels, pixels with highest wavelength
            are discarded.

            The flux of the new bin is computed by averaging the fluxes of the
            original array, weighted by the inverse variance associated to each
            pixel. The inverse variance of the new bin is computed by summing the
            inverse variances of the original array. The wavelength of the new bin
            is computed by averaging the wavelength of the original array.

            Parameters
            ----------
            num_pixels : int
            Number of pixels to combine in the rebinning
            """
        # define matrixes
        new_size = self.flux.size//num_pixels
        rebinned_flux = np.zeros(new_size, dtype=float)
        rebinned_ivar = np.zeros_like(rebinned_flux)
        rebinned_wave = np.zeros_like(rebinned_flux)

        # rebin
        for index in range(0, rebinned_flux.size):
            rebinned_flux[index] = np.average(self._flux[index*num_pixels: (index + 1)*num_pixels], weigths=self._ivar[index*num_pixels: (index + 1)*num_pixels])
            rebinned_ivar[index] = np.sum(self._ivar[index*num_pixels: (index + 1)*num_pixels])
            rebinned_wave[index] = self._wave[index*num_pixels: (index + 1)*num_pixels].mean()

        # return flux, error and wavelength
        return rebinned_flux, rebinned_ivar, rebinned_wave

    def smooth(self, width):
        """ Returns a smoothed version of the flux. The smoothing is computed
            by convolving the flux with a Gaussian kernel of the specified width
            
            Parameters
            ----------
            width : float
            Width of the Gaussian to be used as smoothing kernel (in number of pixels)
            """
        gauss_kernel = Gaussian1DKernel(width)
        # member must be declared in child class ... pylint: disable=no-member
        return convolve(self._flux, gauss_kernel)

    def wave(self):
        """ Returns the wavelength as a numpy.ndarray
            Units may be different but Angstroms are suggested
            The user is responsible to make sure all wavelength
            are passed with the same units.
            Must have the same size as flux and ivar."""
        # member must be declared in child class ... pylint: disable=no-member
        return self._wave

if __name__ == "__main__":
    pass
