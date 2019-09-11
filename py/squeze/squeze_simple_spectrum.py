"""
    SQUEzE
    ======

    This file implements the class SimpleTrainingSpectrum, that is used to make format
    spectrum to be usable by SQUEzE in training mode
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

from squeze.squeze_spectrum import Spectrum

class SimpleSpectrum(Spectrum):
    """
        Manage the spectrum data

        CLASS: SimpleSpectrum
        PURPOSE: Format a spectrum for SQUEzE to be able to run in
        training mode
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
        self._flux = flux
        self._ivar = ivar
        self._wave = wave

        self._metadata = metadata

if __name__ == "__main__":
    pass
