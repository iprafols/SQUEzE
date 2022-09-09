"""
    SQUEzE
    ======

    This file implements the class MySpectrum, that is a "fill the blanks"
    example of a class inheriting from Spectrum defined in squeze_spectrum.py.
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"
# "fill the blanks" example ... pylint: disable=fixme

from squeze.error import Error
from squeze.spectrum import Spectrum


class MySpectrum(Spectrum):
    """ Example of a class inheriting from Spectrum

    CLASS: MySpectrum
    PURPOSE: Format spectrum following the required SQUEzE
    constraints
    """
    # TODO: add arguments as required
    def __init__(self, flux, ivar, wave, metadata):
        """ Initialize class instance.

        This function should be modified as required or removed if no
        specific initialization operations are required

        Arguments
        ---------
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
        super().__init__(flux, ivar, wave, metadata)
        # TODO: fill function
        raise Error("Not implemented")


if __name__ == "__main__":
    pass
