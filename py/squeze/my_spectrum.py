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
    """
        Example of a class inheriting from Spectrum

        CLASS: MySpectrum
        PURPOSE: Format spectrum following the required SQUEzE
        constraints
        """

    # TODO: add arguments as required
    def __init__(self):
        """ This is the main function of the class. It should load
            the data. The flux must be stored in a variable named self._flux,
            the inverse variance in self._ivar, the wavelength as self._wave,
            and the metadata in self._metadata.
            Check the definition of Spectrum for more details"""
        # TODO: fill function
        raise Error("Not implemented")


if __name__ == "__main__":
    pass
