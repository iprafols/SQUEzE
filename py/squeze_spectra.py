"""
    SQUEzE
    ======

    This file implements the class Spectra, that is used to format a list
    of spectrum so that they can be used by SQUEzE
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

from squeze_error import Error
from squeze_spectrum import Spectrum

class Spectra(object):
    """
        Manage the spectra list

        CLASS: Spectra
        PURPOSE: Manage the spectra list
        """
    def __init__(self):
        """ Initialize class instance

            Parameters
            ----------
            spectra_list : list of Spectrum
            List of spectra
            """
        self.__spectra_list = []

    def append(self, spectrum):
        """ Add a spectrum to the list """
        if not isinstance(spectrum, Spectrum):
            raise Error("""Invalid spectrum""")
        self.__spectra_list.append(spectrum)

    def spectra_list(self):
        """ Returns the list of spectra. """
        return self.__spectra_list

    def spectrum(self, index):
        """ Return the nth spectrum of the list of spectra. """
        return self.__spectra_list[index]

if __name__ == "__main__":
    pass
