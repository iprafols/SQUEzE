"""
    SQUEzE
    ======

    This file implements the class Spectra, that is used to format a list
    of spectrum so that they can be used by SQUEzE
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

from squeze.squeze_error import Error
from squeze.squeze_spectrum import Spectrum
from squeze.squeze_simple_spectrum import SimpleSpectrums

class Spectra(object):
    """
        Manage the spectra list

        CLASS: Spectra
        PURPOSE: Manage the spectra list
        """
    def __init__(self, spectra_list=[]):
        """ Initialize class instance

            Parameters
            ----------
            spectra_list : list of Spectrum
            List of spectra
            """
        self.__spectra_list = spectra_list

    def append(self, spectrum):
        """ Add a spectrum to the list """
        if not isinstance(spectrum, Spectrum):
            raise Error("""Invalid spectrum""")
        self.__spectra_list.append(spectrum)

    def size(self):
        """ Return the number of spectra """
        return len(self.__spectra_list)

    def spectra_list(self):
        """ Returns the list of spectra. """
        return self.__spectra_list

    def spectrum(self, index):
        """ Return the nth spectrum of the list of spectra. """
        return self.__spectra_list[index]

    @classmethod
    def from_json(cls, data):
        """ This function deserializes a json string to correclty build the class.
            It uses the deserialization function of class SimpleSpectrum to reconstruct
            the instances of Spectrum. For this function to work, data should have been
            serialized using the serialization method specified in `save_json` function
            present on `squeze_common_functions.py` """
        spectra_list = list(map(SimpleSpectrum.from_json, data.get("_Spectra__spectra_list")))
        return cls(spectra_list)

if __name__ == "__main__":
    pass
