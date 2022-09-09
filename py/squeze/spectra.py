"""
    SQUEzE
    ======

    This file implements the class Spectra, that is used to format a list
    of spectrum so that they can be used by SQUEzE
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"

import numpy as np

from squeze.error import Error
from squeze.spectrum import Spectrum
from squeze.simple_spectrum import SimpleSpectrum
from squeze.utils import quietprint


class Spectra:
    """ Manage the spectra list

    CLASS: Spectra
    PURPOSE: Manage the spectra list
    """

    def __init__(self, spectra_list=None):
        """ Initialize class instance

        Arguments
        ---------
        spectra_list : list of Spectrum
        List of spectra
        """
        if spectra_list is None:
            spectra_list = []
        self.spectra_list = spectra_list

    def append(self, spectrum):
        """ Add a spectrum to the list """
        if not isinstance(spectrum, Spectrum):
            raise Error("""Invalid spectrum""")
        self.spectra_list.append(spectrum)

    def size(self):
        """ Return the number of spectra """
        return len(self.spectra_list)

    def spectrum(self, index):
        """ Return the nth spectrum of the list of spectra. """
        return self.spectra_list[index]

    @classmethod
    def from_json(cls, data):
        """ This function deserializes a json string to correclty build the class.
            It uses the deserialization function of class SimpleSpectrum to reconstruct
            the instances of Spectrum. For this function to work, data should have been
            serialized using the serialization method specified in `save_json` function
            present on `utils.py` """
        spectra_list = list(
            map(SimpleSpectrum.from_json, data.get("spectra_list")))
        return cls(spectra_list)

    @classmethod
    def from_weave(cls, ob_data, userprint=quietprint):
        """ This function builds a Spectra instance containing instances of SimpleSpectrum
            from a WEAVE OB

        Arguments
        ---------
        ob_data : aps.utils.APSOB
        OB data loaded using the APS utils APSOB class

        userprint : function - default: quietprint
        Function to manage the printing with the correct level of verbosity
        """

        spectra_list = []
        for targs in ob_data.data():
            if targs.fib_status.upper() != 'A':
                userprint(f"***** Reject APS_ID = {targs.aps_id} : "
                          f"FIB_STATUS={targs.fib_status}")
                continue

            metadata = {
                'TARGID': np.str(targs.targid),
                'CNAME': np.str(targs.cname),
                'TARGCLASS': np.str(targs.targclass),
                'SPECID': np.int(targs.id),
                'APS_ID': np.int(targs.id)
            }

            for specid, _ in enumerate(targs.spectra):
                spectra_list.append(
                    SimpleSpectrum(targs.spectra[specid].flux,
                                   targs.spectra[specid].ivar,
                                   targs.spectra[specid].wave, metadata))

        return cls(spectra_list)


if __name__ == "__main__":
    pass
