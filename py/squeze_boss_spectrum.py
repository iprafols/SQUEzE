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

import astropy.io.fits as fits

from squeze_error import Error
from squeze_spectrum import Spectrum

class BossSpectrum(Spectrum):
    """
        Load and format a BOSS spectrum to be digested by SQUEzE

        CLASS: BossSpectrum
        PURPOSE: Load and format a BOSS spectrum to be digested by
        SQUEzE
        """
    def __init__(self, spectrum_file, metadata):
        """ Initialize class instance

            Parameters
            ----------
            spectrum_file : str
            Name of the fits files containing the spectrum

            metadata : dict
            A dictionary with the metadata. Keys should be strings
            """
        # check that "specid" is present in metadata
        if "specid" not in metadata.keys():
            raise Error("""The property "specid" must be present in metadata""")

        spectrum_hdu = fits.open(spectrum_file)
        self._flux = spectrum_hdu[1].data["flux"].copy()
        self._wave = 10**spectrum_hdu[1].data["loglam"].copy()
        self._ivar = spectrum_hdu[1].data["ivar"].copy()
        self._metadata = metadata
        del spectrum_hdu[1].data
        spectrum_hdu.close()


if __name__ == "__main__":
    pass
