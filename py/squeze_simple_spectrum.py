"""
    SQUEzE
    ======

    This file implements the class SimpleTrainingSpectrum, that is used to make format
    spectrum to be usable by SQUEzE in training mode
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

from squeze_spectrum import Spectrum

class SimpleSpectrum(Spectrum):
    """
        Manage the spectrum data

        CLASS: SimpleSpectrum
        PURPOSE: The purpose of this class is twofold.
        First it serves as tool to create instances of class Spectrum loaded from
        JSON files. Second, it provides an example of the minimum requirements a
        derived Spectrum class should have in order for them to run on SQUEzE
        (for this second purpose, ignore the from_json method)
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

    @classmethod
    def from_json(cls, data):
        """ This function deserializes a json string to correclty build the class.
            For this function to work, data should have been serialized using the
            serialization method specified in `save_json` function present on
            `squeze_common_functions.py`. The current deserialisation includes the
            possibility to interpret the flux, ivar, and wave arrays as either
            normal (np.array) or masked (np.ma.array) arrays."""
        if data.get("_flux").get("mask", None) is None:
            flux = np.array(data.get("_flux").get("data"))
        else:
            flux = np.ma.array(data.get("_flux").get("data"),
                               mask=data.get("_flux").get("mask"))
        if data.get("_ivar").get("mask", None) is None:
            ivar = np.array(data.get("_ivar").get("data"))
        else:
            ivar = np.ma.array(data.get("_ivar").get("data"),
                               mask=data.get("_ivar").get("mask"))
        if data.get("_wave").get("mask", None) is None:
            wave = np.array(data.get("_wave").get("data"))
        else:
            wave = np.ma.array(data.get("_wave").get("data"),
                               mask=data.get("_wave").get("mask"))
        metadata = data.get("_metadata")
        
        return cls(flux, ivar, wave, metadata)

if __name__ == "__main__":
    pass
