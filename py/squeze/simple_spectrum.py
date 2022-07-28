"""
    SQUEzE
    ======

    This file implements the class SimpleTrainingSpectrum, that is used to make format
    spectrum to be usable by SQUEzE in training mode
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

from squeze.spectrum import Spectrum
from squeze.common_functions import deserialize


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

    @classmethod
    def from_json(cls, data):
        """ This function deserializes a json string to correclty build the class.
            For this function to work, data should have been serialized using the
            serialization method specified in `save_json` function present on
            `squeze_common_functions.py`. The current deserialisation includes the
            possibility to interpret the flux, ivar, and wave arrays as either
            normal (np.array) or masked (np.ma.array) arrays."""
        flux = deserialize(data.get("_flux"))
        ivar = deserialize(data.get("_ivar"))
        wave = deserialize(data.get("_wave"))
        metadata = {
            key.upper(): value for key, value in data.get("_metadata").items()
        }
        metadata_dtype = {
            key.upper(): value for key, value in data.get("_metadata_dtype").items()
        }

        return cls(flux, ivar, wave, metadata, metadata_dtype)


if __name__ == "__main__":
    pass
