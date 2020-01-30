"""
    SQUEzE - DESI
    ==============
    This file is a modified version of format_spectra tailored to load
    DESI data

    From format_superset_dr12q.py :

    This file shows an example of how should the spectra should be formatted.
    The example is based on loading BOSS data and some parts should probably
    be heavily modified for other surveys.

    Each individual spectrum should be loaded as an instance of the Spectrum
    class defined in the file squeze_spectum.py. There is basically two ways
    of doing this. For users with object-oriented exeprience, it is recommended
    to create a new class YourSurveySpectrum that inherits from Spectrum. A
    simpler way (but a bit more restrictive) is to make use of the SimpleSpectrum
    class provided in squeze_simple_spectrum.py.

    The complete set of spectra is to be loaded in the class Spectra defined
    in squeze_spectra.py.
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import argparse

import numpy as np

from desispec.io import read_spectra

from squeze.squeze_common_functions import save_json
from squeze.squeze_common_functions import verboseprint, quietprint
from squeze.squeze_error import Error
from squeze.squeze_desi_spectrum import DesiSpectrum
from squeze.squeze_spectra import Spectra

def main():
    """ Load DESI spectra using the Spectra and DESISpectrum Classes
        defined in squeze_boss_spectra.py and squeze_desi_spectrum.py
        respectively.
        """

    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--input-filename", type=str, required=True,
                        help="""Name of the filename to be loaded to be loaded.""")
    parser.add_argument("--output-filename", type=str, required=True,
                        help="""Name of the output filename.""")
    parser.add_argument("--single-exp", "store_true",
                        help="""Load only the first reobservation for each spectrum""")

    args = parser.parse_args()

    # read desi spectra
    desi_spectra = read_spectra(args.input_filename)

    # initialize squeze Spectra class
    squeze_spectra = Spectra()

    # get targetids
    targetid = np.unique(desi_spectra.fibermap["TARGETID"])

    # loop over targeid
    for targid in targetid:

        # select objects
        pos = np.where(desi_spectra.fibermap["TARGETID"] == targid)

        # prepare metadata
        metadata = {"targetid": targid,
                    "specid": targid,
                    "sv1_desi_target": desi_spectra.fibermap["SV1_DESI_TARGET"][pos[0][0]] }

        # Extract-2 data
        flux = {}
        wave = {}
        ivar = {}
        mask = {}
        for band in desi_spectra.bands:
            flux[band] = desi_spectra.flux[band][pos]
            wave[band] = desi_spectra.wave[band]
            ivar[band] = desi_spectra.ivar[band][pos]
            mask[band] = desi_spectra.mask[band][pos]

        # format spectrum
        spectrum = DesiSpectrum(flux, wave, ivar, mask, metadata, args.single_exp)

        # append to list
        squeze_spectra.append(spectrum)

    # save formated spectra
    save_json(args.output_filename, squeze_spectra)

if __name__ == '__main__':
    main()
