"""
    SQUEzE - WEAVE
    ==============
    This file is a modified version of format_spectra tailored to load
    WEAVE OPR2.5++ data

    From format_spectra.py :

    This file shows an example of how should the spectra should be formatted.
    The example is based on loading BOSS data and some parts should probably
    be heavily modified for other surveys.

    Each individual spectrum should be loaded as an instance of the Spectrum
    class defined in the file squeze_spectum.py. There is basically two ways
    of doing this. For users with object-oriented exeprience, it is recommended
    to create a new class YourSurveySpectrum that inherits from Spectrum. A
    simpler way (but a bit more restrictive) is to make use of the SimpleSpectrum
    class provided in squeze_simple_spectrum.py. This example covers both options.

    The complete set of spectra is to be loaded in the class Spectra defined
    in squeze_spectra.py.
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import argparse
import sys

import tqdm

import numpy as np

import pandas as pd

import astropy.io.fits as fits

from squeze.quasar_catalogue import QuasarCatalogue
from squeze.parsers import PARENT_PARSER, QUASAR_CATALOGUE_PARSER
from squeze.spectra import Spectra
from squeze.utils import save_json, verboseprint, quietprint
from squeze.weave_spectrum import WeaveSpectrum

def main(cmdargs):
    """ Load WEAVE spectra using the WeaveSpectrum Class defined in
        squeze_weave_spectrum.py.
        """

    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     parents=[PARENT_PARSER,
                                              QUASAR_CATALOGUE_PARSER])

    parser.add_argument("--red-spectra", nargs='+', type=str, default=None, required=True,
                        help="""Name of the fits file containig the red CCD data.
                            Size should be the same as the list in --blue-spectra.""")

    parser.add_argument("--blue-spectra", nargs='+', type=str, default=None, required=True,
                        help="""Name of the fits file containig the blue CCD data.
                            Size should be the same as the list in --blue-spectra.""")

    parser.add_argument("--out", type=str, default="spectra.json", required=False,
                        help="""Name of the json file where the list of spectra
                            will be saved""")

    parser.add_argument("--mag-cat", type=str, default=None, required=False,
                        help="""Name of the text file containing the measured magnitudes
                            for the observed spectra""")

    args = parser.parse_args(cmdargs)

    # manage verbosity
    userprint = verboseprint if not args.quiet else quietprint

    # load quasar catalogue
    userprint("loading catalogue from {}".format(args.qso_cat))
    quasar_catalogue = QuasarCatalogue(args.qso_cat, args.qso_cols, args.qso_specid, args.qso_hdu)
    quasar_catalogue = quasar_catalogue.quasar_catalogue()

    # load magnitudes catalogue
    if args.mag_cat is not None:
        mags_catalogue = pd.read_csv(args.mag_cat, delim_whitespace=True)

    # intialize spectra variable
    spectra = Spectra()

    for red_filename, blue_filename in zip(args.red_spectra, args.blue_spectra):
        # load red spectra
        userprint("loading red spectra from {}".format(red_filename))
        observed_red = fits.open(red_filename)
        wave = {"red_delta_wave" : observed_red["RED_DATA"].header["CD1_1"],
                "red_wave" : np.zeros(observed_red["RED_DATA"].header["NAXIS1"], dtype=float)}
        wave.get("red_wave")[0] = observed_red["RED_DATA"].header["CRVAL1"]
        for index in range(1, wave.get("red_wave").size):
            wave["red_wave"][index] = wave["red_wave"][index - 1] + wave.get("red_delta_wave")
        targid = observed_red["FIBTABLE"].data["TARGID"].astype(str)

        # load blue spectra
        userprint("loading blue spectra from {}".format(blue_filename))
        observed_blue = fits.open(blue_filename)
        wave["blue_delta_wave"] = observed_blue["BLUE_DATA"].header["CD1_1"]
        wave["blue_wave"] = np.zeros(observed_blue["BLUE_DATA"].header["NAXIS1"], dtype=float)
        wave.get("blue_wave")[0] = observed_blue["BLUE_DATA"].header["CRVAL1"]
        for index in range(1, wave.get("blue_wave").size):
            wave.get("blue_wave")[index] = wave.get("blue_wave")[index - 1] + \
                wave.get("blue_delta_wave")

        # format spectra
        userprint("formatting red and blue data into a single spectra")
        for index in tqdm.tqdm(range(observed_red["RED_DATA"].header["NAXIS2"])):
            spectrum_dict = {"red_flux" : observed_red["RED_DATA"].data[index]*\
                observed_red["RED_SENSFUNC"].data[index],
                             "red_ivar" : observed_red["RED_IVAR"].data[index]*\
                observed_red["RED_SENSFUNC"].data[index],
                             "blue_flux" : observed_blue["BLUE_DATA"].data[index]*\
                observed_blue["BLUE_SENSFUNC"].data[index],
                             "blue_ivar" : observed_blue["BLUE_IVAR"].data[index]*\
                observed_blue["BLUE_SENSFUNC"].data[index]}

            # check that we don't have an empty spectrum
            if np.unique(spectrum_dict.get("red_flux")).size == 1:
                continue

            # add targid to metadata
            metadata = {"TARGID" : targid[index], "SPECID" : targid[index]}

            # add true redshift to metadata
            if quasar_catalogue[quasar_catalogue["TARGID"] == targid[index]]["Z"].shape[0] > 0:
                metadata["Z_TRUE"] = quasar_catalogue[quasar_catalogue["TARGID"] ==
                                                      targid[index]]["Z"].values[0]
            else:
                metadata["Z_TRUE"] = np.nan

            # add magnitudes to metadata
            if mags_catalogue[mags_catalogue["TARGID"] == targid[index]]["GMAG"].shape[0] > 0:
                metadata["GMAG"] = mags_catalogue[mags_catalogue["TARGID"] == \
                                                  targid[index]]["GMAG"].values[0]
                metadata["RMAG"] = mags_catalogue[mags_catalogue["TARGID"] == \
                                                  targid[index]]["RMAG"].values[0]
            else:
                metadata["GMAG"] = np.nan
                metadata["RMAG"] = np.nan

            # add spectrum to list
            spectra.append(WeaveSpectrum(spectrum_dict, wave, metadata))

    # save them as a json file to be used by SQUEzE
    save_json(args.out, spectra)

if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
