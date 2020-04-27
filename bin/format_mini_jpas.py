#!/usr/bin/python3
"""
    SQUEzE - Mini J-PAS
    ==============
    This file is a modified version of format_spectra tailored to load
    Mini J-PAS data

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

import astropy.io.fits as fits

from squeze.common_functions import save_json
from squeze.common_functions import verboseprint, quietprint
from squeze.error import Error
from squeze.simple_spectrum import SimpleSpectrum
from squeze.spectra import Spectra

def main():
    """ Load DESI spectra using the Spectra and DESISpectrum Classes
        defined in squeze_boss_spectra.py and squeze_desi_spectrum.py
        respectively.
        """

    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--input-filename", type=str, required=True,
                        help="""Name of the fits file to be loaded.""")
    parser.add_argument("--filters-info", type=str, required=True,
                        help="""Name of the file containing the filters
                        information""")
    parser.add_argument("--mag-col", type=str, required=True,
                        help="""Name of the magnitude system to be used. For example,
                        type PSFCOR to use the column FLambdaDualObj.FLUX_PSFCOR for them
                        flux and FLambdaDualObj.MAG_RELERR_PSFCOR for the errors""")
    parser.add_argument("--output-filename", type=str, required=True,
                        help="""Name of the output filename.""")
    parser.add_argument("--keep-cols", nargs='+', default=None, required=False,
                        help="""White-spaced list of the columns to be kept in the
                        formatted spectra""")

    args = parser.parse_args()

    # load filters information
    hdu_filters = fits.open(args.filters_info)
    filter_names = hdu_filters[1].data["Filter.name"]
    select_filters = np.array([i for i, name in enumerate(filter_names) if name.startswith("J")])
    wave = hdu_filters[1].data["Filter.wavelength"][select_filters]
    hdu_filters.close()

    # initialize squeze Spectra class
    squeze_spectra = Spectra()

    # loop over spectra
    hdu = fits.open(args.input_filename)
    for row in hdu[1].data:

        # load data
        mask = ((row["FLambdaDualObj.FLAGS"] > 0) |
                (row["FLambdaDualObj.MASK_FLAGS"] > 0))
        flux = np.ma.array(row["FLambdaDualObj.FLUX_{}".format(args.mag_col)], mask=mask)
        relerr = np.ma.array(row["FLambdaDualObj.FLUX_RELERR_{}".format(args.mag_col)], mask=mask)
        ivar = 1/(flux*relerr)**2

        # select the filters in select_filters
        flux = flux[select_filters]
        ivar = ivar[select_filters]


        # prepare metadata
        metadata = {}
        for column in args.keep_cols:
            metadata[column] = row[column]
        metadata["specid"] = int("{}{}".format(row["FLambdaDualObj.TILE_ID"],
                                               row["FLambdaDualObj.NUMBER"]))

        # format spectrum
        spectrum = SimpleSpectrum(flux, ivar, wave , metadata)

        # append to list
        squeze_spectra.append(spectrum)

    # save formated spectra
    save_json(args.output_filename, squeze_spectra)

if __name__ == '__main__':
    main()
