"""
    SQUEzE - BOSS
    ==============
    This file is a modified version of format_spectra tailored to load
    BOSS DR12 data

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

from os import listdir
from os.path import isfile, join

import tqdm

import numpy as np

import astropy.io.fits as fits



from squeze_common_functions import save_pkl
from squeze_common_functions import verboseprint, quietprint
from squeze_quasar_catalogue import QuasarCatalogue
from squeze_boss_spectrum import BossSpectrum
from squeze_spectra import Spectra
from squeze_parsers import PARENT_PARSER, QUASAR_CATALOGUE_PARSER

def main():
    """ Load BOSS spectra using the BossSpectrum Class defined in
        squeze_boss_spectrum.py.

        Spectra will be saved in blocs of spectra belonging to the same
        plate. This is useful to be able to use SQUEzE without using
        too much memory.
        """

    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     parents=[PARENT_PARSER,
                                              QUASAR_CATALOGUE_PARSER])

    parser.add_argument("--plate-list", type=str, required=True,
                        help="""Name of the fits file containing the list of spectra
                            to be loaded""")

    parser.add_argument("--out", type=str, default="spectra", required=False,
                        help="""Base name of the pkl files where the list of spectra
                            will be saved. The sufix _plate####.pkl, where #### will be
                            replaced with the plate number, will be added to save the
                            spectra on the different plates.""")
    parser.add_argument("--input-folder", type=str, required=True,
                        help="""Name of the folder containg the spectra to process. In
                            this folder, spectra are found in a subfoldare with the plate
                            number.""")

    args = parser.parse_args()

    # manage verbosity
    userprint = verboseprint if not args.quiet else quietprint

    # load plate list
    userprint("loading list of plates")
    plate_list_hdu = fits.open(args.plate_list)
    plate_list = plate_list_hdu[1].data["plate"][
        np.where((plate_list_hdu[1].data["programname"] == "boss") &
                 (plate_list_hdu[1].data["platequality"] == "good"))].copy()
    plate_list = np.unique(plate_list)
    del plate_list_hdu[1].data
    plate_list_hdu.close()

    # load quasar catalogue
    userprint("loading quasar catalogue")
    quasar_catalogue = QuasarCatalogue(args.qso_cat, args.qso_cols, args.qso_specid, args.qso_hdu)
    quasar_catalogue = quasar_catalogue.quasar_catalogue()

    # initialize specid_count for those spectra not in the quasar catalogue
    specid_counter = -1

    # loop over plates, will save a pkl file for each plate
    userprint("loading spectra in each of the plates")
    for plate in tqdm.tqdm(plate_list):

        # reset spectra object
        spectra = Spectra()

        # get list of spectra in this plate
        folder = "{}{:04d}/".format(args.input_folder, plate)
        spectrum_file_list = [f for f in listdir(folder) if isfile(join(folder, f))]

        # loop over spectra
        for spectrum_file in spectrum_file_list:

            # get metadata
            entry = quasar_catalogue[(quasar_catalogue["fiberid"] == int(spectrum_file[16:20])) &
                                     (quasar_catalogue["mjd"] == int(spectrum_file[10:15])) &
                                     (quasar_catalogue["plate"] == plate)]
            if entry.shape[0] > 0:
                metadata = {}
                for column in quasar_catalogue.columns:
                    metadata[column] = entry[column].values[0]
                metadata["z_true"] = entry["z_vi"].values[0]
            else:
                metadata = {key: np.nan for key in quasar_catalogue.columns if key != "specid"}
                metadata["z_true"] = 0.0
                metadata["specid"] = specid_counter
                specid_counter -= 1

            # add spectra to list
            spectra.append(BossSpectrum("{}{}".format(folder, spectrum_file), metadata))

        # save spectra in the current plate
        save_pkl("{}_plate{:04d}.pkl".format(args.out, plate), spectra)

if __name__ == '__main__':
    main()
