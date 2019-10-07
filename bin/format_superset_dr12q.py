"""
    SQUEzE - SupersetDR12Q
    ==============
    This file is a modified version of format_spectra tailored to load
    SuperesetDR12Q data

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

import numpy as np

import astropy.io.fits as fits

from squeze.squeze_common_functions import save_json
from squeze.squeze_common_functions import verboseprint, quietprint
from squeze.squeze_error import Error
from squeze.squeze_quasar_catalogue import QuasarCatalogue
from squeze.squeze_boss_spectrum import BossSpectrum
from squeze.squeze_spectra import Spectra
from squeze.squeze_parsers import PARENT_PARSER, QUASAR_CATALOGUE_PARSER

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
                        help="""Base name of the json files where the list of spectra
                            will be saved. The sufix _plate####.json, where #### will be
                            replaced with the plate number, will be added to save the
                            spectra on the different plates.""")
    parser.add_argument("--input-folder", type=str, required=True,
                        help="""Name of the folder containg the spectra to process. In
                            this folder, spectra are found in a subfoldare with the plate
                            number.""")
    parser.add_argument("--rebin-pixels-width", type=float, default=0,
                        help="""Width of the new pixel (in Angstroms).""")
    parser.add_argument("--extend-pixels", type=float, default=0,
                        help="""Pixel overlap region (in Angstroms)""")
    parser.add_argument("--mask-jpas", action="store_true",
                        help="""If set, mask pixels corresponding to filters in trays T3 and T4
                            Only works if the bin size is 100 Angstroms""")
    parser.add_argument("--mask-jpas-alt", action="store_true",
                        help="""If set, mask pixels corresponding to filters in trays T3* and T4
                            Only works if the bin size is 100 Angstroms. Ignored if --mask-jpas is
                            passed""")
    parser.add_argument("--noise", type=int, default=1,
                        help="""Adds noise to the spectrum by adding a gaussian random
                            number of width equal to the (noise-1) times the given
                            variance. Then increase the variance by a factor of
                            sqrt(noise)""")
    parser.add_argument("--sky-mask", type=str, required=True,
                        help="""Name of the file containing the sky mask""")
    parser.add_argument("--margin", type=float, default=1.5e-4,
                        help="""Margin used in the masking. Wavelengths separated to
                            wavelength given in the array by less than the margin
                            will be masked""")
    parser.add_argument("--sequels", action="store_true",
                        help="""Format SEQUELS plates instead of BOSS plates""")
    parser.add_argument("--single-plate", type=int, required=False, default=0,
                        help="""Loadd BOSS spectra only from this plate""")
    parser.add_argument("--forbidden-wavelengths", type=float, nargs='*',
                        help="""A list containing floats specifying ranges of wavelengths
                            that will be masked (both ends included). Odd (even) positions
                            will be lower (upper) limits of the ranges""")

    args = parser.parse_args()
    # parsing --forbidden-wavelengths
    if args.forbidden_wavelengths is not None:
        aux = []
        for i in range(0, len(args.forbidden_wavelengths), 2):
            try:
                aux.append((args.forbidden_wavelengths[i], args.forbidden_wavelengths[i + 1]))
            except IndexError:
                verboseprint("--forbidden-wavelengths must have an even number of elements.\n Stopping...")
                return
        args.forbidden_wavelengths = aux

    # manage verbosity
    userprint = verboseprint if not args.quiet else quietprint

    # load plate list
    userprint("loading list of plates")
    plate_list_hdu = fits.open(args.plate_list)
    if args.sequels:
        plate_list = plate_list_hdu[1].data["plate"][
            np.where((plate_list_hdu[1].data["programname"] == "sequels") &
                     (plate_list_hdu[1].data["platequality"] == "good"))].copy()
    else:
        plate_list = plate_list_hdu[1].data["plate"][
            np.where((plate_list_hdu[1].data["programname"] == "boss") &
                     (plate_list_hdu[1].data["platequality"] == "good"))].copy()
    plate_list = np.unique(plate_list.astype(int))
    del plate_list_hdu[1].data
    plate_list_hdu.close()

    # load quasar catalogue
    userprint("loading quasar catalogue")
    quasar_catalogue = QuasarCatalogue(args.qso_cat, args.qso_cols, args.qso_specid, args.qso_hdu)
    quasar_catalogue = quasar_catalogue.quasar_catalogue()
    
    # load sky mask
    masklambda = np.genfromtxt(args.sky_mask)

    # loop over plates, will save a json file for each plate
    userprint("loading spectra in each of the plates")
    missing_files = []
    for plate in plate_list:
        
        if not (args.single_plate == 0 or plate == args.single_plate):
            continue

        # reset spectra object
        spectra = Spectra()

        # get list of spectra in this plate
        folder = "{}{:04d}/".format(args.input_folder, plate)

        # loop over spectra
        for index in quasar_catalogue[quasar_catalogue["plate"] == plate].index:
            entry = quasar_catalogue.loc[index]
            z_conf_person = entry["z_conf_person"]
            boss_target1 = entry["boss_target1"].astype(int)
            eboss_target0 = entry["eboss_target0"].astype(int)
            if (z_conf_person != 3):
                continue
            if not ((boss_target1 & 2**40 > 0) |
                    (boss_target1 & 2**41 > 0) |
                    (boss_target1 & 2**42 > 0) |
                    (boss_target1 & 2**43 > 0) |
                    (boss_target1 & 2**44 > 0) |
                    (boss_target1 & 2**10 > 0) |
                    (boss_target1 & 2**11 > 0) |
                    (boss_target1 & 2**12 > 0) |
                    (boss_target1 & 2**13 > 0) |
                    (boss_target1 & 2**14 > 0) |
                    (boss_target1 & 2**15 > 0) |
                    (boss_target1 & 2**16 > 0) |
                    (boss_target1 & 2**17 > 0) |
                    (boss_target1 & 2**18 > 0) |
                    (boss_target1 & 2**19 > 0) |
                    (eboss_target0 & 2**10 > 0) |
                    (eboss_target0 & 2**11 > 0) |
                    (eboss_target0 & 2**12 > 0) |
                    (eboss_target0 & 2**13 > 0) |
                    (eboss_target0 & 2**14 > 0) |
                    (eboss_target0 & 2**15 > 0) |
                    (eboss_target0 & 2**16 > 0) |
                    (eboss_target0 & 2**17 > 0) |
                    (eboss_target0 & 2**18 > 0) |
                    (eboss_target0 & 2**20 > 0) |
                    (eboss_target0 & 2**22 > 0) |
                    (eboss_target0 & 2**30 > 0) |
                    (eboss_target0 & 2**31 > 0) |
                    (eboss_target0 & 2**33 > 0) |
                    (eboss_target0 & 2**34 > 0) |
                    (eboss_target0 & 2**35 > 0) |
                    (eboss_target0 & 2**40 > 0) ):
                continue
            metadata = {}
            for column in quasar_catalogue.columns:
                metadata[column] = entry[column]
            metadata["z_true"] = entry["z_vi"]
            spectrum_file = "spec-{:04d}-{:05d}-{:04d}.fits".format(plate,
                                                                    entry["mjd"].astype(int),
                                                                    entry["fiberid"].astype(int))
            
            
            # add spectra to list
            try:
                spectra.append(BossSpectrum("{}{}".format(folder, spectrum_file), metadata,
                                            (masklambda, args.margin),
                                            mask_jpas=args.mask_jpas,
                                            mask_jpas_alt=args.mask_jpas_alt,
                                            rebin_pixels_width=args.rebin_pixels_width,
                                            noise_increase=args.noise,
                                            extend_pixels=args.extend_pixels,
                                            forbidden_wavelenghts=args.forbidden_wavelengths))
            except IOError:
                missing_files.append(spectrum_file)
                #print "missing file {}".format(spectrum_file)

        # save spectra in the current plate
        save_json("{}_plate{:04d}.json".format(args.out, plate), spectra)

    for item in missing_files:
        userprint(item)

if __name__ == '__main__':
    main()
