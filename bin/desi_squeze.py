"""
    SQUEzE - DESI
    ==============
    Python wrapper to run SQUEzE code on DESI DATA and generate reduced output
    tables.

    For a detailed study of the quasar candidates, run format_desi_minisv.py and
    and then squeze_operation.py
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import argparse

import numpy as np
import astropy.io.fits as fits

from desispec.io import read_spectra

from squeze.candidates import Candidates
from squeze.common_functions import save_json
from squeze.common_functions import verboseprint, quietprint
from squeze.error import Error
from squeze.desi_spectrum import DesiSpectrum
from squeze.spectra import Spectra

def convert_dtype(dtype):
     if dtype == "O":
         return "15A"
     else:
         return dtype

def main():
    """ Load DESI spectra using the Spectra and DESISpectrum Classes
        defined in squeze_boss_spectra.py and squeze_desi_spectrum.py
        respectively.
        """

    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--input-filename", type=str, required=True,
                        help="""Name of the filename to be loaded to be loaded.""")
    parser.add_argument("--model", type=str, required=True,
                        help="""Name of the file containing the trained model.""")
    parser.add_argument("--output-filename", type=str, required=True,
                        help="""Name of the output fits file.""")
    parser.add_argument("--single-exp", action="store_true",
                        help="""Load only the first reobservation for each spectrum""")
    parser.add_argument("--metadata", nargs='+', required=False,
                        default=["TARGETID"],
                        help="""White-spaced list of the list of columns to keep as metadata""")
    parser.add_argument("--keep-cols", nargs='+', required=False,
                        default=["PROB", "Z_TRY", "TARGETID"],
                        help="""Name of the columns kept in the final fits file.""")
    parser.add_argument("--quiet", action="store_true",
                        help="""Do not print messages""")
    args = parser.parse_args()

    # prepare variables
    assert args.output_filename.endswith("fits") or args.output_filename.endswith("fits.gz")
    if args.quiet:
        userprint = verboseprint
    else:
        userprint = quietprint
    args.keep_cols = [col.upper() for col in args.keep_cols]

    # read desi spectra
    userprint("Reading spectra")
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
        metadata = {col.upper(): desi_spectra.fibermap[col][pos[0][0]] for col in args.metadata}

        # add specid
        metadata["SPECID"] = targid

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

    # load model
    userprint("Reading model")
    if args.model.endswith(".json"):
        model = Model.from_json(load_json(args.model))
    else:
        model = Model.from_fits(args.model)

    # initialize candidates object
    userprint("Initialize candidates object")
    candidates = Candidates(mode="operation", model=model,
                            name=args.output_filename)

    # look for candidates
    userprint("Looking for candidates")
    if save_file is None:
        candidates.find_candidates(spectra.spectra_list(), save=False)
    else:
        candidates.find_candidates(spectra.spectra_list(), save=True)

    # compute probabilities
    userprint("Computing probabilities")
    candidates.classify_candidates()

    # filter results
    data_frame = candidates.candidates()
    data_frame = data_frame[~data_frame["DUPLICATED"]][args.keep_cols]

    # save results
    hdu = fits.BinTableHDU.from_columns([fits.Column(name=col,
                                                     format=convert_dtype(dtype),
                                                     array=data_frame[col])
                                         for col, dtype in zip(data_frame.columns,
                                                               data_frame.dtypes)])
    hdu.writeto(args.output_filename, overwrite=True)

if __name__ == '__main__':
    main()
