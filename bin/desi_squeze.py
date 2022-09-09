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
import sys

import numpy as np
import astropy.io.fits as fits

from desispec.io import read_spectra

from squeze.candidates import Candidates
from squeze.desi_spectrum import DesiSpectrum
from squeze.error import Error
from squeze.model import Model
from squeze.spectra import Spectra
from squeze.utils import save_json, load_json, verboseprint, quietprint


def convert_dtype(dtype):
     if dtype == "O":
         return "15A"
     else:
         return dtype

def main(cmdargs):
    """ Load DESI spectra using the Spectra and DESISpectrum Classes
        defined in squeze_boss_spectra.py and squeze_desi_spectrum.py
        respectively.
        """

    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input-filename", type=str, required=True,
                        help="""Name of the filename to be loaded to be loaded.""")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="""Name of the file containing the trained model.""")
    parser.add_argument("-o","--output-filename", type=str, required=True,
                        help="""Name of the output fits file.""")
    parser.add_argument("-e","--single-exp", action="store_true",
                        help="""Load only the first reobservation for each spectrum""")
    parser.add_argument("--metadata", nargs='+', required=False,
                        default=["TARGETID"],
                        help="""White-spaced list of the list of columns to keep as metadata""")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="""Print messages""")
    args = parser.parse_args(cmdargs)

    # prepare variables
    assert args.output_filename.endswith("fits") or args.output_filename.endswith("fits.gz")
    if args.verbose:
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
    userprint('Looking for candidates')
    candidates.find_candidates(squeze_spectra.spectra_list
    columns_candidates = squeze_spectra.spectra_list[0].metadata_names()
    candidates.candidates_list_to_dataframe(columns_candidates, save=False)

    # compute probabilities
    userprint("Computing probabilities")
    candidates.classify_candidates(save=False)

    # filter results
    data_frame = candidates.candidates()
    data_frame = data_frame[~data_frame["DUPLICATED"]]

    # save results
    data_out = np.zeros(len(data_frame), dtype=[('TARGETID','int64'),('Z_SQ','float64'),('Z_SQ_CONF','float64')])
    data_out['TARGETID'] = data_frame['TARGETID'].values
    data_out['Z_SQ'] = data_frame['Z_TRY'].values
    data_out['Z_SQ_CONF'] = data_frame['PROB'].values

    data_hdu = fits.BinTableHDU.from_columns(data_out,name='SQZ_CAT')
    data_hdu.writeto(args.output_filename)

if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
