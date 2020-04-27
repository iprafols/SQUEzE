#!/usr/bin/python3
# pylint: disable=duplicate-code
"""
    SQUEzE
    ======

    This file allows the user to execute SQUEzE in operation mode.
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import argparse

from squeze.common_functions import load_json
from squeze.common_functions import verboseprint, quietprint
from squeze.error import Error
from squeze.model import Model
from squeze.spectra import Spectra
from squeze.candidates import Candidates
from squeze.parsers import OPERATION_PARSER

def main():
    """ Run SQUEzE in operation mode """
    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     parents=[OPERATION_PARSER])
    args = parser.parse_args()

    # manage verbosity
    userprint = verboseprint if not args.quiet else quietprint

    # load model
    userprint("Loading model")
    model = Model.from_json(load_json(args.model))

    # initialize candidates object
    userprint("Looking for candidates")
    if args.output_candidates is None:
        candidates = Candidates(mode="operation", model=model)
    else:
        candidates = Candidates(mode="operation", name=args.output_candidates,
                                model=model)

    # load candidates dataframe if they have previously looked for
    if args.load_candidates:
        userprint("Loading existing candidates")
        candidates.load_candidates(args.input_candidates)

    # load spectra
    if args.input_spectra is not None:
        userprint("Loading spectra")
        userprint("There are {} files with spectra to be loaded".format(len(args.input_spectra)))
        for index, spectra_filename in enumerate(args.input_spectra):
            userprint("Loading spectra from {} ({}/{})".format(spectra_filename, index,
                                                               len(args.input_spectra)))
            spectra = Spectra.from_json(load_json(spectra_filename))
            if not isinstance(spectra, Spectra):
                raise Error("Invalid list of spectra")

            # look for candidates
            userprint("Looking for candidates")
            candidates.find_candidates(spectra.spectra_list())

    # compute probabilities
    userprint("Computing probabilities")
    candidates.classify_candidates()

    # save the catalogue as a fits file
    if not args.no_save_fits:
        found_catalogue = candidates.candidates()
        found_catalogue = found_catalogue[(~found_catalogue["duplicated"]) &
                                          (found_catalogue["prob"] > args.prob_cut)]
        candidates.to_fits(args.output_catalogue, data_frame=found_catalogue)

    userprint("Done")

if __name__ == '__main__':
    main()
