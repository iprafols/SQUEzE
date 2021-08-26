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
import sys
import time

from squeze.common_functions import load_json
from squeze.common_functions import verboseprint, quietprint
from squeze.error import Error
from squeze.model import Model
from squeze.spectra import Spectra
from squeze.candidates import Candidates
from squeze.parsers import OPERATION_PARSER

def main(cmdargs):
    """ Run SQUEzE in operation mode """
    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     parents=[OPERATION_PARSER])
    args = parser.parse_args(cmdargs)

    # manage verbosity
    userprint = verboseprint if not args.quiet else quietprint

    t0 = time.time()
    # load model
    userprint("Loading model")
    if args.model.endswith(".json"):
        model = Model.from_json(load_json(args.model))
    else:
        model = Model.from_fits(args.model)
    t1 = time.time()
    userprint(f"INFO: time elapsed to load model", (t1-t0)/60.0, 'minutes')

    # initialize candidates object
    userprint("Initializing candidates object")
    if args.output_candidates is None:
        candidates = Candidates(mode="operation", model=model, userprint=userprint)
    else:
        candidates = Candidates(mode="operation", name=args.output_candidates,
                                model=model, userprint=userprint)

    # load candidates dataframe if they have previously looked for
    if args.load_candidates:
        userprint("Loading existing candidates")
        t2 = time.time()
        candidates.load_candidates(args.input_candidates)
        t3 = time.time()
        userprint(f"INFO: time elapsed to load candidates: {(t3-t2)/60.0} minutes")

    # load spectra
    if args.input_spectra is not None:
        userprint("Loading spectra")
        t4 = time.time()
        columns_candidates = []
        userprint("There are {} files with spectra to be loaded".format(len(args.input_spectra)))
        for index, spectra_filename in enumerate(args.input_spectra):
            userprint("Loading spectra from {} ({}/{})".format(spectra_filename, index,
                                                               len(args.input_spectra)))
            t40 = time.time()
            spectra = Spectra.from_json(load_json(spectra_filename))
            if not isinstance(spectra, Spectra):
                raise Error("Invalid list of spectra")

            if index == 0:
                columns_candidates += spectra.spectra_list()[0].metadata_names()

            # look for candidates
            userprint("Looking for candidates")
            candidates.find_candidates(spectra.spectra_list(), columns_candidates)

            t41 = time.time()
            userprint(f"INFO: time elapsed to find candidates from {spectra_filename}:"
                      f" {(t41-t40)/60.0} minutes")

        t5 = time.time()
        userprint(f"INFO: time elapsed to find candidates: {(t5-t4)/60.0} minutes")

        # convert to dataframe
        userprint("Converting candidates to dataframe")
        t6 = time.time()
        candidates.candidates_list_to_dataframe(columns_candidates)
        t7 = time.time()
        userprint(f"INFO: time elapsed to convert candidates to dataframe: {(t7-t6)/60.0} minutes")

    # compute probabilities
    userprint("Computing probabilities")
    t8 = time.time()
    candidates.classify_candidates()
    t9 = time.time()
    userprint(f"INFO: time elapsed to classify candidates: {(t9-t8)/60.0} minutes")

    # save the catalogue as a fits file
    if not args.no_save_catalogue:
        candidates.save_catalogue(args.output_catalogue, args.prob_cut)

    t10 = time.time()
    userprint(f"INFO: total elapsed time: {(t10-t0)/60.0} minutes")
    userprint("Done")

if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
