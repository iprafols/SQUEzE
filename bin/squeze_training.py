#!/usr/bin/python3
# pylint: disable=duplicate-code
"""
    SQUEzE
    ======

    This file allows the user to execute SQUEzE in training mode.
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import argparse
import sys
import time

from squeze.common_functions import load_json
from squeze.common_functions import deserialize
from squeze.common_functions import verboseprint, quietprint
from squeze.error import Error
from squeze.quasar_catalogue import QuasarCatalogue
from squeze.spectra import Spectra
from squeze.candidates import Candidates
from squeze.defaults import LINES
from squeze.defaults import RANDOM_FOREST_OPTIONS
from squeze.defaults import RANDOM_STATE
from squeze.defaults import TRY_LINES
from squeze.defaults import Z_PRECISION
from squeze.defaults import PEAKFIND_WIDTH
from squeze.defaults import PEAKFIND_SIG
from squeze.parsers import TRAINING_PARSER


def main(cmdargs):
    """ Run SQUEzE in training mode """
    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     parents=[TRAINING_PARSER])
    args = parser.parse_args(cmdargs)

    # manage verbosity
    userprint = verboseprint if not args.quiet else quietprint

    t0 = time.time()
    # load lines
    userprint("Loading lines")
    lines = LINES if args.lines is None else deserialize(load_json(args.lines))

    # load try_line
    try_line = TRY_LINES if args.try_lines is None else args.try_lines

    # load redshift precision
    z_precision = Z_PRECISION if args.z_precision is None else args.z_precision

    # load peakfinder options
    peakfind_width = PEAKFIND_WIDTH if args.peakfind_width is None else args.peakfind_width
    peakfind_sig = PEAKFIND_SIG if args.peakfind_sig is None else args.peakfind_sig

    # load random forest options
    random_forest_options = RANDOM_FOREST_OPTIONS if args.random_forest_options is None else load_json(args.random_forest_options)
    random_state = RANDOM_STATE if args.random_state is None else args.random_state

    # initialize candidates object
    userprint("Initializing candidates object")
    if args.output_candidates is None:
        candidates = Candidates(lines_settings=(lines, try_line),
                                z_precision=z_precision, mode="training",
                                peakfind=(peakfind_width, peakfind_sig),
                                model=None, userprint=userprint,
                                model_options=(random_forest_options, random_state))
    else:
        candidates = Candidates(lines_settings=(lines, try_line),
                                z_precision=z_precision, mode="training",
                                name=args.output_candidates,
                                peakfind=(peakfind_width, peakfind_sig),
                                model=None, userprint=userprint,
                                model_options=(random_forest_options, random_state))

    # load candidates dataframe if they have previously looked for
    if args.load_candidates:
        userprint("Loading existing candidates")
        t1 = time.time()
        candidates.load_candidates(args.input_candidates)
        t2 = time.time()
        userprint(f"INFO: time elapsed to load candidates: {(t2-t1)/60.0} minutes")

    # load spectra
    if args.input_spectra is not None:
        userprint("Loading spectra")
        t3 = time.time()
        columns_candidates = []
        userprint("There are {} files with spectra to be loaded".format(len(args.input_spectra)))
        for index, spectra_filename in enumerate(args.input_spectra):
            userprint("Loading spectra from {} ({}/{})".format(spectra_filename, index,
                                                               len(args.input_spectra)))
            t30 = time.time()
            spectra = Spectra.from_json(load_json(spectra_filename))
            if not isinstance(spectra, Spectra):
                raise Error("Invalid list of spectra")

            if index == 0:
                columns_candidates += spectra.spectra_list()[0].metadata_names()

            # look for candidates
            userprint("Looking for candidates")
            candidates.find_candidates(spectra.spectra_list(), columns_candidates)

            t31 = time.time()
            userprint(f"INFO: time elapsed to find candidates from {spectra_filename}: "
                      f"{(t31-t30)/60.0} minutes")



        t4 = time.time()
        userprint(f"INFO: time elapsed to find candidates: {(t4-t3)/60.0} minutes")

        # convert to dataframe
        userprint("Converting candidates to dataframe")
        t5 = time.time()
        candidates.candidates_list_to_dataframe(columns_candidates)
        t6 = time.time()
        userprint(f"INFO: time elapsed to convert candidates to dataframe: {(t6-t5)/60.0} minutes")

    # train model
    userprint("Training model")
    t7 = time.time()
    candidates.train_model(args.model_fits)
    t8 = time.time()
    userprint(f"INFO: time elapsed to train model: {(t8-t7)/60.0} minutes")

    userprint(f"INFO: total elapsed time: {(t8-t0)/60.0} minutes")
    userprint("Done")
if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
