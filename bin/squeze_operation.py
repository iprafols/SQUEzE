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

from squeze.candidates import Candidates
from squeze.config import Config
from squeze.error import Error
from squeze.model import Model
from squeze.parsers import OPERATION_PARSER
from squeze.spectra import Spectra
from squeze.utils import load_json, verboseprint, quietprint

def main(cmdargs):
    """ Run SQUEzE in operation mode """
    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     parents=[OPERATION_PARSER])
    args = parser.parse_args(cmdargs)

    # load default options
    config = Config()
    config.set_option("general", "mode", "operation")

    # manage verbosity
    if args.quiet:
        config.set_option("general", "userprint", "quietprint")
    userprint = verboseprint if not args.quiet else quietprint

    t0 = time.time()
    # load model
    if args.model is not None:
        config.set_option("model", "filename", args.model)

    # load candidates dataframe if they have previously looked for
    if args.load_candidates:
        config.set_option("candidates", "load candidates", "True")
        if args.input_candidates is not None:
            config.set_option("candidates", "input candidates", args.input_candidates)

    # initialize candidates object
    userprint("Initializing candidates object")
    if args.output_candidates is not None:
        config.set_option("general", "output", args.output_candidates)
    candidates = Candidates(config)

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
