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

from squeze.config import Config
from squeze.common_functions import load_json
from squeze.common_functions import deserialize
from squeze.common_functions import verboseprint, quietprint
from squeze.error import Error
from squeze.spectra import Spectra
from squeze.candidates import Candidates
from squeze.defaults import LINES
from squeze.defaults import TRY_LINES
from squeze.defaults import Z_PRECISION
from squeze.defaults import PEAKFIND_WIDTH
from squeze.defaults import PEAKFIND_SIG
from squeze.parsers import CANDIDATES_PARSER
from squeze.model import Model


def main(cmdargs):
    """ Run SQUEzE in candidates mode """
    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     parents=[CANDIDATES_PARSER])
    args = parser.parse_args(cmdargs)

    # load default options
    config = Config()
    config.set_option("general", "mode", "candidates")

    # manage verbosity
    if args.quiet:
        config.set_option("general", "userprint", "quietprint")
    userprint = verboseprint if not args.quiet else quietprint

    t0 = time.time()
    # load model
    if args.model is not None:
        config.set_option("model", "filename", args.model)

    # load lines
    userprint("Loading lines")
    if args.lines is not None:
        config.set_option("candidates", "lines", args.lines)

    # load try_line
    if args.try_lines is not None:
        config.set_option("candidates", "try lines", str(args.try_lines))

    # load redshift precision
    if args.z_precision is not None:
        config.set_option("candidates", "z precision", str(args.z_precision))

    # load peakfinder options
    if args.peakfind_width is not None:
        config.set_option("peak finder", "width", str(args.peakfind_width))
    if args.peakfind_sig is not None:
        config.set_option("peak finder", "min significance", str(args.peakfind_sig))

    # initialize candidates object
    userprint("Initializing candidates object")
    if args.output_candidates is not None:
        config.set_option("general", "output", args.output_candidates)
    candidates = Candidates(config)

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


    userprint(f"INFO: total elapsed time: {(t7-t0)/60.0} minutes")
    userprint("Done")

if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
