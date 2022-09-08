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
from squeze.error import Error
from squeze.spectra import Spectra
from squeze.candidates import Candidates
from squeze.model import Model
from squeze.parsers import CANDIDATES_PARSER
from squeze.utils import load_json, deserialize, verboseprint, quietprint


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

    # load candidates dataframe if they have previously looked for
    if args.load_candidates:
        config.set_option("candidates", "load candidates", "True")
        if args.input_candidates is not None:
            config.set_option("candidates", "input candidates", args.input_candidates)

    # setting to load spectra
    if args.input_spectra is not None:
        config.set_option("candidates", "input spectra", " ".join(args.input_spectra))

    # initialize candidates object
    userprint("Initializing candidates object")
    if args.output_candidates is not None:
        config.set_option("general", "output", args.output_candidates)
    candidates = Candidates(config)

    # load spectra
    candidates.load_spectra()
    t1 = time.time()

    userprint(f"INFO: total elapsed time: {(t1-t0)/60.0} minutes")
    userprint("Done")

if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
