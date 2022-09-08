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

    # settings to load spectra
    if args.input_spectra is not None:
        config.set_option("candidates", "input spectra", " ".join(args.input_spectra))

    # settings to save catalogue
    if args.no_save_catalogue is not None:
        config.set_option("candidates", "save catalogue flag", str(~args.no_save_catalogue))
        config.set_option("candidates", "prob cut", str(args.prob_cut))

    # initialize candidates object
    userprint("Initializing candidates object")
    if args.output_candidates is not None:
        config.set_option("general", "output", args.output_candidates)
    candidates = Candidates(config)

    # load spectra
    candidates.load_spectra()

    # compute probabilities
    candidates.classify_candidates()

    # save the catalogue as a fits file
    candidates.save_catalogue()

    t10 = time.time()
    userprint(f"INFO: total elapsed time: {(t10-t0)/60.0} minutes")
    userprint("Done")

if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
