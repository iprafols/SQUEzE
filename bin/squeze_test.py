#!/usr/bin/python3
# pylint: disable=duplicate-code
"""
    SQUEzE
    ======

    This file allows the user to execute SQUEzE in test mode.
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import argparse
import sys
import time
import numpy as np

from squeze.candidates import Candidates
from squeze.config import Config
from squeze.error import Error
from squeze.quasar_catalogue import QuasarCatalogue
from squeze.model import Model
from squeze.parsers import TEST_PARSER, quasar_parser_check
from squeze.spectra import Spectra
from squeze.utils import load_json, deserialize, verboseprint, quietprint


def main(cmdargs):
    """ Run SQUEzE in test mode """
    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     parents=[TEST_PARSER])
    args = parser.parse_args(cmdargs)
    if args.check_statistics:
        quasar_parser_check(parser, args)

    # load default options
    config = Config()
    config.set_option("general", "mode", "test")

    # manage verbosity
    if args.quiet:
        config.set_option("general", "userprint", "quietprint")
    userprint = verboseprint if not args.quiet else quietprint

    t0 = time.time()
    # load quasar catalogue (only if --check-statistics is passed)
    config.set_option("stats", "run stats", str(args.check_statistics))
    if args.check_statistics:
        if args.qso_dataframe is not None:
            config.set_option("stats", "qso dataframe", args.qso_dataframe)
        else:
            config.set_option("stats", "filename", args.qso_cat)
            config.set_option("stats", "columns", " ".join(args.qso_cols))
            config.set_option("stats", "specid column", args.qso_specid)
            config.set_option("stats", "ztrue column", args.qso_ztrue)
            config.set_option("stats", "hdu", str(args.qso_hdu))
        if args.check_probs is not None:
            check_probs_str_list = [str(item) for item in args.check_probs]
            config.set_option("stats", "check probs", " ".join(check_probs_str_list))

    # load model
    if args.model is not None:
        config.set_option("model", "filename", args.model)

    # load candidates dataframe if they have previously looked for
    if args.load_candidates:
        config.set_option("candidates", "load candidates", "True")
        if args.input_candidates is not None:
            config.set_option("candidates", "input candidates", args.input_candidates)

    # setting to load spectra
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

    # check completeness
    candidates.check_statistics()

    # save the catalogue as a fits file
    candidates.save_catalogue()

    t12 = time.time()
    userprint(f"INFO: total elapsed time: {(t12-t0)/60.0} minutes")
    userprint("Done")

if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
