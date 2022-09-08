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
    if args.check_statistics:
        userprint("Loading quasar catalogue")
        if args.qso_dataframe is not None:
            quasar_catalogue = deserialize(load_json(args.qso_dataframe))
            quasar_catalogue["LOADED"] = True
        else:
            quasar_catalogue = QuasarCatalogue(args.qso_cat, args.qso_cols,
                                               args.qso_specid, args.qso_ztrue,
                                               args.qso_hdu).quasar_catalogue()
            quasar_catalogue["LOADED"] = False
        t1 = time.time()
        userprint(f"INFO: time elapsed to load quasar catalogue: {(t1-t0)/60.0} minutes")

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
    if args.check_statistics:
        probs = args.check_probs if args.check_probs is not None else np.arange(0.9, 0.0, -0.05)
        userprint("Check statistics")
        data_frame = candidates.candidates
        userprint("\n---------------")
        userprint("step 1")
        candidates.find_completeness_purity(quasar_catalogue.reset_index(), data_frame)
        for prob in probs:
            userprint("\n---------------")
            userprint("proba > {}".format(prob))
            candidates.find_completeness_purity(quasar_catalogue.reset_index(),
                                                data_frame[(data_frame["PROB"] > prob) &
                                                           ~(data_frame["DUPLICATED"]) &
                                                           (data_frame["Z_CONF_PERSON"] == 3)],
                                                )

    # save the catalogue as a fits file
    candidates.save_catalogue()

    t12 = time.time()
    userprint(f"INFO: total elapsed time: {(t12-t0)/60.0} minutes")
    userprint("Done")

if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
