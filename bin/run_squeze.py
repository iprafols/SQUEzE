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
from squeze.candidates import Candidates
from squeze.utils import load_json, deserialize, verboseprint, quietprint


def main(cmdargs):
    """ Run SQUEzE"""
    t0 = time.time()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the delta field '
                     'from a list of spectra'))

    parser.add_argument('config_file',
                        type=str,
                        default=None,
                        help='Configuration file')

    args = parser.parse_args(cmdargs)

    # load config
    config = Config(args.config_file)

    # load printing funtion
    userprint = config.userprint

    # initialize candidates object
    userprint("Initialize candidates object")
    candidates = Candidates(config)
    userprint("Done")

    if candidates.mode == "merge":
        return

    # load spectra
    candidates.load_spectra()

    # train model
    if candidates.mode in ["training", "merge_training"]:
        candidates.train_model()

    if candidates.mode in ["operation", "test", "merge_test", "merge_operation"]:
        # compute probabilities
        candidates.classify_candidates()

        # save the catalogue as a fits file
        candidates.save_catalogue()

    # check statistics
    if candidates.mode == "test":
        candidates.check_statistics()

    t1 = time.time()
    userprint(f"INFO: total elapsed time: {(t1-t0)/60.0} minutes")
    userprint("Done")

if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
