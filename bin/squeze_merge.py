#!/usr/bin/python3
"""
    SQUEzE
    ======

    This file allows the user to merge two or more SQUEzE candidate objects
    into a single candidate object.
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import argparse
import sys
import time

from squeze.config import Config
from squeze.common_functions import verboseprint, quietprint
from squeze.candidates import Candidates
from squeze.parsers import MERGING_PARSER

def main(cmdargs):
    """ Run SQUEzE in merging mode """
    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     parents=[MERGING_PARSER])
    args = parser.parse_args(cmdargs)

    # load default options
    config = Config()
    config.set_option("general", "mode", "merge")

    # manage verbosity
    if args.quiet:
        config.set_option("general", "userprint", "quietprint")
    userprint = verboseprint if not args.quiet else quietprint

    t0 = time.time()
    # load candidates object
    userprint("Initializing candidates object")
    if args.output_candidates is None:
        config.set_option("general", "output", args.output_candidates)
    candidates = Candidates(config)

    # load the first candidates object
    userprint("Loading first candidate object")
    candidates.load_candidates(args.input_candidates[0])

    # merge the other candidates objects
    userprint("Merging with the other candidate objects")
    candidates.merge(args.input_candidates[1:])

    t1 = time.time()
    userprint(f"INFO: total elapsed time: {(t1-t0)/60.0} minutes")
    userprint("Done")

if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
