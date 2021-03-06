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

from squeze.common_functions import verboseprint, quietprint
from squeze.candidates import Candidates
from squeze.parsers import MERGING_PARSER

def main():
    """ Run SQUEzE in merging mode """
    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     parents=[MERGING_PARSER])
    args = parser.parse_args()

    # manage verbosity
    userprint = verboseprint if not args.quiet else quietprint

    # load candidates object
    userprint("Looking for candidates")
    if args.output_candidates is None:
        candidates = Candidates(mode="merge")
    else:
        candidates = Candidates(mode="merge",
                                name=args.output_candidates)

    # load the first candidates object
    userprint("Loading first candidate object")
    candidates.load_candidates(args.input_candidates[0])

    # merge the other candidates objects
    userprint("Merging with the other candidate objects")
    candidates.merge(args.input_candidates[1:], userprint=userprint)

    userprint("Done")

if __name__ == '__main__':
    main()
