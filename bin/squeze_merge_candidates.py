"""
    SQUEzE
    ======

    This file allows the user to merge two or more SQUEzE candidate objects
    into a single candidate object. See the 'Usage' section in the README for
    detailed usage instructions.
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import argparse

import tqdm

from squeze_common_functions import load_pkl
from squeze_common_functions import verboseprint, quietprint
from squeze_candidates import Candidates
from squeze_parsers import MERGING_PARSER

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
    candidates.merge(args.input_candidates[1:])

if __name__ == '__main__':
    main()
