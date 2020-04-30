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


def main():
    """ Run SQUEzE in candidates mode """
    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     parents=[CANDIDATES_PARSER])
    args = parser.parse_args()

    # manage verbosity
    userprint = verboseprint if not args.quiet else quietprint

    # load model
    if args.model is not None:
        userprint("Loading model")
    model = None if args.model is None else Model.from_json(load_json(args.model))

    # load lines
    userprint("Loading lines")
    lines = LINES if args.lines is None else load_json(args.lines)

    # load try_line
    try_line = TRY_LINES if args.try_lines is None else args.try_lines

    # load redshift precision
    z_precision = Z_PRECISION if args.z_precision is None else args.z_precision

    # load peakfinder options
    peakfind_width = PEAKFIND_WIDTH if args.peakfind_width is None else args.peakfind_width
    peakfind_sig = PEAKFIND_SIG if args.peakfind_sig is None else args.peakfind_sig

    # initialize candidates object
    userprint("Looking for candidates")
    if args.output_candidates is None:
        candidates = Candidates(lines_settings=(lines, try_line),
                                z_precision=z_precision, mode="candidates",
                                peakfind=(peakfind_width, peakfind_sig),
                                model=model)
    else:
        candidates = Candidates(lines_settings=(lines, try_line),
                                z_precision=z_precision, mode="candidates",
                                name=args.output_candidates,
                                peakfind=(peakfind_width, peakfind_sig),
                                model=model)

    # load candidates dataframe if they have previously looked for
    if args.load_candidates:
        userprint("Loading existing candidates")
        candidates.load_candidates(args.input_candidates)

    # load spectra
    if args.input_spectra is not None:
        userprint("Loading spectra")
        userprint("There are {} files with spectra to be loaded".format(len(args.input_spectra)))
        for index, spectra_filename in enumerate(args.input_spectra):
            userprint("Loading spectra from {} ({}/{})".format(spectra_filename, index,
                                                               len(args.input_spectra)))
            spectra = Spectra.from_json(load_json(spectra_filename))
            if not isinstance(spectra, Spectra):
                raise Error("Invalid list of spectra")

            # look for candidates
            userprint("Looking for candidates")
            candidates.find_candidates(spectra.spectra_list())

    userprint("Done")
if __name__ == '__main__':
    main()
