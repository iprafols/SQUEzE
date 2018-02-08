"""
    SQUEzE
    ======

    This file allows the user to execute SQUEzE in operation mode. See
    the 'Usage' section in the README for detailed usage instructions.
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import argparse

from squeze_common_functions import load_pkl
from squeze_common_functions import verboseprint, quietprint
from squeze_error import Error
from squeze_spectra import Spectra
from squeze_candidates import Candidates
from squeze_defaults import CUTS_OPERATION as CUTS
from squeze_defaults import LINES
from squeze_defaults import TRY_LINE
from squeze_parsers import OPERATION_PARSER

def main():
    """ Run SQUEzE in operation mode """
    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     parents=[OPERATION_PARSER])
    args = parser.parse_args()

    # manage verbosity
    userprint = verboseprint if not args.quiet else quietprint

    # load lines
    userprint("Loading lines")
    lines = LINES if args.lines is None else load_pkl(args.lines)

    # load try_line
    try_line = TRY_LINE if args.try_line is None else args.try_line

    # load candidates object
    userprint("Looking for candidates")
    if args.output_candidates is None:
        candidates = Candidates(lines_settings=(lines, try_line),
                                mode="operation",
                                weighting_mode=args.weighting_mode)
    else:
        candidates = Candidates(lines_settings=(lines, try_line),
                                mode="operation",
                                name=args.output_candidates,
                                weighting_mode=args.weighting_mode)

    # load candidates object if they have previously looked for
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
            spectra = load_pkl(spectra_filename)
            if not isinstance(spectra, Spectra):
                raise Error("Invalid list of spectra")

            # look for candidates
            userprint("Looking for candidates")
            candidates.find_candidates(spectra.spectra_list())

    # load cuts
    userprint("Loading cuts")
    cuts = CUTS if args.cuts is None else load_pkl(args.cuts)

    # apply cuts
    userprint("Applying cuts")
    found_catalogue = candidates.apply_cuts(cuts, args.output_cuts)

    # save the catalogue
    candidates.to_fits(args.output_catalogue, data_frame=found_catalogue)

if __name__ == '__main__':
    main()
