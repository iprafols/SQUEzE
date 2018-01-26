"""
    SQUEzE
    ======

    This file allows the user to execute SQUEzE in training mode. See
    the 'Usage' section in the README for detailed usage instructions.
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import argparse

from squeze_common_functions import load_pkl
from squeze_common_functions import verboseprint, quietprint
from squeze_error import Error
from squeze_quasar_catalogue import QuasarCatalogue
from squeze_spectra import Spectra
from squeze_candidates import Candidates
from squeze_defaults import CUTS_TRAINING as CUTS
from squeze_defaults import LINES
from squeze_defaults import TRY_LINE
from squeze_defaults import Z_PRECISION
from squeze_parsers import TRAINING_PARSER

def main():
    """ Run SQUEzE in training mode """
    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     parents=[TRAINING_PARSER])
    args = parser.parse_args()

    # manage verbosity
    userprint = verboseprint if not args.quiet else quietprint

    # load quasar catalogue
    userprint("Loading quasar catalogue")
    if args.qso_dataframe is not None:
        if ((args.qso_cat is not None) or (args.qso_cols is not None) or
            (args.qso_specid is not None)):
            parser.error("options --qso-cat, --qso-cols, and --qso-specid are incompatible with --qso-dataframe")
        quasar_catalogue = load_pkl(args.qso_dataframe)
        quasar_catalogue["loaded"] = True
    else:
        if (args.qso_cat is None) or (args.qso_cols is None) or (args.qso_specid is None):
            parser.error("--qso-cat, --qso-cols and --qso-specid are required if --qso-dataframe is not passed")
        quasar_catalogue = QuasarCatalogue(args.qso_cat, args.qso_cols,
                                       args.qso_specid, args.qso_hdu).quasar_catalogue()
        quasar_catalogue["loaded"] = False

    # load lines
    userprint("Loading lines")
    lines = LINES if args.lines is None else load_pkl(args.lines)

    # load try_line
    try_line = TRY_LINE if args.try_line is None else args.try_line

    # load redshift precision
    z_precision = Z_PRECISION if args.z_precision is None else args.z_precision

    # load candidates object
    userprint("Looking for candidates")
    if args.output_candidates is None:
        candidates = Candidates(lines_settings=(lines, try_line),
                                z_precision=z_precision, mode="training",
                                weighting_mode=args.weighting_mode)
    else:
        candidates = Candidates(lines_settings=(lines, try_line),
                                z_precision=z_precision, mode="training",
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

            # flag loaded quasars as such
            for spec in spectra.spectra_list():
                if quasar_catalogue[
                        quasar_catalogue["specid"] == spec.metadata_by_key("specid")].shape[0] > 0:
                    index = quasar_catalogue.index[
                        quasar_catalogue["specid"] == spec.metadata_by_key("specid")].tolist()[0]
                    quasar_catalogue.at[index, "loaded"] = True

            # look for candidates
            userprint("Looking for candidates")
            candidates.find_candidates(spectra.spectra_list())

    # load cuts
    userprint("Loading cuts")
    cuts = CUTS if args.cuts is None else load_pkl(args.cuts)

    # overload cuts
    def overload_cuts(cuts):
        """ Overwrites the information passed down as cuts with the information given
            in args.cuts_percentiles and args.cuts_names """
        if (args.cuts_names is None and args.cuts_percentiles is not None) or \
            (args.cuts_names is not None and args.cuts_percentiles is None):
            raise Error("""both or none of the arguments --cut-names and
                --cut-percentiles are required""")
        elif args.cuts_names is None:
            pass
        elif len(args.cuts_names) != len(args.cuts_percentiles):
            raise Error("""arguments --cut-names and --cut-percentiles should
                have the same size """)
        else:
            for (name, percentile) in zip(args.cuts_names, args.cuts_percentiles):
                for i in range(0, len(cuts)):
                    if cuts[i][0] == name and cuts[i][2] == "percentile":
                        cuts[i][1] = percentile
        return cuts
    cuts = overload_cuts(cuts)

    # apply cuts
    userprint("Applying cuts")
    loaded_quasars = quasar_catalogue[quasar_catalogue["loaded"] == True]
    stats = candidates.find_completeness_purity_cuts(loaded_quasars, cuts,
                                                     quiet=args.quiet, get_results=True)

    # process cuts to use them in operation mode
    userprint("Processing cuts to use in operation mode")
    candidates.process_cuts(cuts, args.output_cuts, stats)

if __name__ == '__main__':
    main()
