# pylint: disable=duplicate-code
"""
    SQUEzE
    ======

    This file allows the user to execute SQUEzE in training mode. See
    the 'Usage' section in the README for detailed usage instructions.
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import argparse

from squeze.squeze_common_functions import load_json
from squeze.squeze_common_functions import verboseprint, quietprint
from squeze.squeze_error import Error
from squeze.squeze_quasar_catalogue import QuasarCatalogue
from squeze.squeze_spectra import Spectra
from squeze.squeze_candidates import Candidates
from squeze.squeze_defaults import CUTS
from squeze.squeze_defaults import LINES
from squeze.squeze_defaults import RANDOM_FOREST_OPTIONS
from squeze.squeze_defaults import RANDOM_STATE
from squeze.squeze_defaults import TRY_LINES
from squeze.squeze_defaults import Z_PRECISION
from squeze.squeze_defaults import PEAKFIND_WIDTH
from squeze.squeze_defaults import PEAKFIND_SIG
from squeze.squeze_parsers import TRAINING_PARSER


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
            parser.error("options --qso-cat, --qso-cols, and --qso-specid " \
                         "are incompatible with --qso-dataframe")
        quasar_catalogue = load_json(args.qso_dataframe)
        quasar_catalogue["loaded"] = True
    else:
        if (args.qso_cat is None) or (args.qso_cols is None) or (args.qso_specid is None):
            parser.error("--qso-cat, --qso-cols and --qso-specid are " \
                         "required if --qso-dataframe is not passed")
        quasar_catalogue = QuasarCatalogue(args.qso_cat, args.qso_cols,
                                           args.qso_specid, args.qso_hdu).quasar_catalogue()
        quasar_catalogue["loaded"] = False

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

    # load cut options
    cuts = CUTS if args.cuts is None else load_json(args.cuts)

    # initialize candidates object
    userprint("Looking for candidates")
    if args.output_candidates is None:
        candidates = Candidates(lines_settings=(lines, try_line),
                                z_precision=z_precision, mode="training",
                                weighting_mode=args.weighting_mode,
                                peakfind=(peakfind_width, peakfind_sig),
                                model=(None, cuts))
    else:
        candidates = Candidates(lines_settings=(lines, try_line),
                                z_precision=z_precision, mode="training",
                                name=args.output_candidates,
                                weighting_mode=args.weighting_mode,
                                peakfind=(peakfind_width, peakfind_sig),
                                model=(None, cuts))

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

    # train model
    userprint("Training model")
    candidates.train_model()

    # check completeness
    if args.check_statistics:
        probs = args.check_probs if args.check_probs is not None else np.arange(0.9, 0.0, -0.05)
        userprint("Check statistics")
        data_frame = candidates.candidates()
        userprint("\n---------------")
        userprint("step 1")
        candidates.find_completeness_purity(quasar_catalogue, data_frame)
        for prob in probs:
            userprint("\n---------------")
            userprint("SVM proba > {}".format(prob))
            candidates.find_completeness_purity(quasar_catalogue,
                                                data_frame[(data_frame["prob"] > prob) &
                                                           ~(data_frame["duplicated"]) &
                                                           (data_frame["z_conf_person"] == 3)],
                                                userprint=userprint)

if __name__ == '__main__':
    main()
