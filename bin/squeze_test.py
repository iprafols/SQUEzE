# pylint: disable=duplicate-code
"""
    SQUEzE
    ======

    This file allows the user to execute SQUEzE in test mode.
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import argparse
import numpy as np

from squeze.common_functions import load_json
from squeze.common_functions import deserialize
from squeze.common_functions import verboseprint, quietprint
from squeze.error import Error
from squeze.quasar_catalogue import QuasarCatalogue
from squeze.model import Model
from squeze.spectra import Spectra
from squeze.candidates import Candidates
from squeze.defaults import CUTS
from squeze.parsers import TEST_PARSER


def main():
    """ Run SQUEzE in test mode """
    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     parents=[TEST_PARSER])
    args = parser.parse_args()

    # manage verbosity
    userprint = verboseprint if not args.quiet else quietprint

    # load quasar catalogue
    userprint("Loading quasar catalogue")
    if args.qso_dataframe is not None:
        if ((args.qso_cat is not None) or (args.qso_cols is not None) or
                (args.qso_specid is not None) or (args.qso_ztrue is not None)):
            parser.error("options --qso-cat, --qso-cols, --qso-specid, and --qso-ztrue " \
                         "are incompatible with --qso-dataframe")
        quasar_catalogue = deserialize(load_json(args.qso_dataframe))
        quasar_catalogue["loaded"] = True
    else:
        if (args.qso_cat is None) or (args.qso_cols is None) or (args.qso_specid is None)  or (args.qso_ztrue is None):
            parser.error("--qso-cat, --qso-cols, --qso-specid, and --qso-ztrue are " \
                         "required if --qso-dataframe is not passed")
        quasar_catalogue = QuasarCatalogue(args.qso_cat, args.qso_cols,
                                           args.qso_specid, args.qso_hdu).quasar_catalogue()
        quasar_catalogue["loaded"] = False

    # load model
    userprint("Loading model")
    model = Model.from_json(load_json(args.model))

    # initialize candidates object
    userprint("Looking for candidates")
    if args.output_candidates is None:
        candidates = Candidates(mode="test", model=(model, CUTS))
    else:
        candidates = Candidates(mode="test", name=args.output_candidates,
                                model=(model, CUTS))

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

    # compute probabilities
    userprint("Computing probabilities")
    candidates.classify_candidates()

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
            userprint("proba > {}".format(prob))
            candidates.find_completeness_purity(quasar_catalogue,
                                                data_frame[(data_frame["prob"] > prob) &
                                                           ~(data_frame["duplicated"]) &
                                                           (data_frame["z_conf_person"] == 3)],
                                                userprint=userprint)

    # save the catalogue as a fits file
    if not args.no_save_fits:
        found_catalogue = candidates.candidates()
        found_catalogue = found_catalogue[(~found_catalogue["duplicated"]) &
                                          (found_catalogue["prob"] > args.prob_cut)]
        candidates.to_fits(args.output_catalogue, data_frame=found_catalogue)

if __name__ == '__main__':
    main()
