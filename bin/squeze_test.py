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
import time
import numpy as np

from squeze.common_functions import load_json
from squeze.common_functions import deserialize
from squeze.common_functions import verboseprint, quietprint
from squeze.error import Error
from squeze.quasar_catalogue import QuasarCatalogue
from squeze.model import Model
from squeze.spectra import Spectra
from squeze.candidates import Candidates
from squeze.parsers import TEST_PARSER, quasar_parser_check


def main():
    """ Run SQUEzE in test mode """
    # load options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     parents=[TEST_PARSER])
    args = parser.parse_args()
    if args.check_statistics:
        quasar_parser_check(parser, args)

    # manage verbosity
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
        userprint("INFO: time elapsed to load quasar catalogue", (t1-t0)/60.0,
                  'minutes')

    # load model
    userprint("Loading model")
    t2 = time.time()
    if args.model.endswith(".json"):
        model = Model.from_json(load_json(args.model))
    else:
        model = Model.from_fits(args.model)
    t3 = time.time()
    userprint("INFO: time elapsed to load model", (t3-t2)/60.0, 'minutes')

    # initialize candidates object
    userprint("Initializing candidates object")
    if args.output_candidates is None:
        candidates = Candidates(mode="test", model=model)
    else:
        candidates = Candidates(mode="test", name=args.output_candidates,
                                model=model)

    # load candidates dataframe if they have previously looked for
    if args.load_candidates:
        userprint("Loading existing candidates")
        t4 = time.time()
        candidates.load_candidates(args.input_candidates)
        t5 = time.time()
        userprint("INFO: time elapsed to load candidates", (t5-t4)/60.0, 'minutes')

    # load spectra
    if args.input_spectra is not None:
        userprint("Loading spectra")
        t6 = time.time()
        userprint("There are {} files with spectra to be loaded".format(len(args.input_spectra)))
        for index, spectra_filename in enumerate(args.input_spectra):
            userprint("Loading spectra from {} ({}/{})".format(spectra_filename, index,
                                                               len(args.input_spectra)))
            t60 = time.time()
            spectra = Spectra.from_json(load_json(spectra_filename))
            if not isinstance(spectra, Spectra):
                raise Error("Invalid list of spectra")

            # flag loaded quasars as such
            if args.check_statistics:
                for spec in spectra.spectra_list():
                    if quasar_catalogue[
                            quasar_catalogue["SPECID"] == spec.metadata_by_key("SPECID")].shape[0] > 0:
                        index = quasar_catalogue.index[
                            quasar_catalogue["SPECID"] == spec.metadata_by_key("SPECID")].tolist()[0]
                        quasar_catalogue.at[index, "LOADED"] = True

            # look for candidates
            userprint("Looking for candidates")
            candidates.find_candidates(spectra.spectra_list())

            t61 = time.time()
            userprint(f"INFO: time elapsed to find candidates from {spectra_filename}",
                      (t61-t60)/60.0, 'minutes')
        t7 = time.time()
        userprint("INFO: time elapsed to find candidates", (t7-t6)/60.0, 'minutes')

    # compute probabilities
    userprint("Computing probabilities")
    t8 = time.time()
    candidates.classify_candidates()
    t9 = time.time()
    userprint("INFO: time elapsed to classify candidates", (t9-t8)/60.0, 'minutes')

    # check completeness
    if args.check_statistics:
        probs = args.check_probs if args.check_probs is not None else np.arange(0.9, 0.0, -0.05)
        userprint("Check statistics")
        data_frame = candidates.candidates()
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
                                                userprint=userprint)

    # save the catalogue as a fits file
    if not args.no_save_catalogue:
        candidates.save_catalogue(args.output_catalogue, args.prob_cut)

    t8 = time.time()
    userprint("INFO: total elapsed time", (t8-t0)/60.0, 'minutes')
    userprint("Done")

if __name__ == '__main__':
    main()
