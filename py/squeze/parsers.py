"""
    SQUEzE
    ======

    This file defines parsers used by SQUEzE. They are parent
    parsers that allow the different executables to load blocks
    of options efficiently.
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import argparse

def min_length(nmin):
    """Require a list argument to take a minimum number of items"""
    class RequiredLength(argparse.Action):
        # Inheriting from argparse.Action pylint: disable=too-few-public-methods
        """Require a list argument to take a minimum number of items"""
        def __call__(self, parser, args, values, option_string=None):
            if not nmin <= len(values):
                msg = """argument "{f}" requires at leats {nmin}
                    arguments""".format(f=self.dest.replace("_", "-"),
                                        nmin=nmin)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return RequiredLength

"""
This PARENT_PARSER contains the common options used by all SQUEzE executables.
Other, more specific parent_parsers are defined later in this file
""" # description of PARENT_PARSER ... pylint: disable=pointless-string-statement
PARENT_PARSER = argparse.ArgumentParser(add_help=False)

PARENT_PARSER.add_argument("--quiet", action="store_true",
                           help="""Do not print messages""")

"""
This PEAKFIND_PARSER contains the options passed to the peak finding algorithms
""" # description of PEAKFIND_PARSER ... pylint: disable=pointless-string-statement
PEAKFIND_PARSER = argparse.ArgumentParser(add_help=False)

PEAKFIND_PARSER.add_argument("--peakfind-width", type=float, default=None,
                             required=False,
                             help="""Width (in pixels) of the tipical peak""")

PEAKFIND_PARSER.add_argument("--peakfind-sig", type=float, default=None,
                             required=False,
                             help="""Minimum significance required to accept a peak""")


"""
This MODE_PARSER contains the common options for the training, test,
and operation modes
""" # description of MODE_PARSER ... pylint: disable=pointless-string-statement
MODE_PARSER = argparse.ArgumentParser(add_help=False)

MODE_PARSER.add_argument("--input-spectra", nargs='+', type=str, default=None,
                         required=False,
                         help="""Name of the json file(s) containing the spectra
                             that are to be analysed. If multiple files are given,
                             then they must be passed as a white-spaced list.
                             The spectra in each of the files will be loaded into
                             memory as a block, and candidates will be looked for
                             before loading the next set of spectra.""")

MODE_PARSER.add_argument("--load-candidates", action="store_true",
                         help="""Load candidates previously found. If
                             --input-candidates is passed, then load from there.
                             Otherwise, load from --output-candidates.""")

MODE_PARSER.add_argument("--input-candidates", type=str, default=None, required=False,
                         help="""Name of the csv file from where candidates will be
                             loaded.""")

MODE_PARSER.add_argument("--output-candidates", type=str, default=None, required=False,
                         help="""Name of the fits.gz file where the candidates will be saved.
                             In training mode, the model will be saved using this name
                             (without the extension) as base name and append the extension
                             _model.json to it""")

"""
This QUASAR_CATALOGUE_PARSER contains the common options used to load the quasar catalogue.
""" # description of QUASAR_CATALOGUE_PARSER ... pylint: disable=pointless-string-statement
QUASAR_CATALOGUE_PARSER = argparse.ArgumentParser(add_help=False)

QUASAR_CATALOGUE_PARSER.add_argument("--qso-dataframe", type=str, default=None, required=False,
                                     help="""[REQUIRED] Name of the csv file containing the quasar catalogue
                                         formatted into pandas dataframe. Must only contain information
                                         of quasars that will be loaded. Must be present if --qso-cat
                                         is not passed.""")
QUASAR_CATALOGUE_PARSER.add_argument("--qso-cat", type=str, default=None, required=False,
                                     help="""[REQUIRED] Name of the fits file containig the quasar
                                         catalogue. Must be present if --qso-dataframe is not
                                         passed""")

QUASAR_CATALOGUE_PARSER.add_argument("--qso-cols", nargs='+', required=False,
                                     default=["ra", "dec", "thing_id", "plate", "mjd", "fiberid",
                                              "z_vi", "class_person", "z_conf_person", "boss_target1",
                                              "ancillary_target1", "ancillary_target2", "eboss_target0"],
                                     help="""White-spaced list of the data arrays
                                         (of the quasar catalogue) to be loaded. Must be present
                                         only if --qso-cat is passed""")

QUASAR_CATALOGUE_PARSER.add_argument("--qso-hdu", type=int, default=1, required=False,
                                     help="""[REQUIRED] Number of the Header Data Unit in --qso-cat
                                         where the catalogue is stored.""")

QUASAR_CATALOGUE_PARSER.add_argument("--qso-specid", type=str, default=None, required=False,
                                     help="""[REQUIRED] Name of the column that will be used as specid.
                                         Must be included in --qso-cols. Must be present
                                         only if --qso-cat is passed""")

QUASAR_CATALOGUE_PARSER.add_argument("--qso-ztrue", type=str, default=None, required=False,
                                     help="""[REQUIRED] Name of the column that will be used as z_true.
                                         Must be included in --qso-cols. Must be present
                                         only if --qso-cat is passed""")




"""
This TRAINING_PARSER contains the common options used to run SQUEzE in training mode
""" # description of TRAINING_PARSER ... pylint: disable=pointless-string-statement
TRAINING_PARSER = argparse.ArgumentParser(add_help=False,
                                          parents=[PARENT_PARSER,
                                                   MODE_PARSER,
                                                   PEAKFIND_PARSER,
                                                   QUASAR_CATALOGUE_PARSER])

TRAINING_PARSER.add_argument("--z-precision", type=float, default=None, required=False,
                             help="""Maximum difference betwee the true redshift and the
                                 measured redshift for a candidate to be considered a
                                 true detection. This option only works on cuts of
                                 type 'percentile'.""")

TRAINING_PARSER.add_argument("--lines", type=str, default=None, required=False,
                             help="""Name of the json file containing the lines ratios
                                 to be computed.""")

TRAINING_PARSER.add_argument("--try-lines", nargs='*', type=str, default=None, required=False,
                             help="""Name of the lines that will be associated to the peaks
                             to estimate the redshift.""")

TRAINING_PARSER.add_argument("--weighting-mode", type=str, default="weights", required=False,
                             help="""Selects the weighting mode when computing the line ratios.
                                 Can be 'weights' if ivar is to be used as weights when computing
                                 the line ratios, 'flags' if ivar is to be used as flags when
                                 computing the line ratios (pixels with 0 value will be ignored,
                                 the rest will be averaged without weighting), or 'none' if weights
                                 are to be ignored.""")

"""
This OPERATION_PARSER contains the options used to run SQUEzE in operation mode
""" # description of OPERATION_PARSER ... pylint: disable=pointless-string-statement
OPERATION_PARSER = argparse.ArgumentParser(add_help=False, parents=[PARENT_PARSER,
                                                                    MODE_PARSER])

OPERATION_PARSER.add_argument("--prob-cut", default=0.0, type=float,
                              help="""Only objects with probability >= PROB_CUT will be included
                                  in the catalogue""")

OPERATION_PARSER.add_argument("--model", required=True, type=str,
                              help="""[REQUIRED] Name of the json file containing the model to be used
                                  in the computation of the probabilities of candidates
                                  being quasars""")

OPERATION_PARSER.add_argument("--no-save-catalogue", action="store_true",
                              help="""Do not save the final catalogue excluding duplicated candidates
                               and those candidates with probability < PROB_CUT """)

OPERATION_PARSER.add_argument("--output-catalogue", default=None, required=False, type=str,
                              help="""Name of the fits file where the final catalogue will be
                               stored. If not specified, the catalogue will be saved using
                               --output-candidates as name base, Ignored if --save-fits is
                               not passed""")

"""
This TEST_PARSER contains the common options used to run SQUEzE in training mode
""" # description of TRAINING_PARSER ... pylint: disable=pointless-string-statement
TEST_PARSER = argparse.ArgumentParser(add_help=False,
                                      parents=[OPERATION_PARSER,
                                               QUASAR_CATALOGUE_PARSER])

TEST_PARSER.add_argument("--check-statistics", action="store_true",
                         help="""Check the candidates' statistics at the end""")

TEST_PARSER.add_argument("--check-probs", nargs='+', default=None, required=False,
                         type=float,
                         help="""White-spaced list of the probabilities to check.
                             The candidates' statistics will be computed for these
                             cuts in probabilities. Ignored if --check-statistics
                             is not passed. If it is not passed and --check-statistics
                             is then np.arange(0.9, 0.0, -0.05)""")

"""
This MERGING_PARSER contains the options used to run SQUEzE in merging mode
""" # description of MERGING_PARSER ... pylint: disable=pointless-string-statement
MERGING_PARSER = argparse.ArgumentParser(add_help=False, parents=[PARENT_PARSER])

MERGING_PARSER.add_argument("--input-candidates", nargs='+', default=None, required=True,
                            action=min_length(2),
                            help="""[REQUIRED] List of fits files containing candidates objects to
                                merge.""")

MERGING_PARSER.add_argument("--output-candidates", type=str, default=None, required=False,
                            help="""Name of the fits file where the candidates will be saved.""")


if __name__ == '__main__':
    pass
