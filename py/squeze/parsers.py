"""
    SQUEzE
    ======

    This file defines parsers used by SQUEzE. They are parent
    parsers that allow the different executables to load blocks
    of options efficiently.
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"

import argparse


def min_length(nmin):
    """Require a list argument to take a minimum number of items"""

    class RequiredLength(argparse.Action):
        # Inheriting from argparse.Action pylint: disable=too-few-public-methods
        """Require a list argument to take a minimum number of items"""

        def __call__(self, parser, args, values, option_string=None):
            if not nmin <= len(values):
                msg = (f"argument {self.dest.replace('_', '-')} requires at "
                       f"least {nmin} arguments")
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)

    return RequiredLength


def quasar_parser_check(parser, args):
    """Require either --qso-dataframe or all of --qso-cat, --qso-specid,
    and --qso-ztrue are passed as arguments"""
    if args.qso_dataframe is not None:
        if ((args.qso_cat is not None) or (args.qso_specid is not None) or
            (args.qso_ztrue is not None)):
            parser.error("options --qso-cat, --qso-cols, --qso-specid, and --qso-ztrue " \
                         "are incompatible with --qso-dataframe")
    else:
        if (args.qso_cat is None) or (args.qso_specid is
                                      None) or (args.qso_ztrue is None):
            parser.error("--qso-cat, --qso-cols, --qso-specid, and --qso-ztrue are " \
                         "required if --qso-dataframe is not passed")


# PARENT_PARSER contains the common options used by all SQUEzE executables.
# Other, more specific parent_parsers are defined later in this file
PARENT_PARSER = argparse.ArgumentParser(add_help=False)

PARENT_PARSER.add_argument("--quiet",
                           action="store_true",
                           help="""Do not print messages""")

# PEAKFIND_PARSER contains the options passed to the peak finding algorithms
PEAKFIND_PARSER = argparse.ArgumentParser(add_help=False)

PEAKFIND_PARSER.add_argument("--peakfind-width",
                             type=float,
                             default=None,
                             required=False,
                             help="""Width (in pixels) of the tipical peak""")

# QUASAR_CATALOGUE_PARSER contains the common options used to load the quasar catalogue.
QUASAR_CATALOGUE_PARSER = argparse.ArgumentParser(add_help=False)

QUASAR_CATALOGUE_PARSER.add_argument(
    "--qso-dataframe",
    type=str,
    default=None,
    required=False,
    help="""[REQUIRED] Name of the csv file containing the quasar catalogue
                                         formatted into pandas dataframe. Must only contain information
                                         of quasars that will be loaded. Must be present if --qso-cat
                                         is not passed.""")
QUASAR_CATALOGUE_PARSER.add_argument(
    "--qso-cat",
    type=str,
    default=None,
    required=False,
    help="""[REQUIRED] Name of the fits file containig the quasar
                                         catalogue. Must be present if --qso-dataframe is not
                                         passed""")

QUASAR_CATALOGUE_PARSER.add_argument(
    "--qso-cols",
    nargs='+',
    required=False,
    default=[
        "ra", "dec", "thing_id", "plate", "mjd", "fiberid", "z_vi",
        "class_person", "z_conf_person", "boss_target1", "ancillary_target1",
        "ancillary_target2", "eboss_target0"
    ],
    help="""White-spaced list of the data arrays
                                         (of the quasar catalogue) to be loaded. Must be present
                                         only if --qso-cat is passed""")

QUASAR_CATALOGUE_PARSER.add_argument(
    "--qso-hdu",
    type=int,
    default=1,
    required=False,
    help="""[REQUIRED] Number of the Header Data Unit in --qso-cat
                                         where the catalogue is stored.""")

QUASAR_CATALOGUE_PARSER.add_argument(
    "--qso-specid",
    type=str,
    default=None,
    required=False,
    help="""[REQUIRED] Name of the column that will be used as specid.
                                         Must be included in --qso-cols. Must be present
                                         only if --qso-cat is passed""")

QUASAR_CATALOGUE_PARSER.add_argument(
    "--qso-ztrue",
    type=str,
    default=None,
    required=False,
    help="""[REQUIRED] Name of the column that will be used as z_true.
                                         Must be included in --qso-cols. Must be present
                                         only if --qso-cat is passed""")

if __name__ == '__main__':
    pass
