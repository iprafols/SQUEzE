"""
    SQUEzE
    ======

    This file implements the class Config, that is used to manage the
    run-time options of SQUEzE
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"

import os
import re
from configparser import ConfigParser
import numpy as np

from squeze.error import Error
from squeze.utils import class_from_string, function_from_string

CHECK_PROBS = " ".join([str(item) for item in np.arange(0.9, 0.0, -0.05)])

QSO_COLS = ("ra dec thing_id plate mjd fiberid z_vi class_person z_conf_person "
            "boss_target1 ancillary_target1 ancillary_target2 eboss_target0")

default_config = {
    "general": {
        "mode": "training",
        "output": "SQUEzE_candidates.fits.gz",
        "userprint": "verboseprint",
    },
    "candidates": {
        # This variable sets the characteristics of the lines used by the code.
        # see $SQUEZE/bin/format_lines.py for details
        "lines": "$SQUEZE/data/default_lines.json",

        # This variable specifies a file containing candidates to be read
        #"input candidates": "candidates_file.fits.gz",

        # This variable specifies a file containing spectra to be read
        #"input spectra": "spectra1.json spectra2.json",

        # This variable specifies whether or not to read candidates from file
        "load candidates": "False",

        # This variable sets the number of pixels to each side of the peak to be
        # used as metrics. Ignored if 'pixels as metrics is false'
        "num pixels": "30",

        # If this variable is True, pixels close to the peak will be passed to
        # the random forest
        "pixels as metrics": "False",

        # Only objects with probability >= PROB_CUT will be included in the
        # catalogue
        "prob cut": "0.0",

        # This variable controls whether final catalogue (not including
        # duplicates) is saved
        "save catalogue flag": "False",

        # This variable sets the labels of each peak from where redshfit is
        # derived. Labels must be in "lines"
        "try lines": "lya civ ciii mgii hb ha",

        # This variable sets the redshift precision with which the code will assume
        # a candidate has the correct redshift. The truth table will be constructed
        # such that any candidates with Z_TRY = Z_TRUE +/- Z_PRECISION
        # will be considered as a true quasar.
        "z precision": "0.15",
    },
    "peak finder": {
        "name": "PeakFinder",
    },
    "model": {
        # Filename of the trained model to load
        #"filename" = "trained_model.json"
        #"filename" = "trained_model.fits.gz"

        # This variable contains a list of columns to be passed to the random forest
        # classifier(s). None for no columns. Columns must be in the input catalogue.
        #"pass cols to random forest": "col1 col2",

        # This variable sets the options to be passed to the random forest classifier
        "random forest options":
            "$SQUEZE/data/default_random_forest_options.json",
        # This variable sets the random states of the random forest instances
        "random state":
            "2081487193",
        # This variable specifies if model is saved in json (False) or
        # fits (True) file format
        "fits file":
            "False",
    },
    "stats": {
        # List of probability cuts to check
        "check probs": CHECK_PROBS,
        # This variable controls the check of statistiscs on the candidates
        "run stats": "True",
    },
    "quasar catalogue": {
        # White-spaced list of the data arrays (of the quasar catalogue) to
        # be loaded.
        "columns": QSO_COLS,

        # Name of the fits file containig the quasar catalogue. Must be present
        # if "qso dataframe" is not passed
        #"filename": "qso_cat.fits",

        # Name of the csv file containing the quasar catalogue formatted into
        # pandas dataframe. Must only contain information of quasars that will
        # be loaded. Must be present if "filename" is not.
        #"qso dataframe": "qso_dataframe.json",

        # Number of the Header Data Unit in "filename" where the catalogue is stored.
        "hdu": "1",

        # Name of the column that will be used as specid. Must be included
        # in "columns".
        "specid column": "THINGID",

        # Name of the column that will be used as z_true. Must be included
        # in "columns".
        "ztrue column": "Z_VI",
    },
}


class Config:
    """ Manage run-time options for SQUEzE

    CLASS: Config
    PURPOSE: Manage run-time options for SQUEzE
    """

    def __init__(self, filename="", config_dict=None):
        """ Initialize class instance.

        Parameters
        ----------
        filename : str
        The configuration file

        config: dict or None - Default None
        If not None, ignore the passed file and read configuration from this
        dictionary instead
        """
        self.config = ConfigParser()
        # load default configuration
        self.config.read_dict(default_config)
        # now read the configuration file
        if config_dict is None:
            if os.path.exists(os.path.expandvars(filename)):
                self.config.read(os.path.expandvars(filename))
            else:
                message = f"Config file not found: {filename}; using default config"
                raise Error(message)
        else:
            self.config.read_dict(config_dict)

        # load the printing function
        self.load_print_function()

        # parse the peak finder section
        self.peak_finder = None
        self.__format_peak_finder_section()

    def __format_peak_finder_section(self):
        """Format section [peak finder] into usable data"""
        if "peak finder" not in self.config:
            raise Error("Missing section [peak finder]")
        section = self.config["peak finder"]

        peak_finder_name = section.get("name")
        if peak_finder_name is None:
            raise Error("In section [peak finder], variable 'name' is required")
        module_name = re.sub('(?<!^)(?=[A-Z])', '_', peak_finder_name).lower()
        module_name = f"squeze.{module_name.lower()}"
        try:
            (PeakFinderType, default_args,
             accepted_options) = class_from_string(peak_finder_name,
                                                   module_name)
        except ImportError as error:
            raise Error(f"Error loading class {peak_finder_name}, "
                        f"module {module_name} could not be loaded") from error
        except AttributeError as error:
            raise Error(f"Error loading class {peak_finder_name}, "
                        f"module {module_name} did not contain requested class"
                       ) from error

        for key in section:
            if key != "name" and key not in accepted_options:
                message = ("Unrecognised option in section [peak finder]. "
                           f"Found: '{key}'. Accepted options are "
                           f"{accepted_options}")
                raise Error(message)

        for key, value in default_args.items():
            if key not in section:
                section[key] = str(value)

        self.peak_finder = (PeakFinderType, section)

    def get_section(self, section):
        """Get the required section of the configuration

        Raise
        -----
        ConfigError if an environ variable was not defined

        Raise
        -----
        ConfigError if section is non-existant
        """
        if section not in self.config:
            raise Error(
                f"Unkown section. Expected one of {self.config.keys()}. "
                f"Found {section}")
        return self.config[section]

    def load_print_function(self):
        """Load the printing function"""
        userprint = self.config["general"].get("userprint")
        if userprint is None:
            message = "In section [general], variable 'userprint' is required"
            raise Error(message)
        try:
            self.userprint = function_from_string(userprint, "squeze.utils")
        except ImportError as error:
            raise Error(f"Error loading function {userprint}, "
                        f"module squeze.utils could not be loaded") from error
        except AttributeError as error:
            raise Error(
                f"Error loading function {userprint}, "
                f"module squeze.utils did not contain requested function"
            ) from error

    def set_option(self, section_name, key, value):
        """Sets a key from a section

        Parameters
        ----------
        section_name: str
        The name of the section

        key: str
        The name of the key

        value: str
        The value to be set
        """
        section = self.config[section_name]
        section[key] = value

    def update_from_model(self, other):
        """Update the configuration file to the values read from a trained model

        Arguments
        ---------
        other: Config
        A configuration instance
        """
        # candidates section
        section = self.config["candidates"]
        section_other = other.get_section("candidates")
        for key, value in section_other.items():
            if key not in ["input spectra", "input candidates"]:
                section[key] = value
        # peak finder section
        self.config["peak finder"] = other.get_section("peak finder")
        self.__format_peak_finder_section()
        
        # model section
        self.config["model"] = other.get_section("model")

    def write(self, config_file):
        """Write the configuration to file

        Arguments
        ---------
        config_file: File
        An open file in write mode
        """
        self.config.write(config_file)
