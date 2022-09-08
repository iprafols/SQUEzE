"""
    SQUEzE
    ======

    This file implements the class Config, that is used to manage the
    run-time options of SQUEzE
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import os
import re
import json
from configparser import ConfigParser

from squeze.error import Error
from squeze.utils import class_from_string, function_from_string

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

        # This variable specifies whether or not to read candidates from file
        "load candidates": "False",

        # This variable sets the number of pixels to each side of the peak to be
        # used as metrics. Ignored if 'pixels as metrics is false'
        "num pixels": "30",

        # This variable contains a list of columns to be passed to the random forest
        # classifier(s). None for no columns. Columns must be in the input catalogue.
        #"pass cols to random forest": "col1 col2",

        # If this variable is True, pixels close to the peak will be passed to
        # the random forest
        "pixels as metrics": "False",

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
        # This variable sets the options to be passed to the random forest classifier
        "random forest options": "$SQUEZE/data/default_random_forest_options.json",
        # This variable sets the random states of the random forest instances
        "random state": "2081487193",
        # This variable specifies if model is saved in json (False) or
        # fits (True) file format
        "fits file": "False",
    }
}


class Config(object):
    """ Manage run-time options for SQUEzE

    CLASS: Config
    PURPOSE: Manage run-time options for SQUEzE
    """

    def __init__(self, filename=None):
        """ Initialize class instance.

        Parameters
        ----------
        filename : str or None - Default: None
        The configuration file. None for default configuration
        """
        self.config = ConfigParser()
        # load default configuration
        self.config.read_dict(default_config)
        # now read the configuration file
        if filename is not None and os.path.isfile(filename):
            self.config.read(filename)
        else:
            print(f"WARNING: Config file not found: {filename}; using default config")

        # load the printing function
        self.load_print_function()

        # parse the environ variables
        self.__parse_environ_variables()

        # parse the peak finder section
        self.__peak_finder = None
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
             accepted_options) = class_from_string(peak_finder_name, module_name)
        except ImportError as error:
            raise Error(
                f"Error loading class {peak_finder_name}, "
                f"module {module_name} could not be loaded") from error
        except AttributeError as error:
            raise Error(
                f"Error loading class {peak_finder_name}, "
                f"module {module_name} did not contain requested class"
            ) from error

        for key in section:
            if key != "name" and key not in accepted_options:
                message = (
                    "Unrecognised option in section [peak finder]. "
                    f"Found: '{key}'. Accepted options are "
                    f"{accepted_options}")
                raise Error(message)

        for key, value in default_args.items():
            if key not in section:
                section[key] = str(value)

        self.peak_finder = (PeakFinderType, section)

    def __parse_environ_variables(self):
        """Read all variables and replaces the enviroment variables for their
        actual values. This assumes that enviroment variables are only used
        at the beggining of the paths.

        Raise
        -----
        ConfigError if an environ variable was not defined
        """
        for section in self.config:
            for key, value in self.config[section].items():
                if value.startswith("$"):
                    pos = value.find("/")
                    if os.getenv(value[1:pos]) is None:
                        raise Error(
                            f"In section [{section}], undefined "
                            f"environment variable {value[1:pos]} "
                            "was found")
                    self.config[section][key] = value.replace(
                        value[:pos], os.getenv(value[1:pos]))

    def get_peak_finder(self):
        """Get the peak finder type and options"""
        return self.peak_finder

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
            raise Error(
                f"Error loading class {peak_finder_name}, "
                f"module {module_name} could not be loaded") from error
        except AttributeError as error:
            raise Error(
                f"Error loading class {peak_finder_name}, "
                f"module {module_name} did not contain requested class"
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
