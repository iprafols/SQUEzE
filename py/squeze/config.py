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
from squeze.common_functions import class_from_string

default_config = {
    "general": {
        "userprint": "verboseprint",
        "mode": "training",
        "output": "SQUEzE_candidates.fits.gz",
    },
    "candidates": {
        # This variable sets the characteristics of the lines used by the code.
        # see $SQUEZE/bin/format_lines.py for details
        "lines": "$SQUEZE/data/default_lines.json",
        # This variable sets the labels of each peak from where redshfit is
        # derived. Labels must be in "lines"
        "try lines": "lya civ ciii mgii hb ha",
        # This variable sets the redshift precision with which the code will assume
        # a candidate has the correct redshift. The truth table will be constructed
        # such that any candidates with Z_TRY = Z_TRUE +/- Z_PRECISION
        # will be considered as a true quasar.
        "z precision": "0.15",
        # If this variable is True, pixels close to the peak will be passed to
        # the random forest
        "pixels as metrics": "False",
        # This variable sets the number of pixels to each side of the peak to be
        # used as metrics. Ignored if 'pixels as metrics is false'
        "num pixels": "30",
        # This variable contains a list of columns to be passed to the random forest
        # classifier(s). None for no columns. Columns must be in the input catalogue.
        "pass cols to random forest": "None",
    },
    "peak finder": {
        "name": "PeakFinder",
        "width": "70",
        "min significance": "6",
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

        # parse the environ variables
        self.__parse_environ_variables()

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
        return self.__peak_finder

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
