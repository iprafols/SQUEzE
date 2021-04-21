"""
    SQUEzE
    ======

    This file intializes the default values of SQUEzE variables
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import pandas as pd

import numpy as np

"""
This variable sets the characteristics of the lines used by the code.
The first two columns are the line name and wavelength. Their names
must be "line" and "wave". The first column will be used as the index
of the DataFrame
The second two columns specify the wavelength interval where the peak
emission will be estimated. Their names must be "start" and "end".
The following two columns specify the wavelength interval where the
blue end of the peak will be estimated. Their names must be
"blue_start" and "blue_end".
The following two columns specify the wavelength interval where the
red end of the peak will be estimated. Their names must be
"red_start" and "red_end".
Units may be different but Angstroms are suggested.
Any number of lines might be introduced, but the Lyman-alpha line must
always be present and named 'lya'.
The user is responsible to make sure all wavelength are passed with
the same units.

DO NOT MODIFY this value. If another set of lines is to be used,
please define it elsewhere and pass it as an argument when creating
the candidates DataFrame (see README.md)
""" # description of LINE ... pylint: disable=pointless-string-statement
LINES = pd.DataFrame(
    data=[
        ("lyb", 1033.03, 1023.0, 1041.0, 998.0, 1014.0, 1050.0, 1100.0),
        ("lya", 1215.67, 1194.0, 1250.0, 1103.0, 1159.0, 1285.0, 1341.0),
        ("siiv", 1396.76, 1377.0, 1417.0, 1346.0, 1370.0, 1432.0, 1497.0),
        ("civ", 1549.06, 1515.0, 1575.0, 1449.5, 1494.5, 1603.0, 1668.0),
        ("civ_blue", 1549.06, 1515.0, 1549.06, 1449.5, 1494.5, 1603.0, 1668.0),
        ("civ_red", 1549.06, 1549.06, 1575.0, 1449.5, 1494.5, 1603.0, 1668.0),
        ("ciii", 1908.73, 1880.0, 1929.0, 1756.0, 1845.0, 1964.0, 2053.0),
        ("neiv", 2423.83, 2410.0, 2435.0, 2365.0, 2400.0, 2450.0, 2480.0),
        ("mgii", 2798.75, 2768.0, 2816.0, 2610.0, 2743.0, 2851.0, 2984.0),
        ("nev", 3426.84, 3415.0, 3435.0, 3375.0, 3405.0, 3445.0, 3480.0),
        ("oii", 3728.48, 3720.0, 3745.0, 3650.0, 3710.0, 3750.0, 3790.0),
        ("hb", 4862.68, 4800.0, 4910.0, 4700.0, 4770.0, 5030.0, 5105.0),
        ("oiii", 5008.24, 4990.0, 5020.0, 4700.0, 4770.0, 5030.0, 5105.0),
        ("ha", 6564.61, 6480.0, 6650.0, 6320.0, 6460.0, 6750.0, 6850.0),
        ],
    columns=["LINE", "WAVE", "START", "END",
             "BLUE_START", "BLUE_END",
             "RED_START", "RED_END"]
    ).set_index("LINE")

"""
Name of the json and log file (without extension) where the
variable TRY_LINES will be saved
""" # description of TRY_LINE ... pylint: disable=pointless-string-statement
TRY_LINES = ["lya", "civ", "ciii", "mgii", "hb", "ha"]

"""
This variable sets the redshift precision with which the code will assume
a candidate has the correct redshift. The truth table will be constructed
such that any candidates with Z_TRY = Z_TRUE +/- Z_PRECISION
will be considered as a true quasar.
This will be ignored in operation mode.
""" # description of Z_PRECISION ... pylint: disable=pointless-string-statement
Z_PRECISION = 0.15


"""
This variable sets the width (in pixels) of the typical peak to be detected.
This parameter will be passed to the peak finding function. Check the documentation
on the module squeze_peak_finder for more details
""" # description of PEAKFIND_WIDTH ... pylint: disable=pointless-string-statement
PEAKFIND_WIDTH = 70

"""
This variable sets the minimum signal-to-noise ratio of a peak.
This parameter will be passed to the peak finding function. Check the documentation
on the module squeze_peak_finder for more details
""" # description of PEAKFIND_SIG ... pylint: disable=pointless-string-statement
PEAKFIND_SIG = 6

"""
This variable sets the maximum number of peaks per spectrum (-1 for no maximum).
This parameter will be passed to the peak finding function. Check the documentation
on the module squeze_peak_finder for more details
""" # description of PEAKFIND_NPEAKS ... pylint: disable=pointless-string-statement
PEAKFIND_NPEAKS = -1

"""
This variable sets the options to be passed to the random forest classifier
""" # description of RANDOM_FOREST_OPTIONS ... pylint: disable=pointless-string-statement
RANDOM_FOREST_OPTIONS = {"high": {"class_weight": "balanced_subsample",
                                  "n_jobs": 3, "n_estimators": 1000,
                                  "max_depth": 10,},
                         "low": {"class_weight": "balanced_subsample",
                                 "n_jobs": 3, "n_estimators": 1000,
                                 "max_depth": 10,},
                        }

"""
This variable sets the random states of the random forest instances
""" # description of RANDOM_STATE ... pylint: disable=pointless-string-statement
RANDOM_STATE = 2081487193

"""
This variable contains the transcription from numerical predicted class to
named predicted class
""" # description of CLASS_PREDICTED ... pylint: disable=pointless-string-statement
CLASS_PREDICTED = {"star": 1,
                   "quasar": 3,
                   "quasar, wrong z": 35,
                   "quasar, bal": 30,
                   "quasar, bal, wrong z": 305,
                   "galaxy": 4,
                   "galaxy, wrong z": 45,
                  }


if __name__ == '__main__':
    pass
