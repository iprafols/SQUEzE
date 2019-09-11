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
    columns=["line", "wave", "start", "end",
             "blue_start", "blue_end",
             "red_start", "red_end"]
    ).set_index("line")

"""
Name of the pkl and log file (without extension) where the
cuts will be saved
""" # description of TRY_LINE ... pylint: disable=pointless-string-statement
TRY_LINES = ["lya", "civ", "ciii", "mgii", "hb", "ha"]

"""
This variable sets the redshift precision with which the code will assume
a candidate has the correct redshift. The truth table will be constructed
such that any candidates with z_try = z_true +/- Z_PRECISION
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
This variable sets the options to be passed to the random forest classifier
""" # description of RANDOM_FOREST_OPTIONS ... pylint: disable=pointless-string-statement
RANDOM_FOREST_OPTIONS = {"high": {"num_estimators": 1000, "max_depth": 10, "min_node_record": 2},
                         "low": {"num_estimators": 1000, "max_depth": 10, "min_node_record": 2},
                        }

"""
This variable sets the random states of the random forest instances
""" # description of RANDOM_STATE ... pylint: disable=pointless-string-statement
RANDOM_STATE = 2081487193

"""
This variable sets the lines that will be included in each of the SVM
instances that will be used to determine the probability of the
candidate being a quasar.

DO NOT MODIFY this value. If another set of lines is to be used,
please define it elsewhere and pass it as an argument when creating
the candidates DataFrame (see README.md)

Deprecated.
""" # description of SVMS ... pylint: disable=pointless-string-statement
SVMS = {1: np.array(['lyb_ratio_SN',
                     'lya_ratio_SN',
                     'siiv_ratio_SN',
                     #'civ_ratio_SN',
                     'civ_blue_ratio_SN',
                     'civ_red_ratio_SN',
                     'ciii_ratio_SN',
                     'class_person', 'correct_redshift']),
        #2: np.array(['lyb_ratio_SN',
        #             'lya_ratio_SN',
        #             'siiv_ratio_SN',
        #             #'civ_ratio_SN',
        #             'civ_blue_ratio_SN',
        #             'civ_red_ratio_SN',
        #             'ciii_ratio_SN',
        #             'neiv_ratio_SN',
        #             'mgii_ratio_SN',
        #             'class_person', 'correct_redshift']),
        3: np.array(['lya_ratio_SN',
                     'siiv_ratio_SN',
                     #'civ_ratio_SN',
                     'civ_blue_ratio_SN',
                     'civ_red_ratio_SN',
                     'ciii_ratio_SN',
                     'neiv_ratio_SN',
                     'mgii_ratio_SN',
                     'class_person', 'correct_redshift']),
        4: np.array(['siiv_ratio_SN',
                     #'civ_ratio_SN',
                     'civ_blue_ratio_SN',
                     'civ_red_ratio_SN',
                     'ciii_ratio_SN',
                     'neiv_ratio_SN',
                     'mgii_ratio_SN',
                     'class_person', 'correct_redshift']),
        5: np.array(['siiv_ratio_SN',
                     #'civ_ratio_SN',
                     'civ_blue_ratio_SN',
                     'civ_red_ratio_SN',
                     'ciii_ratio_SN',
                     'neiv_ratio_SN',
                     'mgii_ratio_SN',
                     'nev_ratio_SN',
                     'oii_ratio_SN',
                     'class_person', 'correct_redshift']),
        6: np.array([#'civ_ratio_SN',
                     'civ_blue_ratio_SN',
                     'civ_red_ratio_SN',
                     'ciii_ratio_SN',
                     'neiv_ratio_SN',
                     'mgii_ratio_SN',
                     'nev_ratio_SN',
                     'oii_ratio_SN',
                     'class_person', 'correct_redshift']),
        7: np.array(['ciii_ratio_SN',
                     'neiv_ratio_SN',
                     'mgii_ratio_SN',
                     'nev_ratio_SN',
                     'oii_ratio_SN',
                     'class_person', 'correct_redshift']),
        8: np.array(['ciii_ratio_SN',
                     'neiv_ratio_SN',
                     'mgii_ratio_SN',
                     'nev_ratio_SN',
                     'oii_ratio_SN',
                     'hb_ratio_SN',
                     'oiii_ratio_SN',
                     'class_person', 'correct_redshift']),
        9: np.array(['mgii_ratio_SN',
                     'nev_ratio_SN',
                     'oii_ratio_SN',
                     'hb_ratio_SN',
                     'oiii_ratio_SN',
                     'class_person', 'correct_redshift']),
        10: np.array(['mgii_ratio_SN',
                      'nev_ratio_SN',
                      'oii_ratio_SN',
                      'hb_ratio_SN',
                      'oiii_ratio_SN',
                      'ha_ratio_SN',
                      'class_person', 'correct_redshift']),
       }

"""
This variable sets the random states of the SVM instances. Deprecated.
""" # description of RANDOM_STATES ... pylint: disable=pointless-string-statement
RANDOM_STATES = {1: 2081487193,
                 2: 2302130440,
                 3: 1566237261,
                 4: 3197800101,
                 5: 1478310587,
                 6: 1493514726,
                 7: 2145873089,
                 8: 912267904,
                 9: 689368146,
                 10: 4091585312,
                }


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


"""
This variable sets the characteristics of the cuts to be applied in the
code. The variable is a tuple with two elements.

The first element is the quantile cut. This corresponds to a different
value for each of the metrics used, and this value is computed using the
distribution of the contaminants. The value for this quantile should be
between 0 and 1.

The second element is a list of lists. Each item in
the list is a group of metrics to be considered.
When running SQUEzE elements where the value of all the metrics on one of
the groups is greater than the corresponding value for the specified
quantile will be assigned a probability of 1 and therefore be directly
included in the final catalogue.

The user is responsible to make sure all metrics are valid metrics.
""" # description of CUTS ... pylint: disable=pointless-string-statement
CUTS = (0.8, [["lyb_ratio_SN", "lya_ratio_SN", "siiv_ratio_SN", "civ_ratio_SN",
               "ciii_ratio_SN"],
              ["lya_ratio_SN", "siiv_ratio_SN", "civ_ratio_SN",
               "ciii_ratio_SN", "mgii_ratio_SN"],
              ])
               

    
if __name__ == '__main__':
    pass
