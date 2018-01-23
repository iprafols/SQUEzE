"""
    SQUEzE
    ======

    This file intializes the default values of SQUEzE variables
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import pandas as pd

"""
This function set the cuts to be applied by SQUEzE.
Format is a list with tuples (name, value, type). Type can be "sample-high-cut",
"sample-low-cut", or "percentile" for training mode and "sample-high-cut",
"sample-low-cut", or "min_ratio" for operation mode.
"sample-high-cut" cuts everything with a value higher or equal than the provided value.
"sample-low-cut" cuts everything with a value lower than the provided value.
"percentile" cuts everything with a value lower than the provided percentile.
"min_ratio" cuts everything with a value lower than the provided value, and should
only be used with cuts in line ratios.
""" # description of CUTS_TRAINING ... pylint: disable=pointless-string-statement
CUTS_TRAINING = [
    ("lya_ratio", 1.0, 'percentile'),
    ("civ_ratio", 1.0, 'percentile'),
    ("ciii_ratio", 1.0, 'percentile'),
    ("siiv_ratio", 1.0, 'percentile'),
    ("z_vi", 2.0, 'sample-low-cut')
    ]

CUTS_OPERATION = [
    ("lya_ratio", 0.8, 'min_ratio'),
    ("civ_ratio", 0.8, 'min_ratio'),
    ("ciii_ratio", 0.8, 'min_ratio'),
    ("siiv_ratio", 0.8, 'min_ratio'),
    ("z", 2.0, 'sample-low-cut')
    ]

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
        ("lya", 1215.67, 1194.0, 1250.0, 1103.0, 1159.0, 1285.0, 1341.0),
        ("siiv", 1396.76, 1377.0, 1417.0, 1346.0, 1370.0, 1432.0, 1497.0),
        ("civ", 1549.06, 1515.0, 1575.0, 1449.5, 1494.5, 1603.0, 1668.0),
        ("ciii", 1908.73, 1880.0, 1929.0, 1756.0, 1845.0, 1964.0, 2053.0),
        ("mgii", 2798.75, 2783.0, 2816.0, 2615.0, 2748.0, 2851.0, 2984.0),
        ],
    columns=["line", "wave", "start", "end",
             "blue_start", "blue_end",
             "red_start", "red_end"]
    ).set_index("line")

"""
Name of the pkl and log file (without extension) where the
cuts will be saved
""" # description of TRY_LINE ... pylint: disable=pointless-string-statement
TRY_LINE = "lya"

"""
This variable sets the redshift precision with which the code will assume
a candidate has the correct redshift. The truth table will be constructed
such that any candidates with z_try = z_true +/- Z_PRECISION
will be considered as a true quasar.
This will be ignored in operation mode.
""" # description of Z_PRECISION ... pylint: disable=pointless-string-statement
Z_PRECISION = 0.1

if __name__ == '__main__':
    pass
