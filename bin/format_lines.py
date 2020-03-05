"""
    SQUEzE
    ======

    This file shows an example of how should the lines variable be formatted.
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import pandas as pd

from squeze.common_functions import save_json

def main():
    """
        This function sets the characteristics of the lines used by the code.
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
        """ # description of LINE ... pylint: disable=pointless-string-statement
    # define the lines
    lines = pd.DataFrame(
        data=[
            ("lyb", 1033.03, 1023.0, 1041.0, 998.0, 1014.0, 1050.0, 1100.0),
            ("lya", 1215.67, 1194.0, 1250.0, 1103.0, 1159.0, 1285.0, 1341.0),
            ("siiv", 1396.76, 1377.0, 1417.0, 1346.0, 1370.0, 1432.0, 1497.0),
            ("civ", 1549.06, 1515.0, 1575.0, 1449.5, 1494.5, 1603.0, 1668.0),
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

    # save them as a csv file to be used by SQUEzE
    save_json("lines.csv", lines)

if __name__ == '__main__':
    main()
