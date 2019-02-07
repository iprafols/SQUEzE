"""
    SQUEzE
    ======

    This file shows an example of how should the svms varaible be formatted.
    
    DEPRECATED
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import numpy as np

from squeze_common_functions import save_pkl

def main():
    """
        This function sets the lines that will be included in each of the SVM
        instances that will be used to determine the probability of the
        candidate being a quasar. It must be a tuple containing two elemets.
        The first element is a dictionary containing an array of the
        lines that each of the SVM instances will consider when attempting the
        classification. If a spectra does not contain valid information (i.e.
        values that are not NaNs) for any of the selected lines in a given
        SVM instance, it will not be considered by that instance, and a probability
        of -1 will be returned. The second element is another dictionary with the
        same keys containing the randoms states that will be used to initialize the
        SVMs
        """ # description of SVMs ... pylint: disable=pointless-string-statement
    # define the svms
    svms = {1: np.array(['lyb_ratio_SN',
                         'lya_ratio_SN',
                         'siiv_ratio_SN',
                         'civ_ratio_SN',
                         'ciii_ratio_SN',
                         'is_line', 'is_correct']),
            2: np.array(['lyb_ratio_SN',
                         'lya_ratio_SN',
                         'siiv_ratio_SN',
                         'civ_ratio_SN',
                         'ciii_ratio_SN',
                         'neiv_ratio_SN',
                         'mgii_ratio_SN',
                         'is_line', 'is_correct']),
            3: np.array(['lya_ratio_SN',
                         'siiv_ratio_SN',
                         'civ_ratio_SN',
                         'ciii_ratio_SN',
                         'neiv_ratio_SN',
                         'mgii_ratio_SN',
                         'is_line', 'is_correct']),
            4: np.array(['siiv_ratio_SN',
                         'civ_ratio_SN',
                         'ciii_ratio_SN',
                         'neiv_ratio_SN',
                         'mgii_ratio_SN',
                         'is_line', 'is_correct']),
            5: np.array(['siiv_ratio_SN',
                         'civ_ratio_SN',
                         'ciii_ratio_SN',
                         'neiv_ratio_SN',
                         'mgii_ratio_SN',
                         'nev_ratio_SN',
                         'oii_ratio_SN',
                         'is_line', 'is_correct']),
            6: np.array(['civ_ratio_SN',
                         'ciii_ratio_SN',
                         'neiv_ratio_SN',
                         'mgii_ratio_SN',
                         'nev_ratio_SN',
                         'oii_ratio_SN',
                         'is_line', 'is_correct']),
            7: np.array(['ciii_ratio_SN',
                         'neiv_ratio_SN',
                         'mgii_ratio_SN',
                         'nev_ratio_SN',
                         'oii_ratio_SN',
                         'is_line', 'is_correct']),
            8: np.array(['ciii_ratio_SN',
                         'neiv_ratio_SN',
                         'mgii_ratio_SN',
                         'nev_ratio_SN',
                         'oii_ratio_SN',
                         'hb_ratio_SN',
                         'oiii_ratio_SN',
                         'is_line', 'is_correct']),
            9: np.array(['mgii_ratio_SN',
                         'nev_ratio_SN',
                         'oii_ratio_SN',
                         'hb_ratio_SN',
                         'oiii_ratio_SN',
                         'is_line', 'is_correct']),
            10: np.array(['mgii_ratio_SN',
                          'nev_ratio_SN',
                          'oii_ratio_SN',
                          'hb_ratio_SN',
                          'oiii_ratio_SN',
                          'ha_ratio_SN',
                          'is_line', 'is_correct']),
           }

    random_states = {1: 2081487193,
                     2: 2302130440,
                     3: 1566237261,
                     4: 3197800101,
                     5: 1478310587,
                     6: 1493514726,
                     7: 2145873089,
                     8: 912267904,
                     9: 689368146,
                     10: 4091585312}


    # save them as a pkl file to be used by SQUEzE
    save_pkl("svms.pkl", (svms, random_states))

if __name__ == '__main__':
    main()
