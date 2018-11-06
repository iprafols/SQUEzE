"""
    SQUEzE
    ======

    This file shows an example of how should the variable cuts be formatted.
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import pandas as pd

from squeze_common_functions import save_pkl

def main():
    """
        This function sets the characteristics of the cuts to be applied in the
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
    cuts = (0.8,
            [["lyb_ratio_SN", "lya_ratio_SN", "siiv_ratio_SN", "civ_ratio_SN",
               "ciii_ratio_SN"],
             ["lya_ratio_SN", "siiv_ratio_SN", "civ_ratio_SN",
               "ciii_ratio_SN", "mgii_ratio_SN"],
            ])

    # save them as a pkl file to be used by SQUEzE
    save_pkl("cuts.pkl", lines)

if __name__ == '__main__':
    main()
