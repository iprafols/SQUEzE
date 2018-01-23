"""
    SQUEzE
    ======

    This file shows an example of how should the cuts varaible be formatted.
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

from squeze_common_functions import save_pkl

def main():
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

        To use this example comment/uncomment the sections referring to training
        and/or operation mode
        """

    # to use training mode uncomment this block and comment the next block
    # define the cuts
    cuts = [
        ("lya_ratio", 1.0, 'percentile'),
        ("civ_ratio", 1.0, 'percentile'),
        ("ciii_ratio", 1.0, 'percentile'),
        ("siiv_ratio", 1.0, 'percentile'),
        ("gmag", 21.7, 'sample-high-cut')
        ]
    # end of block ... pylint: disable=pointless-string-statement

    """# to use operation mode uncomment this block and comment the previous block
    # define the cuts
    cuts = [
        ("lya_ratio", 1.0, 'min_ratio'),
        ("civ_ratio", 1.0, 'min_ratio'),
        ("ciii_ratio", 1.0, 'min_ratio'),
        ("siiv_ratio", 1.0, 'min_ratio'),
        ("mag_g", 21.7, 'sample-high-cut')
        ]
    """ # end of block ... pylint: disable=pointless-string-statement

    # save them as a pkl file to be used by SQUEzE
    save_pkl("cuts.pkl", cuts)

if __name__ == '__main__':
    main()
