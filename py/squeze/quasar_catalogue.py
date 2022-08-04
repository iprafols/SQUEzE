"""
    SQUEzE
    ======

    This file implements the class QuasarCatalogue, that is used to load the
    quasar catalogue and format it accordingly
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import fitsio

import numpy as np

from squeze.common_functions import join_struct_arrays


class QuasarCatalogue(object):
    """
        Manage the quasar catalogue.

        CLASS: QuasarCatalogue
        PURPOSE: Load the quasar catalogue and format it accordingly
        """

    def __init__(self, filename, columns, specid_column, ztrue_column, hdu):
        """
            Initialize instance

            Parameters
            ----------
            filename : str
            Name of the fits file containing the quasar catalogue.

            columns : list of str
            Name of the data arrays to load.

            specid_column : str
            Name of the data array to act as specid

            ztrue_column : str
            Name of the data array to act as z_true

            hdu : int
            Number of the Header Data Unit to load
            """
        catalogue_hdul = fitsio.FITS(filename)
        data = [
            np.array(catalogue_hdul[hdu][col][:],
                     dtype=[(col.upper(), catalogue_hdul[hdu][col][:].dtype)])
            for col in columns
        ]
        data += [
            np.array(catalogue_hdul[hdu][specid_column][:],
                     dtype=[("SPECID", catalogue_hdul[hdu][specid_column][:].dtype)]),
            np.array(catalogue_hdul[hdu][ztrue_column][:],
                     dtype=[("Z_TRUE", catalogue_hdul[hdu][ztrue_column][:].dtype)]),
            np.zeros_like(catalogue_hdul[hdu][specid_column][:],
                          dtype=[("LOADED", np.bool_)])
        ]
        self.__quasar_catalogue = join_struct_arrays(data)
        catalogue_hdul.close()

    def quasar_catalogue(self):
        """ Access the quasar catalogue """
        return self.__quasar_catalogue.copy()

    def set_value(self, index, col, value):
        """ Sets value in the quasars dataframe

            Parameters
            ----------
            index : row label
            col : column label
            value : scalar value
            """
        self.__quasar_catalogue[index, col] =  value


if __name__ == "__main__":
    pass
