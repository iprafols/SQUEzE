"""
    SQUEzE
    ======

    This file implements the class QuasarCatalogue, that is used to load the
    quasar catalogue and format it accordingly
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import astropy.io.fits as fits

import pandas as pd

class QuasarCatalogue(object):
    """
        Manage the quasar catalogue.

        CLASS: QuasarCatalogue
        PURPOSE: Load the quasar catalogue and format it accordingly
        """
    def __init__(self, filename, columns, specid_column, hdu):
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

            hdu : int
            Number of the Header Data Unit to load
            """

        catalogue_hdu = fits.open(filename)
        data = [catalogue_hdu[hdu].data[col].copy() for col in columns]
        data.append(catalogue_hdu[hdu].data[specid_column].copy())
        columns = [col for col in columns]
        columns.append("specid")
        self.__quasar_catalogue = pd.DataFrame(zip(*data), columns=columns)
        del catalogue_hdu[hdu].data
        catalogue_hdu.close()

    def quasar_catalogue(self):
        """ Access the quasar catalogue """
        return self.__quasar_catalogue.copy()

    def set_value(self, index, col, value, takeable=False):
        """ Sets value in the quasars dataframe

            From pd.DataFrame.set_value:
            Put single value at passed column and index

            Parameters
            ----------
            index : row label
            col : column label
            value : scalar value
            takeable : interpret the index/col as indexers, default False
            """
        self.__quasar_catalogue.set_value(index, col, value, takeable)

if __name__ == "__main__":
    pass
