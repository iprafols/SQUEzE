"""
    SQUEzE
    ======

    This file implements the class QuasarCatalogue, that is used to load the
    quasar catalogue and format it accordingly
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import fitsio

import pandas as pd

from squeze.error import Error


class QuasarCatalogue:
    """ Manage the quasar catalogue.

    CLASS: QuasarCatalogue
    PURPOSE: Load the quasar catalogue and format it accordingly
    """

    def __init__(self, config):
        """ Initialize instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        self.parse_config(config)

        catalogue_hdul = fitsio.FITS(self.filename)
        data = [catalogue_hdul[self.hdu][col][:] for col in self.columns]
        data.append(catalogue_hdul[self.hdu][self.specid_column][:].copy())
        data.append(catalogue_hdul[self.hdu][self.ztrue_column][:].copy())
        colnames = [col.upper() for col in self.columns]
        colnames.append("SPECID")
        colnames.append("Z_TRUE")
        self.quasar_catalogue = pd.DataFrame(list(zip(*data)), columns=colnames)
        catalogue_hdul.close()

    def parse_config(self, config):
        """ Parse configuration

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        self.filename = config.get("filename")
        if self.filename is None:
            message = "In section [stats], variable 'filename' is required"
            raise Error(message)

        columns = config.get("columns")
        if columns is None:
            message = "In section [stats], variable 'columns' is required"
            raise Error(message)
        self.columns = columns.split()

        self.specid_column = config.get("specid column")
        if self.specid_column is None:
            message = "In section [stats], variable 'specid column' is required"
            raise Error(message)

        self.ztrue_column = config.get("ztrue column")
        if self.ztrue_column is None:
            message = "In section [stats], variable 'ztrue column' is required"
            raise Error(message)

        self.hdu = config.getint("hdu")
        if self.hdu is None:
            message = "In section [stats], variable 'hdu' is required"
            raise Error(message)

    def set_value(self, index, col, value, takeable=False):
        """ Sets value in the quasars dataframe

        From pd.DataFrame.set_value:
        Put single value at passed column and index

        Arguments
        ---------
        index : row label
        col : column label
        value : scalar value
        takeable : interpret the index/col as indexers, default False
        """
        self.quasar_catalogue.set_value(index, col, value, takeable)


if __name__ == "__main__":
    pass
