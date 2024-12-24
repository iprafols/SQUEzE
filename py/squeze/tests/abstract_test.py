# pylint: disable=duplicate-code
"""
    SQUEzE
    ======

    This file contains an abstract class to define functions common to all tests
"""
import unittest
import os
import sys
import numpy as np
import astropy.io.fits as fits

from squeze.candidates_utils import load_df

from squeze.config import Config
from squeze.spectra import Spectra
from squeze.utils import deserialize, load_json
from squeze.utils import verboseprint as userprint

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["THIS_DIR"] = THIS_DIR
SQUEZE = THIS_DIR.split("py/squeze")[0]
os.environ["SQUEZE"] = SQUEZE
SQUEZE_BIN = SQUEZE + "bin/"
if SQUEZE_BIN not in sys.path:
    sys.path.append(SQUEZE_BIN)

import run_squeze


class AbstractTest(unittest.TestCase):
    """Test the training mode

        CLASS: AbstractTest
        PURPOSE: Abstrac test class to define functions used in all tests
        """

    def setUp(self):
        """ Check that the results folder exists and create it
            if it does not."""
        if not os.path.exists("{}/results/".format(THIS_DIR)):
            os.makedirs("{}/results/".format(THIS_DIR))

    def run_squeze(self,
                   filename,
                   out_file,
                   test_file,
                   json_model=False,
                   fits_model=False):
        """ Run a squeze with the specified configuration and check the results

        Arguments
        ---------
        filename : string
        The config file

        out_file: string
        Name of the output file

        test_file: string
        Name of the test file

        json_model: boolean - Default: True
        If True, check for the existance of a json model

        json_model: boolean - Default: True
        If True, check for the existance of a fits model
        """
        command = ["python", f"{SQUEZE_BIN}/run_squeze.py", filename]
        userprint("Running command: ", " ".join(command))
        run_squeze.main(command[2:])

        self.assertTrue(os.path.isfile(out_file))
        if json_model:
            self.assertTrue(
                os.path.isfile(out_file.replace(".fits.gz", "_model.json")))
        if fits_model:
            self.assertTrue(
                os.path.isfile(out_file.replace(".fits.gz", "_model.fits.gz")))
        self.compare_data_frames(test_file, out_file)

    def compare_data_frames(self, orig_file, new_file):
        """ Compares two dataframes to check that they are equal """
        # load dataframes
        orig_df = load_df(orig_file)
        new_df = load_df(new_file)

        # compare them
        equal_df = orig_df.equals(new_df)
        if not equal_df:
            # this bit tests if they are equal within machine z_precision
            are_similar = True
            for col, dtype in zip(orig_df.columns, orig_df.dtypes):
                if not col in new_df.columns:
                    self.report_dataframe_mismatch(orig_file,
                                                   new_file,
                                                   orig_df,
                                                   new_df,
                                                   col,
                                                   missing_col="new")
                if (dtype == "O") and not orig_df[col].equals(new_df[col]):
                    self.report_dataframe_mismatch(orig_file,
                                                   new_file,
                                                   orig_df,
                                                   new_df,
                                                   col,
                                                   equals=True)
                elif not np.allclose(orig_df[col], new_df[col], equal_nan=True):
                    self.report_dataframe_mismatch(orig_file, new_file, orig_df,
                                                   new_df, col)
            for col in new_df.columns:
                if not col in orig_df.columns:
                    self.report_dataframe_mismatch(orig_file,
                                                   new_file,
                                                   orig_df,
                                                   new_df,
                                                   col,
                                                   missing_col="orig")

    def compare_fits(self, orig_file, new_file):
        """Compare two fits files to check that they are equal

        Arguments
        ---------
        orig_file: str
        Control file

        new_file: str
        New file
        """
        # open fits files
        orig_hdul = fits.open(orig_file)
        new_hdul = fits.open(new_file)
        try:
            # compare them
            if not len(orig_hdul) == len(new_hdul):
                self.report_fits_mismatch_hdul(orig_file, new_file, orig_hdul,
                                               new_hdul)

            # loop over HDUs
            for hdu_index, hdu in enumerate(orig_hdul):
                if "EXTNAME" in hdu.header:
                    hdu_name = hdu.header["EXTNAME"]
                else:
                    hdu_name = hdu_index
                # check header
                self.compare_fits_headers(orig_file, new_file,
                                          orig_hdul[hdu_name].header,
                                          new_hdul[hdu_name].header)
                # check data
                self.compare_fits_data(orig_file, new_file, orig_hdul[hdu_name],
                                       new_hdul[hdu_name])
        finally:
            orig_hdul.close()
            new_hdul.close()

    def compare_fits_data(self, orig_file, new_file, orig_hdu, new_hdu):
        """Compare the data of two HDUs

        Arguments
        ---------
        orig_file: str
        Control file. Used only for error reporting

        new_file: str
        New file. Used only for error reporting

        orig_hdu: fits.hdu.table.BinTableHDU or fits.hdu.image.ImageHDU
        Control header

        new_hdu: fits.hdu.table.BinTableHDU or fits.hdu.image.ImageHDU
        New header
        """
        orig_data = orig_hdu.data
        new_data = new_hdu.data

        # Empty HDU
        if orig_data is None:
            if new_data is not None:
                self.report_fits_mismatch_data(orig_file, new_file, orig_data,
                                               new_data,
                                               orig_hdu.header["EXTNAME"])

        # Image HDU
        elif orig_data.dtype.names is None:
            if not np.allclose(orig_data, new_data, equal_nan=True):
                self.report_fits_mismatch_data(orig_file, new_file, orig_data,
                                               new_data,
                                               orig_hdu.header["EXTNAME"])

        # Table HDU
        else:
            for col in orig_data.dtype.names:
                if not col in new_data.dtype.names:
                    self.report_fits_mismatch_data(orig_file,
                                                   new_file,
                                                   orig_data,
                                                   new_data,
                                                   orig_hdu.header["EXTNAME"],
                                                   col=col,
                                                   missing_col="new")
                self.assertTrue(col in new_data.dtype.names)
                # This is passed to np.allclose and np.isclose to properly handle IDs
                if col in ['LOS_ID', 'TARGETID', 'THING_ID']:
                    rtol = 0
                # This is the default numpy rtol value
                else:
                    rtol = 1e-5

                if (np.all(orig_data[col] != new_data[col]) and not np.allclose(
                        orig_data[col], new_data[col], equal_nan=True,
                        rtol=rtol)):
                    self.report_fits_mismatch_data(orig_file,
                                                   new_file,
                                                   orig_data,
                                                   new_data,
                                                   orig_hdu.header["EXTNAME"],
                                                   col=col,
                                                   rtol=rtol)
            for col in new_data.dtype.names:
                if col not in orig_data.dtype.names:
                    self.report_fits_mismatch_data(orig_file,
                                                   new_file,
                                                   orig_data,
                                                   new_data,
                                                   orig_hdu.header["EXTNAME"],
                                                   col=col,
                                                   missing_col="orig")

    def compare_fits_headers(self, orig_file, new_file, orig_header,
                             new_header):
        """Compare the headers of two HDUs

        Arguments
        ---------
        orig_file: str
        Control file. Used only for error reporting

        new_file: str
        New file. Used only for error reporting

        orig_header: fits.header.Header
        Control header

        new_header: fits.header.Header
        New header
        """
        for key in orig_header:
            if key not in new_header:
                self.report_fits_mismatch_header(orig_file,
                                                 new_file,
                                                 orig_header,
                                                 new_header,
                                                 key,
                                                 missing_key="new")
            if key in ["CHECKSUM", "DATASUM", "DATETIME"]:
                continue
            if (orig_header[key] != new_header[key] and
                (isinstance(orig_header[key], str) or
                 not np.isclose(orig_header[key], new_header[key]))):
                self.report_fits_mismatch_header(orig_file, new_file,
                                                 orig_header, new_header, key)
        for key in new_header:
            if key not in orig_header:
                self.report_fits_mismatch_header(orig_file,
                                                 new_file,
                                                 orig_header,
                                                 new_header,
                                                 key,
                                                 missing_key="orig")

    def compare_json_spectra(self, orig_file, new_file):
        """Compares two sets of spectra saved in a json file"""
        orig_spectra = Spectra.from_json(load_json(orig_file))
        orig_spectra_list = orig_spectra.spectra_list
        new_spectra = Spectra.from_json(load_json(new_file))
        new_spectra_list = new_spectra.spectra_list

        self.assertTrue(orig_spectra.size(), new_spectra.size())
        for index in range(orig_spectra.size()):
            self.assertTrue(
                np.allclose(orig_spectra_list[index].wave,
                            new_spectra_list[index].wave))
            self.assertTrue(
                np.allclose(orig_spectra_list[index].flux,
                            new_spectra_list[index].flux))
            self.assertTrue(
                np.allclose(orig_spectra_list[index].ivar,
                            new_spectra_list[index].ivar))

    def report_dataframe_mismatch(self,
                                  orig_file,
                                  new_file,
                                  orig_data,
                                  new_data,
                                  col,
                                  missing_col=None,
                                  rtol=1e-5,
                                  equals=False):
        """Print messages to give more details on a mismatch when comparing
        data frames

        Arguments
        ---------
        orig_file: str
        Control file

        new_file: str
        New file

        orig_data: pd.DataFrame
        Control data

        new_data: pd.DataFrame
        New data

        col: str
        Name of the offending column

        missing_col: "new", "orig" or None - Default: None
        HDU where the key is missing. None if it is present in both

        rtol: float - Default: 1e-5
        Relative tolerance parameter (see documentation for
        numpy.islcose or np.allclose). Ignored if equals is True

        equals: boolean - Default: False
        If True, then compare using a direct equality, otherwise, use np.allclose
        with the specified tolerance
        """
        report_mismatch(orig_file, new_file)

        if missing_col is None:
            print(f"Different values found for column {col}")
            print("original new isclose original-new\n")
            if equals:
                for new, orig in zip(new_data[col], orig_data[col]):
                    print(f"{orig} {new} " f"{orig == new} " f"{orig-new}")
            else:
                for new, orig in zip(new_data[col], orig_data[col]):
                    print(f"{orig} {new} "
                          f"{np.isclose(orig, new, equal_nan=True, rtol=rtol)} "
                          f"{orig-new}")
        else:
            print(f"Column {col} missing in {missing_col} file")

        self.fail()

    def report_fits_mismatch_data(self,
                                  orig_file,
                                  new_file,
                                  orig_data,
                                  new_data,
                                  hdu_name,
                                  col=None,
                                  missing_col=None,
                                  rtol=1e-5):
        """Print messages to give more details on a mismatch when comparing
        data arrays in fits files

        Arguments
        ---------
        orig_file: str
        Control file

        new_file: str
        New file

        orig_data: fits.fitsrec.FITS_rec
        Control data

        new_data: fits.fitsrec.FITS_rec
        New data

        hdu_name: str
        Name of the ofending HDU

        col: str or None - Default: None
        Name of the offending column. None if there are differences
        in the data array in ImageHDUs

        missing_col: "new", "orig" or None - Default: None
        HDU where the key is missing. None if it is present in both

        rtol: float - Default: 1e-5
        Relative tolerance parameter (see documentation for
        numpy.islcose or np.allclose)
        """
        report_mismatch(orig_file, new_file)

        if col is None:
            if orig_data is None:
                print("Data found in new HDU but not in orig HDU")
            else:
                print(f"Different values found for HDU {hdu_name}")
                print("original new isclose original-new\n")
                for new, orig in zip(orig_data, new_data):
                    print(f"{orig} {new} "
                          f"{np.isclose(orig, new, equal_nan=True)} "
                          f"{orig-new}")

        else:
            if missing_col is None:
                print(f"Different values found for column {col} in "
                      f"HDU {hdu_name}")
                print("original new isclose original-new\n")
                for new, orig in zip(new_data[col], orig_data[col]):
                    print(f"{orig} {new} "
                          f"{np.isclose(orig, new, equal_nan=True, rtol=rtol)} "
                          f"{orig-new}")
            else:
                print(
                    f"Column {col} in HDU {hdu_name} missing in {missing_col} file"
                )

        self.fail()

    def report_fits_mismatch_header(self,
                                    orig_file,
                                    new_file,
                                    orig_header,
                                    new_header,
                                    key,
                                    missing_key=None):
        """Print messages to give more details on a mismatch when comparing
        headers in fits files

        Arguments
        ---------
        orig_file: str
        Control file

        new_file: str
        New file

        orig_obj: fits.header.Header
        Control header.

        new_obj: fits.header.Header
        New header

        key: str
        Name of the offending key

        missing_key: "new", "orig" or None - Default: None
        HDU where the key is missing. None if it is present in both
        """
        report_mismatch(orig_file, new_file)

        if missing_key is None:
            if "EXTNAME" in orig_header:
                print(f"\n For header {orig_header['EXTNAME']}")
            else:
                print("\n For nameless header (possibly a PrimaryHDU)")
            print(f"Different values found for key {key}: "
                  f"orig: {orig_header[key]}, new: {new_header[key]}")

        else:
            print(f"key {key} missing in {missing_key} header")

        self.fail()

    def report_fits_mismatch_hdul(self, orig_file, new_file, orig_hdul,
                                  new_hdul):
        """Print messages to give more details on a mismatch when comparing
        HDU lists in fits files

        Arguments
        ---------
        orig_file: str
        Control file

        new_file: str
        New file

        orig_hdul: fits.hdu.hdulist.HDUList
        Control HDU list

        new_hdul: fits.hdu.hdulist.HDUList
        New HDU list
        """
        report_mismatch(orig_file, new_file)

        print("Different number of extensions found")
        print("orig_hdul.info():")
        orig_hdul.info()
        print("new_hdul.info():")
        new_hdul.info()

        self.fail()


def report_mismatch(orig_file, new_file):
    """Print messages to give more details on a mismatch when comparing
    files

    Arguments
    ---------
    orig_file: str
    Control file

    new_file: str
    New file
    """
    print(f"\nOriginal file: {orig_file}")
    print(f"New file: {new_file}")


if __name__ == '__main__':
    pass
