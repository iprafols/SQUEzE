# pylint: disable=duplicate-code
"""
    SQUEzE
    ======

    This file contains an abstract class to define functions common to all tests
"""
import unittest
import os
import sys
import subprocess
import numpy as np
import astropy.io.fits as fits

from squeze.candidates import Candidates
from squeze.common_functions import verboseprint as userprint
from squeze.common_functions import deserialize, load_json
from squeze.spectra import Spectra


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SQUEZE_BIN = THIS_DIR.split("py/squeze")[0]+"bin/"
if SQUEZE_BIN not in sys.path:
    sys.path.append(SQUEZE_BIN)


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

    def run_command(self, command, module):
        """ Run a specified command and check it is completed properly

        Parameters
        ----------
        command : list
        A list of items with the script to run and its options

        module : package
        A module with a function main that accepts a list of arguments

        Examples
        --------
        Assuming test is an instance inheriting from AbstractTest:
        `test.run_command(["python"
                           f"{SQUEZE_BIN}/squeze_training.py",
                           "--output-candidates",
                           out_file,
                           "--input-spectra",
                           in_file,
                           ], squeze_training)`
        """
        userprint("Running command: ", " ".join(command))
        module.main(command[2:])

    def compare_data_frames(self, orig_file, new_file):
        """ Compares two dataframes to check that they are equal """
        # load dataframes
        orig_candidates = Candidates()
        orig_candidates.load_candidates(orig_file)
        orig_df = orig_candidates.candidates()
        new_candidates = Candidates()
        new_candidates.load_candidates(new_file)
        new_df = new_candidates.candidates()

        # compare them
        equal_df = orig_df.equals(new_df)
        if not equal_df:
            # this bit tests if they are equal within machine z_precision
            are_similar = True
            for col, dtype in zip(orig_df.columns, orig_df.dtypes):
                self.assertTrue(col in new_df.columns)
                if (dtype == "O"):
                    self.assertTrue(orig_df[col].equals(new_df[col]))
                else:
                    self.assertTrue(np.allclose(orig_df[col], new_df[col],
                                    equal_nan=True))
            for col in new_df.columns:
                self.assertTrue(col in orig_df.columns)

    def compare_fits(self, orig_file, new_file):
        """ Compares two fits files to check that they are equal """
        # open fits files
        orig_hdul = fits.open(orig_file)
        new_hdul = fits.open(new_file)

        # compare them
        self.assertTrue(len(orig_hdul), len(new_hdul))
        # loop over HDUs
        for hdu_index, _ in enumerate(orig_hdul):
            # check header
            orig_header = orig_hdul[hdu_index].header
            new_header = new_hdul[hdu_index].header
            for key in orig_header:
                self.assertTrue(key in new_header)
                if not key in ["CHECKSUM", "DATASUM"]:
                    self.assertTrue((orig_header[key]==new_header[key]) or
                                    (np.isclose(orig_header[key], new_header[key])))

            for key in new_header:
                if key not in orig_header:
                    print(f"key {key} missing in orig header")
                self.assertTrue(key in orig_header)

            # check data
            orig_data = orig_hdul[hdu_index].data
            new_data = new_hdul[hdu_index].data
            if orig_data is None:
                self.assertTrue(new_data is None)
            else:
                for col in orig_data.dtype.names:
                    self.assertTrue(col in new_data.dtype.names)
                    self.assertTrue(((orig_data[col] == new_data[col]).all()) or
                                    (np.allclose(orig_data[col],
                                                 new_data[col],
                                                 equal_nan=True)))
                for col in new_data.dtype.names:
                    if col not in orig_data.dtype.names:
                        print(f"Columns {col} missing in orig header")
                    self.assertTrue(col in orig_data.dtype.names)

    def compare_json_spectra(self, orig_file, new_file):
        """Compares two sets of spectra saved in a json file"""
        orig_spectra = Spectra.from_json(load_json(orig_file))
        orig_spectra_list = orig_spectra.spectra_list()
        new_spectra = Spectra.from_json(load_json(new_file))
        new_spectra_list = new_spectra.spectra_list()

        self.assertTrue(orig_spectra.size(), new_spectra.size())
        for index in range(orig_spectra.size()):
            self.assertTrue(np.allclose(orig_spectra_list[index].wave(),
                                        new_spectra_list[index].wave()))
            self.assertTrue(np.allclose(orig_spectra_list[index].flux(),
                                        new_spectra_list[index].flux()))
            self.assertTrue(np.allclose(orig_spectra_list[index].ivar(),
                                        new_spectra_list[index].ivar()))


if __name__ == '__main__':
    pass
