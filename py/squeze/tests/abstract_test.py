# pylint: disable=duplicate-code
"""
    SQUEzE
    ======

    This file contains an abstract class to define functions common to all tests
"""
import unittest
import os
import subprocess
import numpy as np
import astropy.io.fits as fits

from squeze.candidates import Candidates
from squeze.common_functions import verboseprint as userprint
from squeze.common_functions import deserialize, load_json
from squeze.spectra import Spectra


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

SQUEZE_BIN = THIS_DIR.split("py/squeze")[0]+"bin/"

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

    def run_command(self, command):
        """ Run a specified command and check it is completed properly

        Parameters
        ----------
        command : list
        A list of items with the script to run and its options

        Examples
        --------
        Assuming test is an instance inheriting from AbstractTest:
        `test.run_command(["python"
                           f"{SQUEZE_BIN}/squeze_training.py",
                           "--output-candidates",
                           out_file,
                           "--input-spectra",
                           in_file,
                           ])`
        """
        userprint("Running command: ", " ".join(command))

        with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1,
                              universal_newlines=True) as process:
            for line in process.stdout:
                print(line, end="")

        self.assertEqual(line.strip(), "Done")

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
        orig_columns = sorted(orig_df.columns)
        new_columns = sorted(orig_df.columns)
        if not equal_df and all(orig_columns==new_columns):
            # this bit tests if they are equal within machine z_precision
            are_similar = True
            for col, dtype in zip(orig_df.columns, orig_df.dtypes):
                if (dtype == "O"):
                    if not orig_df[col].equals(new_df[col]):
                        are_similar = False
                        break
                else:
                    if not np.allclose(orig_df[col], new_df[col],
                                       equal_nan=True):
                        are_similar = False
                        break
        else:
            are_similar = False
        self.assertTrue(equal_df or are_similar)

    def compare_fits(self, orig_file, new_file):
        """ Compares two fits files to check that they are equal """
        # open fits files
        orig_hdul = fits.open(orig_file)
        new_hdul = fits.open(new_file)

        # compare them
        self.assertTrue(len(orig_hdul), len(new_hdul))
        # loop over HDUs
        for hdu_index in range(len(orig_hdul)):
            # check header
            orig_header = orig_hdul[1].header
            new_header = new_hdul[1].header
            for key in orig_header:
                self.assertTrue(key in new_header)
                if not key in ["CHECKSUM", "DATASUM"]:
                    self.assertTrue((orig_header[key]==new_header[key]) or
                                    (np.isclose(orig_header[key], new_header[key])))
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
