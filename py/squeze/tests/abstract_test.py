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

from squeze.candidates import Candidates
from squeze.common_functions import verboseprint as userprint
from squeze.common_functions import deserialize, load_json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

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
        `test.run_command(["squeze_training.py",
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
        if not equal_df and orig_df.columns.equals(new_df.columns):
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

if __name__ == '__main__':
    pass
