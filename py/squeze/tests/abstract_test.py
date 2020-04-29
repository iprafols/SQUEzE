# pylint: disable=duplicate-code
"""
    SQUEzE
    ======

    This file contains an abstract class to define functions common to all tests
"""
import unittest
import os
import subprocess

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
        orig_df = deserialize(load_json(orig_file))
        new_df = deserialize(load_json(new_file))
        self.assertTrue(orig_df.equals(new_df))

if __name__ == '__main__':
    pass
