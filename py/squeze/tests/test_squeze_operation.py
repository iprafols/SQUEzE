"""
    SQUEzE
    ======

    This file contains tests related to the operation mode of SQUEzE
"""
import unittest
import os

from squeze.tests.abstract_test import AbstractTest, SQUEZE_BIN
from squeze.common_functions import deserialize, load_json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

import squeze_operation

class TestSquezeOperation(AbstractTest):
    """Test the operation mode

        CLASS: TestSquezeOperation
        PURPOSE: Test operation mode of squeze
        """
    def test_squeze_operation(self):
        """ Run squeze_operation.py"""

        in_file = "{}/data/formatted_boss_test2.json".format(THIS_DIR)
        out_file = "{}/results/operation_boss_test2.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_operation_boss_test2_pred.fits.gz".format(THIS_DIR)
        model_file = "{}/data/candidates_boss_test1_nopred_model.json".format(THIS_DIR)

        command = ["python",
                   f"{SQUEZE_BIN}/squeze_operation.py",
                   "--model", model_file,
                   "--output-candidates",
                   out_file,
                   "--input-spectra",
                   in_file,
                   ]
        self.run_command(command, squeze_operation)
        self.assertTrue(os.path.isfile(out_file))
        self.compare_data_frames(test_file, out_file)

if __name__ == '__main__':
    unittest.main()
