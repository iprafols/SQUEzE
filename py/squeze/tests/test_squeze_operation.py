"""
    SQUEzE
    ======

    This file contains tests related to the operation mode of SQUEzE
"""
import unittest
import os

from squeze.tests.abstract_test import AbstractTest, SQUEZE_BIN
from squeze.utils import deserialize, load_json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestSquezeOperation(AbstractTest):
    """Test the operation mode

        CLASS: TestSquezeOperation
        PURPOSE: Test operation mode of squeze
        """
    def test_squeze_operation(self):
        """ Run run_squeze.py in operation mode"""
        in_file = f"{THIS_DIR}/data/configs/test_squeze_operation.ini"
        out_file = "{}/results/operation_boss_test2.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_operation_boss_test2_pred.fits.gz".format(THIS_DIR)

        self.run_squeze(in_file, out_file, test_file)
        
if __name__ == '__main__':
    unittest.main()
