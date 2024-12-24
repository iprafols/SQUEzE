"""
    SQUEzE
    ======

    This file contains tests related to the testing mode of SQUEzE
"""
import unittest
import os

from squeze.tests.abstract_test import AbstractTest, SQUEZE_BIN
from squeze.utils import deserialize, load_json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestSquezeTest(AbstractTest):
    """Test the test mode

        CLASS: TestSquezeCandidates
        PURPOSE: Test test mode of squeze
        """

    def test_squeze_test_model_extra_columns(self):
        """ Run run_squeze.py in test mode using a model with extra columns"""
        in_file = f"{THIS_DIR}/data/configs/test_squeze_test_model_extra_columns.ini"
        out_file = "{}/results/test_boss_test2_extra_columns.fits.gz".format(
            THIS_DIR)
        test_file = "{}/data/candidates_boss_test2_pred_extra_columns.fits.gz".format(
            THIS_DIR)

        self.run_squeze(in_file, out_file, test_file)

    def test_squeze_test_nostats(self):
        """ Run run_squeze.py in test mode wihtout computing stastics"""
        in_file = f"{THIS_DIR}/data/configs/test_squeze_test_nostats.ini"
        out_file = "{}/results/test_boss_test2_nostats.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test2_pred.fits.gz".format(
            THIS_DIR)

        self.run_squeze(in_file, out_file, test_file)

    def test_squeze_test_stats(self):
        """ Run run_squeze.py in test mode computing statistics"""
        in_file = f"{THIS_DIR}/data/configs/test_squeze_test_stats.ini"
        out_file = "{}/results/test_boss_test2_stats.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test2_pred.fits.gz".format(
            THIS_DIR)

        self.run_squeze(in_file, out_file, test_file)


if __name__ == '__main__':
    unittest.main()
