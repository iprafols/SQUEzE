"""
    SQUEzE
    ======

    This file contains tests related to the training mode of SQUEzE
"""
import unittest
import os

from squeze.tests.abstract_test import AbstractTest, SQUEZE_BIN
from squeze.utils import deserialize, load_json
from squeze.utils import verboseprint as userprint

try:
    import sklearn
    module_not_found = False
except ModuleNotFoundError:
    module_not_found = True

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@unittest.skipIf(module_not_found, ("Skip training tests since sklearn was not"
                                    "installed"))
class TestSquezeTraining(AbstractTest):
    """Test the training mode

        CLASS: TestSquezeTraining
        PURPOSE: Test training mode of squeze
        """

    def test_squeze_training(self):
        """ Run run_squeze.py in training mode """
        in_file = f"{THIS_DIR}/data/configs/test_squeze_training.ini"
        out_file = "{}/results/training_boss_test1.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test1_nopred.fits.gz".format(
            THIS_DIR)

        self.run_squeze(in_file, out_file, test_file, json_model=True)

    def test_squeze_training_with_extra_columns(self):
        """ Run run_squeze.py in training mode with extra columns """
        in_file = f"{THIS_DIR}/data/configs/test_squeze_training_with_extra_columns.ini"
        out_file = "{}/results/training_boss_test1_extra_columns.fits.gz".format(
            THIS_DIR)
        test_file = "{}/data/candidates_boss_test1_nopred.fits.gz".format(
            THIS_DIR)

        self.run_squeze(in_file, out_file, test_file, json_model=True)

    def test_squeze_training_with_single_rf(self):
        """ Run run_squeze.py in training mode with a single rf """
        in_file = f"{THIS_DIR}/data/configs/test_squeze_training_with_single_rf.ini"
        out_file = "{}/results/training_boss_test1_singlerf.fits.gz".format(
            THIS_DIR)
        test_file = "{}/data/candidates_boss_test1_nopred.fits.gz".format(
            THIS_DIR)
        self.run_squeze(in_file, out_file, test_file, json_model=True)


if __name__ == '__main__':
    unittest.main()
