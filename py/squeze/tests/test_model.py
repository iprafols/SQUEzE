"""
    SQUEzE
    ======

    This file contains tests related to the Model
"""
import unittest
import os

try:
    import sklearn
    module_not_found = False
except ModuleNotFoundError:
    module_not_found = True

from squeze.tests.abstract_test import AbstractTest, SQUEZE_BIN
from squeze.utils import deserialize, load_json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["THIS_DIR"] = THIS_DIR

@unittest.skipIf(module_not_found, ("Skip training tests since sklearn was not"
                                    "installed"))
class TestModel(AbstractTest):
    """Test the Model

        CLASS: TestModel
        PURPOSE: Test the Model
        """
    def test_fits_model(self):
        """ Create a fits model and test it"""

        # first create the model by running squeze_training.py
        in_file = f"{THIS_DIR}/data/configs/test_fits_model_training.ini"
        out_file = "{}/results/training_boss_test1.fits.gz".format(THIS_DIR)
        test_file = f"{THIS_DIR}/data/candidates_boss_test1_nopred.fits.gz"

        self.run_squeze(in_file, out_file, test_file, fits_model=True)

        # now test the model
        in_file = f"{THIS_DIR}/data/configs/test_fits_model_test.ini"
        out_file = "{}/results/test_boss_test2_nostats.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test2_pred.fits.gz".format(THIS_DIR)

        self.run_squeze(in_file, out_file, test_file)

    def test_json_model(self):
        """ Create a json model and test it"""

        # first create the model by running squeze_training.py
        in_file = f"{THIS_DIR}/data/configs/test_json_model_training.ini"
        out_file = "{}/results/training_boss_test1.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test1_nopred.fits.gz".format(THIS_DIR)

        self.run_squeze(in_file, out_file, test_file, json_model=True)

        # now test the model
        in_file = f"{THIS_DIR}/data/configs/test_json_model_test.ini"
        out_file = "{}/results/test_boss_test2_nostats.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test2_pred.fits.gz".format(THIS_DIR)

        self.run_squeze(in_file, out_file, test_file)

    def test_single_random_forest_model(self):
        """ Create a json model and test it"""

        # first create the model by running squeze_training.py
        in_file = f"{THIS_DIR}/data/configs/test_single_random_forest_model_training.ini"
        out_file = "{}/results/training_boss_test1_singlerf.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test1_nopred.fits.gz".format(THIS_DIR)

        self.run_squeze(in_file, out_file, test_file, json_model=True)

        # now test the model
        in_file = f"{THIS_DIR}/data/configs/test_single_random_forest_model_test.ini"
        out_file = "{}/results/test_boss_test2_nostats_singlerf.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test2_pred_singlerf.fits.gz".format(THIS_DIR)

        self.run_squeze(in_file, out_file, test_file)
        
if __name__ == '__main__':
    unittest.main()
