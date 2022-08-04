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
from squeze.common_functions import deserialize, load_json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

import squeze_training
import squeze_test

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
        in_file = "{}/data/formatted_boss_test1.json".format(THIS_DIR)
        out_file = "{}/results/training_boss_test1.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test1_nopred.fits.gz".format(THIS_DIR)

        command = ["python",
                   f"{SQUEZE_BIN}/squeze_training.py",
                   "--peakfind-width", "70",
                   "--peakfind-sig", "6",
                   "--z-precision", "0.15",
                   "--output-candidates",
                   out_file,
                   "--input-spectra",
                   in_file,
                   "--model-fits",
                   ]
        self.run_command(command, squeze_training)
        self.assertTrue(os.path.isfile(out_file))
        self.assertTrue(os.path.isfile(out_file.replace(".fits.gz",
                                                        "_model.fits.gz")))

        # now test the model
        model_file = out_file.replace(".fits.gz", "_model.fits.gz")
        in_file = "{}/data/formatted_boss_test2.json".format(THIS_DIR)
        out_file = "{}/results/test_boss_test2_nostats.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test2_pred.fits.gz".format(THIS_DIR)

        command = ["python",
                   f"{SQUEZE_BIN}/squeze_test.py",
                   "--model", model_file,
                   "--output-candidates",
                   out_file,
                   "--input-spectra",
                   in_file,
                   ]
        self.run_command(command, squeze_test)
        self.assertTrue(os.path.isfile(out_file))
        self.compare_candidates(test_file, out_file)

    def test_json_model(self):
        """ Create a json model and test it"""

        # first create the model by running squeze_training.py
        in_file = "{}/data/formatted_boss_test1.json".format(THIS_DIR)
        out_file = "{}/results/training_boss_test1.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test1_nopred.fits.gz".format(THIS_DIR)

        command = ["python",
                   f"{SQUEZE_BIN}/squeze_training.py",
                   "--peakfind-width", "70",
                   "--peakfind-sig", "6",
                   "--z-precision", "0.15",
                   "--output-candidates",
                   out_file,
                   "--input-spectra",
                   in_file,
                   ]
        self.run_command(command, squeze_training)
        self.assertTrue(os.path.isfile(out_file))
        self.assertTrue(os.path.isfile(out_file.replace(".fits.gz",
                                                        "_model.json")))
        self.compare_candidates(test_file, out_file)

        # now test the model
        model_file = out_file.replace(".fits.gz", "_model.json")
        in_file = "{}/data/formatted_boss_test2.json".format(THIS_DIR)
        out_file = "{}/results/test_boss_test2_nostats.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test2_pred.fits.gz".format(THIS_DIR)

        command = ["python",
                   f"{SQUEZE_BIN}/squeze_test.py",
                   "--model", model_file,
                   "--output-candidates",
                   out_file,
                   "--input-spectra",
                   in_file,
                   ]
        self.run_command(command, squeze_test)
        self.assertTrue(os.path.isfile(out_file))
        self.compare_candidates(test_file, out_file)

    def test_single_random_forest_model(self):
        """ Create a json model and test it"""

        # first create the model by running squeze_training.py
        in_file = "{}/data/formatted_boss_test1.json".format(THIS_DIR)
        out_file = "{}/results/training_boss_test1_singlerf.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test1_nopred.fits.gz".format(THIS_DIR)
        rf_options_file = "{}/data/singlerf_options.json".format(THIS_DIR)

        command = ["python",
                   f"{SQUEZE_BIN}/squeze_training.py",
                   "--peakfind-width", "70",
                   "--peakfind-sig", "6",
                   "--z-precision", "0.15",
                   "--output-candidates",
                   out_file,
                   "--input-spectra",
                   in_file,
                   "--random-forest-options",
                   rf_options_file,
                   ]
        self.run_command(command, squeze_training)
        self.assertTrue(os.path.isfile(out_file))
        self.assertTrue(os.path.isfile(out_file.replace(".fits.gz",
                                                        "_model.json")))
        self.compare_candidates(test_file, out_file)

        # now test the model
        model_file = out_file.replace(".fits.gz", "_model.json")
        in_file = "{}/data/formatted_boss_test2.json".format(THIS_DIR)
        out_file = "{}/results/test_boss_test2_nostats_singlerf.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test2_pred_singlerf.fits.gz".format(THIS_DIR)

        command = ["python",
                   f"{SQUEZE_BIN}/squeze_test.py",
                   "--model", model_file,
                   "--output-candidates",
                   out_file,
                   "--input-spectra",
                   in_file,
                   ]
        self.run_command(command, squeze_test)
        self.assertTrue(os.path.isfile(out_file))
        self.compare_candidates(test_file, out_file)

if __name__ == '__main__':
    unittest.main()
