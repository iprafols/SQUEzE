"""
    SQUEzE
    ======

    This file contains tests related to the Model
"""
import unittest
import os

from squeze.tests.abstract_test import AbstractTest
from squeze.common_functions import deserialize, load_json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

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

        command = ["squeze_training.py",
                   "--peakfind-width", "70",
                   "--peakfind-sig", "6",
                   "--z-precision", "0.15",
                   "--output-candidates",
                   out_file,
                   "--input-spectra",
                   in_file,
                   "--model-fits",
                   ]
        self.run_command(command)
        self.assertTrue(os.path.isfile(out_file))
        self.assertTrue(os.path.isfile(out_file.replace(".fits.gz",
                                                        "_model.fits.gz")))

        # now test the model
        model_file = out_file.replace(".fits.gz", "_model.fits.gz")
        in_file = "{}/data/formatted_boss_test2.json".format(THIS_DIR)
        out_file = "{}/results/test_boss_test2_nostats.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test2_pred.fits.gz".format(THIS_DIR)

        command = ["squeze_test.py",
                   "--model", model_file,
                   "--output-candidates",
                   out_file,
                   "--input-spectra",
                   in_file,
                   ]
        self.run_command(command)
        self.assertTrue(os.path.isfile(out_file))
        self.compare_data_frames(test_file, out_file)

    def test_json_model(self):
        """ Create a json model and test it"""

        # first create the model by running squeze_training.py
        in_file = "{}/data/formatted_boss_test1.json".format(THIS_DIR)
        out_file = "{}/results/training_boss_test1.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test1_nopred.fits.gz".format(THIS_DIR)

        command = ["squeze_training.py",
                   "--peakfind-width", "70",
                   "--peakfind-sig", "6",
                   "--z-precision", "0.15",
                   "--output-candidates",
                   out_file,
                   "--input-spectra",
                   in_file,
                   ]
        self.run_command(command)
        self.assertTrue(os.path.isfile(out_file))
        self.assertTrue(os.path.isfile(out_file.replace(".fits.gz",
                                                        "_model.json")))

        # now test the model
        model_file = out_file.replace(".fits.gz", "_model.json")
        in_file = "{}/data/formatted_boss_test2.json".format(THIS_DIR)
        out_file = "{}/results/test_boss_test2_nostats.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test2_pred.fits.gz".format(THIS_DIR)

        command = ["squeze_test.py",
                   "--model", model_file,
                   "--output-candidates",
                   out_file,
                   "--input-spectra",
                   in_file,
                   ]
        self.run_command(command)
        self.assertTrue(os.path.isfile(out_file))
        self.compare_data_frames(test_file, out_file)

if __name__ == '__main__':
    unittest.main()
