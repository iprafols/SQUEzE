"""
    SQUEzE
    ======

    This file contains tests related to the training mode of SQUEzE
"""
import unittest
import os

from squeze.tests.abstract_test import AbstractTest, SQUEZE_BIN
from squeze.common_functions import verboseprint as userprint
from squeze.common_functions import deserialize, load_json

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
        """ Run squeze_training.py """

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
        self.run_command(command)
        self.assertTrue(os.path.isfile(out_file))
        self.assertTrue(os.path.isfile(out_file.replace(".fits.gz",
                                                        "_model.json")))

        self.compare_data_frames(test_file, out_file)

if __name__ == '__main__':
    unittest.main()
