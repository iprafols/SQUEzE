"""
    SQUEzE
    ======

    This file contains tests related to the testing mode of SQUEzE
"""
import unittest
import os

import squeze_test
from squeze.tests.abstract_test import AbstractTest, SQUEZE_BIN
from squeze.utils import deserialize, load_json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestSquezeTest(AbstractTest):
    """Test the test mode

        CLASS: TestSquezeCandidates
        PURPOSE: Test test mode of squeze
        """
    def test_squeze_test_model_extra_columns(self):
        """ Run squeze_test.py using a model with extra columns"""

        in_file = "{}/data/formatted_boss_test2.json".format(THIS_DIR)
        out_file = "{}/results/test_boss_test2_extra_columns.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test2_pred_extra_columns.fits.gz".format(THIS_DIR)
        model_file = "{}/data/candidates_boss_test1_extra_columns_model.json".format(THIS_DIR)

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
        self.compare_data_frames(test_file, out_file)

    def test_squeze_test_nostats(self):
        """ Run squeze_test.py wihtout computing stastics"""

        in_file = "{}/data/formatted_boss_test2.json".format(THIS_DIR)
        out_file = "{}/results/test_boss_test2_nostats.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test2_pred.fits.gz".format(THIS_DIR)
        model_file = "{}/data/candidates_boss_test1_nopred_model.json".format(THIS_DIR)

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
        self.compare_data_frames(test_file, out_file)

    def test_squeze_test_stats(self):
        """ Run squeze_test.py computing statistics"""

        in_file = "{}/data/formatted_boss_test2.json".format(THIS_DIR)
        out_file = "{}/results/test_boss_test2_stats.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test2_pred.fits.gz".format(THIS_DIR)
        model_file = "{}/data/candidates_boss_test1_nopred_model.json".format(THIS_DIR)
        qso_file = "{}/data/quasars_boss_test2.json".format(THIS_DIR)

        command = ["python",
                   f"{SQUEZE_BIN}/squeze_test.py",
                   "--model", model_file,
                   "--output-candidates",
                   out_file,
                   "--input-spectra",
                   in_file,
                   "--qso-dataframe",
                   qso_file,
                   "--check-statistics",
                   "--check-probs", "0.0", "0.5", "0.9",
                   ]
        self.run_command(command, squeze_test)
        self.assertTrue(os.path.isfile(out_file))
        self.compare_data_frames(test_file, out_file)

if __name__ == '__main__':
    unittest.main()
