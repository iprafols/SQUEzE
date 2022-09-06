"""
    SQUEzE
    ======

    This file contains tests related to the candidates mode of SQUEzE
"""
import unittest
import os

import squeze_candidates
from squeze.tests.abstract_test import AbstractTest, SQUEZE_BIN
from squeze.utils import deserialize, load_json
from squeze.utils import verboseprint as userprint

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestSquezeCandidates(AbstractTest):
    """Test the candidates mode

        CLASS: TestSquezeCandidates
        PURPOSE: Test candidates mode of squeze
        """
    def test_squeze_candidates_from_model(self):
        """ Run squeze_candidates.py using a model """

        in_file = "{}/data/formatted_boss_test1.json".format(THIS_DIR)
        out_file = "{}/results/candidates_boss_test1_frommodel.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test1_nopred.fits.gz".format(THIS_DIR)
        model_file = "{}/data/candidates_boss_test1_nopred_model.json".format(THIS_DIR)

        command = ["python",
                   f"{SQUEZE_BIN}/squeze_candidates.py",
                   "--model", model_file,
                   "--output-candidates",
                   out_file,
                   "--input-spectra",
                   in_file,
                   ]
        self.run_command(command, squeze_candidates)
        self.assertTrue(os.path.isfile(out_file))
        self.compare_data_frames(test_file, out_file)

    def test_squeze_candidates_from_settings(self):
        """ Run squeze_candidates.py using specific settings """

        in_file = "{}/data/formatted_boss_test2.json".format(THIS_DIR)
        out_file = "{}/results/candidates_boss_test2_fromsettings.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test2_nopred.fits.gz".format(THIS_DIR)

        command = ["python",
                   f"{SQUEZE_BIN}/squeze_candidates.py",
                   "--peakfind-width", "70",
                   "--peakfind-sig", "6",
                   "--z-precision", "0.15",
                   "--output-candidates",
                   out_file,
                   "--input-spectra",
                   in_file,
                   ]
        self.run_command(command, squeze_candidates)
        self.assertTrue(os.path.isfile(out_file))
        self.compare_data_frames(test_file, out_file)

if __name__ == '__main__':
    unittest.main()
