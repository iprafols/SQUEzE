"""
    SQUEzE
    ======

    This file contains tests related to the candidates mode of SQUEzE
"""
import unittest
import os

from squeze.tests.abstract_test import AbstractTest
from squeze.common_functions import verboseprint as userprint
from squeze.common_functions import deserialize, load_json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestSquezeCandidates(AbstractTest):
    """Test the candidates mode

        CLASS: TestSquezeCandidates
        PURPOSE: Test candidates mode of squeze
        """
    def test_squeze_candidates_from_model(self):
        """ Run squeze_candidates.py using a model """

        in_file = "{}/data/formatted_boss_test1.json".format(THIS_DIR)
        out_file = "{}/results/candidates_boss_test1_from_model.json".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test1_nopred.json".format(THIS_DIR)
        model_file = "{}/data/candidates_boss_test1_nopred_model.json".format(THIS_DIR)

        command = ["squeze_candidates.py",
                   "--model", model_file,
                   "--output-candidates",
                   out_file,
                   "--input-spectra",
                   in_file,
                   ]
        self.run_command(command)
        self.assertTrue(os.path.isfile(out_file))
        self.compare_data_frames(test_file, out_file)

    def test_squeze_candidates_from_settings(self):
        """ Run squeze_candidates.py using specific settings """

        in_file = "{}/data/formatted_boss_test1.json".format(THIS_DIR)
        out_file = "{}/results/candidates_boss_test1_from_settings.json".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test1_nopred.json".format(THIS_DIR)

        command = ["squeze_candidates.py",
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
        self.compare_data_frames(test_file, out_file)

if __name__ == '__main__':
    unittest.main()
