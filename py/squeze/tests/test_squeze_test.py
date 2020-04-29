"""
    SQUEzE
    ======

    This file contains tests related to the testing mode of SQUEzE
"""
import unittest
import os

from squeze.tests.abstract_test import AbstractTest
from squeze.common_functions import deserialize, load_json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestSquezeCandidates(AbstractTest):
    """Test the training mode

        CLASS: TestSquezeCandidates
        PURPOSE: Test candidatse mode of squeze
        """
    def test_squeze_test_nostats(self):
        """ Run squeze_test.py wihtout computing stastics"""

        in_file = "{}/data/formatted_boss_test2.json".format(THIS_DIR)
        out_file = "{}/results/test_boss_test2_nostats.json".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test2_pred.json".format(THIS_DIR)
        model_file = "{}/data/candidates_boss_test1_nopred_model.json".format(THIS_DIR)

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

    def test_squeze_test_stats(self):
        """ Run squeze_test.py computing stastics"""

        in_file = "{}/data/formatted_boss_test2.json".format(THIS_DIR)
        out_file = "{}/results/test_boss_test2_stats.json".format(THIS_DIR)
        test_file = "{}/data/candidates_boss_test2_pred.json".format(THIS_DIR)
        model_file = "{}/data/candidates_boss_test1_nopred_model.json".format(THIS_DIR)
        qso_file = "{}/data/quasars_boss_test2.json".format(THIS_DIR)

        command = ["squeze_test.py",
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
        self.run_command(command)
        self.assertTrue(os.path.isfile(out_file))
        self.compare_data_frames(test_file, out_file)

if __name__ == '__main__':
    unittest.main()
