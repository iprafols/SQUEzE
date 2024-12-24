"""
    SQUEzE
    ======

    This file contains tests related to the candidates mode of SQUEzE
"""
import unittest
import os

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
        """ Run run_squeze.py in candidates mode using a model """
        in_file = f"{THIS_DIR}/data/configs/test_squeze_candidates_from_model.ini"
        out_file = "{}/results/candidates_boss_test1_frommodel.fits.gz".format(
            THIS_DIR)
        test_file = "{}/data/candidates_boss_test1_nopred.fits.gz".format(
            THIS_DIR)

        self.run_squeze(in_file, out_file, test_file)

    def test_squeze_candidates_from_settings(self):
        """ Run run_squeze.py in candidates mode using specific settings """
        in_file = f"{THIS_DIR}/data/configs/test_squeze_candidates_from_settings.ini"
        out_file = "{}/results/candidates_boss_test2_fromsettings.fits.gz".format(
            THIS_DIR)
        test_file = "{}/data/candidates_boss_test2_nopred.fits.gz".format(
            THIS_DIR)

        self.run_squeze(in_file, out_file, test_file)


if __name__ == '__main__':
    unittest.main()
