"""
    SQUEzE
    ======

    This file contains tests related to the testing mode of SQUEzE
"""
import unittest
import os

from squeze.tests.abstract_test import AbstractTest, SQUEZE_BIN
from squeze.utils import deserialize, load_json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestSquezeMerge(AbstractTest):
    """Test the merge mode

        CLASS: TestSquezeMerge
        PURPOSE: Test merge mode of squeze
        """

    def test_squeze_merge(self):
        """ Run run_squeze.py in merge mode"""
        in_file = f"{THIS_DIR}/data/configs/test_squeze_merge.ini"
        out_file = "{}/results/merge_boss_test1_test2.fits.gz".format(THIS_DIR)
        test_file = "{}/data/candidates_merge_boss_test1_test2_nopred.fits.gz".format(
            THIS_DIR)

        self.run_squeze(in_file, out_file, test_file)


if __name__ == '__main__':
    unittest.main()
