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

class TestSquezeMerge(AbstractTest):
    """Test the merge mode

        CLASS: TestSquezeMerge
        PURPOSE: Test merge mode of squeze
        """
    def test_squeze_operation(self):
        """ Run squeze_operation.py"""

        in_file1 = "{}/data/candidates_boss_test1_nopred.json".format(THIS_DIR)
        in_file2 = "{}/data/candidates_boss_test2_nopred.json".format(THIS_DIR)
        out_file = "{}/results/merge_boss_test1_test2.json".format(THIS_DIR)
        test_file = "{}/data/candidates_merge_boss_test1_test2_nopred.json".format(THIS_DIR)

        command = ["squeze_merge.py",
                   "--input-candidates",
                   in_file1, in_file2,
                   "--output-candidates",
                   out_file,
                   ]
        self.run_command(command)
        self.assertTrue(os.path.isfile(out_file))
        self.compare_data_frames(test_file, out_file)

if __name__ == '__main__':
    unittest.main()
