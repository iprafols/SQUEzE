"""
    SQUEzE
    ======

    This file contains tests related to the training mode of SQUEzE
"""
import unittest
import os
import subprocess

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
class TestTraining(unittest.TestCase):
    """Test the training mode

        CLASS: TestTraining
        PURPOSE: Test training mode of squeze
        """
    def setUp(self):
        """ Check that the results folder exists and create it
            if it does not."""
        if not os.path.exists("{}/results/".format(THIS_DIR)):
            os.makedirs("{}/results/".format(THIS_DIR))

    def test_training(self):
        """ Run training on data from plate 7102 and compare the results
            from the stored candidates """

        command = ["squeze_training.py",
                   "--qso-dataframe",
                   "{}/data/quasars_plate7102.json".format(THIS_DIR),
                   "--peakfind-width", "70",
                   "--peakfind-sig", "6",
                   "--z-precision", "0.15",
                   "--output-candidates",
                   "{}/results/candidates_width70_sig6_plate7102.json".format(THIS_DIR),
                   "--input-spectra",
                   "{}/data/boss_dr12_spectra_plate7102.json".format(THIS_DIR),
                   ]

        with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1,
                              universal_newlines=True) as process:
            for line in process.stdout:
                print(line, end="")

        self.assertEqual(line.strip(), "Done")
        self.assertTrue(os.path.isfile(THIS_DIR+
            "/results/candidates_width70_sig6_plate7102.json"))
        self.assertTrue(os.path.isfile(THIS_DIR+
            "/results/candidates_width70_sig6_plate7102_model.json"))

        orig_file = "data/candidates_width70_sig6_plate7102.json"
        new_file = orig_file.replace("data/", "results/")
        self.compare_results(orig_file, new_file)

    def compare_results(self, orig_file, new_file):
        """ Compares two dataframes to check that they are equal """
        orig_df = deserialize(load_json("{}/{}".format(THIS_DIR, orig_file)))
        new_df = deserialize(load_json("{}/{}".format(THIS_DIR, orig_file)))
        self.assertTrue(orig_df.equals(new_df))

if __name__ == '__main__':
    unittest.main()


#7102, 6715, 4634, 5469, 6116, 6289, 3674, 5055
