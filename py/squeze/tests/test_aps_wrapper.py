"""
    SQUEzE
    ======

    This file contains tests related to the SQUEzE wrapper for ApS
"""
import unittest
import os

from squeze.tests.abstract_test import AbstractTest
from squeze.common_functions import verboseprint as userprint
from squeze.common_functions import deserialize, load_json

try:
    import redrock
    module_not_found = False
except ModuleNotFoundError:
    module_not_found = True

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

@unittest.skipIf(module_not_found, ("Skip ApS wrapper tests since redrock was "
                                    "not installed"))

class TestApsWrapper(AbstractTest):
    """Test the ApS wrapper

        CLASS: TestApsWrapper
        PURPOSE: Test the ApS wrapper
        """

    def setUp(self):
        """Check that the required files are present, otherwise skips the test
        """
        self.in_files = ["{}/data/stacked_1004074.fit".format(THIS_DIR),
                         "{}/data/stacked_1004073.fit".format(THIS_DIR),
                         ]
        for file in self.in_files:
            if not os.path.isfile(file):
                userprint("File '{}' not found; "
                          "Skip ApS wrapper tests".format(file))
                self.skipTest("File '{}' not found; "
                              "Skip ApS wrapper tests".format(file))

        self.redrock_templates = "{}/data/redrock-templates/rrtemplate-qso.fits".format(THIS_DIR)
        if not os.path.exists(self.redrock_templates):
            userprint("Redrock templates not found; Skip ApS wrapper tests")
            self.skipTest("Redrock templates not found; Skip ApS wrapper tests")

        self.redrock_archetypes = "{}/data/redrock-quasar-archetypes".format(THIS_DIR)
        if not os.path.exists(self.redrock_archetypes):
            userprint("Redrock archetypes not found; Skip ApS wrapper tests")
            self.skipTest("Redrock archetypes not found; Skip ApS wrapper "
                          "tests")

        self.model_file = ("{}/data/candidates_boss_test1_nopred_model."
                           "json").format(THIS_DIR)

        self.out_file = "{}/results/aps_results.fits.gz".format(THIS_DIR)
        self.test_file = ("{}/data/candidates_boss_test1_nopred"
                          ".fits.gz").format(THIS_DIR)

        self.srvyconf = "{}/data/weave_cls.json".format(THIS_DIR)
        if not os.path.exists(self.redrock_archetypes):
            userprint("json config file (Pre-defined class type, based on "
                      "TARGSRVY) not found; Skip ApS wrapper tests")
            self.skipTest("json config file (Pre-defined class type, based on "
                          "TARGSRVY) not found; Skip ApS wrapper tests")

    def test_aps_wrapper(self):
        """ Run aps_squeze.py """

        command = ["aps_squeze.py",
                   '--infiles'] + self.in_files
        command += ['--aps_ids', 'None',
                    '--targsrvy', 'WQ,WL',
                    '--targclass', 'None',
                    '--mask_aps_ids', 'None',
                    '--area', 'None',
                    '--mask_areas', 'None',
                    '--wlranges', '3676,6088.25', '5772,9594.25',
                    '--sens_corr', 'True',
                    '--mask_gaps', 'True',
                    '--tellurics', 'False',
                    '--vacuum', 'True',
                    '--fill_gap', 'False',
                    '--arms_ratio', '1.0,0.83',
                    '--join_arms', 'True',
                    '--templates', self.redrock_templates,
                    '--srvyconf', self.srvyconf,
                    '--archetypes', self.redrock_archetypes,
                    '--outpath', "{}/results/".format(THIS_DIR),
                    '--headname', 'test',
                    '--zall', 'True',
                    '--chi2_scan', 'None',
                    '--nminima' , '3',
                    '--fig', 'False',
                    '--cache_Rcsr', 'False',
                    '--debug', 'False',
                    '--overwrite', 'True',
                    '--mp' ,'2',
                    '--model', '{}/data/candidates_boss_test1_nopred_model.json'.format(THIS_DIR),
                    '--prob_cut', '0.3',
                    '--output_catalogue', self.out_file,
                    '--clean_dir', 'False',
                    '--quiet', 'False',
                   ]
        self.run_command(command)
        self.assertTrue(os.path.isfile(self.out_file))
        #self.compare_data_frames(self.test_file, self.out_file)

if __name__ == '__main__':
    unittest.main()
