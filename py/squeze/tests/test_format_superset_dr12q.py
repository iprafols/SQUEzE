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
class TestFormatSupersetDr12Q(AbstractTest):
    """Test format superset DR12Q

        CLASS: TestFormatSupersetDr12Q
        PURPOSE: Test formating of sdss quasars
        """
    def test_format_superset_dr12q_dataset1(self):
        """ Run format_superset_dr12q.py """
        out_file = "{}/results/formatted_boss_test1_plate7102.json".format(THIS_DIR)
        test_file = "{}/data/formatted_boss_test1.json".format(THIS_DIR)

        command = ["python",
                   f"{SQUEZE_BIN}/format_superset_dr12q.py",
                   "--qso-cat", f"{THIS_DIR}/data/cat_boss_test1.fits.gz",
                   "--qso-cols", "RA", "DEC", "THING_ID", "PLATE", "MJD",
                   "FIBERID", "Z_VI", "CLASS_PERSON", "Z_CONF_PERSON",
                   "BOSS_TARGET1", "ANCILLARY_TARGET1", "ANCILLARY_TARGET2",
                   "EBOSS_TARGET0",
                   "--qso-specid", "THING_ID",
                   "--qso-ztrue", "Z_VI",
                   "--out", out_file.split("_plate")[0],
                   "--input-folder", f"{THIS_DIR}/data/",
                   "--plate-list", f"{THIS_DIR}/data/platelist_7102.fits",
                   "--sky-mask", f"{THIS_DIR}/data/dr12-sky-mask.txt",
                   ]


        self.run_command(command)
        self.assertTrue(os.path.isfile(out_file))
        self.compare_json_spectra(test_file, out_file)

    def test_format_superset_dr12q_dataset2(self):
        """ Run format_superset_dr12q.py """
        out_file = "{}/results/formatted_boss_test2_plate7102.json".format(THIS_DIR)
        test_file = "{}/data/formatted_boss_test2.json".format(THIS_DIR)

        command = ["python",
                   f"{SQUEZE_BIN}/format_superset_dr12q.py",
                   "--qso-cat", f"{THIS_DIR}/data/cat_boss_test2.fits.gz",
                   "--qso-cols", "RA", "DEC", "THING_ID", "PLATE", "MJD",
                   "FIBERID", "Z_VI", "CLASS_PERSON", "Z_CONF_PERSON",
                   "BOSS_TARGET1", "ANCILLARY_TARGET1", "ANCILLARY_TARGET2",
                   "EBOSS_TARGET0",
                   "--qso-specid", "THING_ID",
                   "--qso-ztrue", "Z_VI",
                   "--out", out_file.split("_plate")[0],
                   "--input-folder", f"{THIS_DIR}/data/",
                   "--plate-list", f"{THIS_DIR}/data/platelist_7102.fits",
                   "--sky-mask", f"{THIS_DIR}/data/dr12-sky-mask.txt",
                   ]


        self.run_command(command)
        self.assertTrue(os.path.isfile(out_file))
        self.compare_json_spectra(test_file, out_file)

if __name__ == '__main__':
    unittest.main()
