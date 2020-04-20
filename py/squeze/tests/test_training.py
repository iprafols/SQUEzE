"""
    SQUEzE
    ======

    This file contains tests related to the training mode of SQUEzE
"""
import unittest

try:
    import sklearn
except ModuleNotFoundError:
    raise unittest.SkipTest(("Skip training tests since sklearn was not"
                             "installed"))

class TestTraining(unittest.TestCase):
    """Test the training mode

        CLASS: TestPeakFinder
        PURPOSE: Test training mode of squeze
        """

if __name__ == '__main__':
    unittest.main()
