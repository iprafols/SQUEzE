"""
    SQUEzE
    ======

    This file contains tests related to the Peak Finder
"""
import unittest
from configparser import ConfigParser
import numpy as np

from squeze.peak_finder import PeakFinder
from squeze.peak_finder_power_law import PeakFinderPowerLaw
from squeze.peak_finder_two_power_law import PeakFinderTwoPowerLaw
from squeze.tests.peakfinder_test_examples import (
    BEST_FIT_POWER_LAW,
    MIN_SIGNIFICANCE_POWER_LAW,
    PEAK_INDICES_POWER_LAW,
    SIGNIFICANCES_POWER_LAW,
    BEST_FIT_TWO_POWER_LAW,
    MIN_SIGNIFICANCE_TWO_POWER_LAW,
    PEAK_INDICES_TWO_POWER_LAW,
    SIGNIFICANCES_TWO_POWER_LAW,
    NOISELESS_FLAT_SPEC,
    SMOOTHING_WIDTH,
    MIN_SIGNIFICANCE,
    PEAK_INDICES,
    SIGNIFICANCES,
    SMOOTHING_WIDTH_CUT,
    MIN_SIGNIFICANCE_CUT,
    PEAK_INDICES_CUT,
    SIGNIFICANCES_CUT,
    SMOOTHING_WIDTH_SMOOTH,
    MIN_SIGNIFICANCE_SMOOTH,
    PEAK_INDICES_SMOOTH,
    SIGNIFICANCES_SMOOTH,
    NOISELESS_POWER_LAW_SPEC,
    NOISELESS_TWO_POWER_LAW_SPEC,
)


class TestPeakFinder(unittest.TestCase):
    """ Test the peak finder.

        CLASS: TestPeakFinder
        PURPOSE: Test the peak finder
        """

    def test_peak_finder(self):
        """ Basic test for class PeakFinder.

        Peak finder is run on a dummy flat noiseless spectrum with three
        peaks. All three peaks should be recovered.
        """
        config = ConfigParser()
        config.read_dict({
            "peak finder": {
                "width": SMOOTHING_WIDTH,
                "min significance": MIN_SIGNIFICANCE,
            }
        })
        peak_finder = PeakFinder(config["peak finder"])
        indices, significances, best_fit = peak_finder.find_peaks(
            NOISELESS_FLAT_SPEC)

        self.assertEqual(indices.size, 3)
        self.assertTrue(np.allclose(indices, PEAK_INDICES))
        self.assertTrue(np.allclose(significances, SIGNIFICANCES))
        self.assertEqual(best_fit.size, 0)

    def test_peak_finder_significance_cut(self):
        """ Test that the PeakFinder significance cut works.

        Peak finder is run on a dummy flat noiseless spectrum with three
        peaks. Only two of the peaks should be recovered, and one
        should be lost due to the high significance cut
        """
        config = ConfigParser()
        config.read_dict({
            "peak finder": {
                "width": SMOOTHING_WIDTH_CUT,
                "min significance": MIN_SIGNIFICANCE_CUT,
            }
        })
        peak_finder = PeakFinder(config["peak finder"])
        indices, significances, best_fit = peak_finder.find_peaks(
            NOISELESS_FLAT_SPEC)

        self.assertEqual(indices.size, 2)
        self.assertTrue(np.allclose(indices, PEAK_INDICES_CUT))
        self.assertTrue(np.allclose(significances, SIGNIFICANCES_CUT))
        self.assertEqual(best_fit.size, 0)

    def test_peak_finder_smoothing(self):
        """ Test that the PeakFinder smoothing works.

        Peak finder is run on a dummy flat noiseless spectrum with three
        peaks. Only two of the peaks should be recovered, since the
        first two should be smoothed into a single peak
        """
        config = ConfigParser()
        config.read_dict({
            "peak finder": {
                "width": SMOOTHING_WIDTH_SMOOTH,
                "min significance": MIN_SIGNIFICANCE_SMOOTH,
            }
        })
        peak_finder = PeakFinder(config["peak finder"])
        indices, significances, best_fit = peak_finder.find_peaks(
            NOISELESS_FLAT_SPEC)

        self.assertEqual(indices.size, 2)
        self.assertTrue(np.allclose(
            indices,
            PEAK_INDICES_SMOOTH,
        ))
        self.assertTrue(np.allclose(significances, SIGNIFICANCES_SMOOTH))
        self.assertEqual(best_fit.size, 0)

    def test_peak_finder_power_law(self):
        """ Basic test for class PeakFinderPowerLaw.

        Peak finder is run on a dummy noiseless spectrum with three
        peaks on top of a power-law. All three peaks should be recovered.
        """
        config = ConfigParser()
        config.read_dict(
            {"peak finder": {
                "min significance": MIN_SIGNIFICANCE_POWER_LAW,
            }})
        peak_finder = PeakFinderPowerLaw(config["peak finder"])
        indices, significances, best_fit = peak_finder.find_peaks(
            NOISELESS_POWER_LAW_SPEC)

        self.assertEqual(indices.size, 3)
        self.assertTrue(np.allclose(indices, PEAK_INDICES_POWER_LAW))
        self.assertTrue(np.allclose(significances, SIGNIFICANCES_POWER_LAW))
        self.assertEqual(best_fit.size, 2)
        self.assertTrue(np.allclose(best_fit, BEST_FIT_POWER_LAW))

    def test_peak_finder_two_power_law(self):
        """ Basic test for class PeakFinderTwoPowerLaw.

        Peak finder is run on a dummy noiseless spectrum with three
        peaks on top of a power-law. All three peaks should be recovered.
        """
        config = ConfigParser()
        config.read_dict({
            "peak finder": {
                "min significance": MIN_SIGNIFICANCE_TWO_POWER_LAW,
            }
        })
        peak_finder = PeakFinderTwoPowerLaw(config["peak finder"])
        indices, significances, best_fit = peak_finder.find_peaks(
            NOISELESS_TWO_POWER_LAW_SPEC)

        self.assertEqual(indices.size, 3)
        self.assertTrue(np.allclose(indices, PEAK_INDICES_TWO_POWER_LAW))
        self.assertTrue(np.allclose(significances, SIGNIFICANCES_TWO_POWER_LAW))
        self.assertEqual(best_fit.size, 4)
        self.assertTrue(np.allclose(best_fit, BEST_FIT_TWO_POWER_LAW))


if __name__ == '__main__':
    unittest.main()
