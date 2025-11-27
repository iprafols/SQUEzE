"""
    SQUEzE
    ======

    This file contains tests related to the Peak Finder
"""
import unittest
from configparser import ConfigParser
import numpy as np
from numpy.random import randn

from squeze.simple_spectrum import SimpleSpectrum
from squeze.peak_finder import PeakFinder
from squeze.tests.test_utils import gaussian


class TestPeakFinder(unittest.TestCase):
    """Test the peak finder.

        CLASS: TestPeakFinder
        PURPOSE: Test the peak finder
        """

    def setUp(self):
        """Create dummy spectra to test."""

        # create arrays
        wave = np.arange(4000, 9000, 1, dtype=float)
        flux = np.zeros_like(wave)
        ivar = np.ones_like(wave)

        # add three peaks
        self.__peaks_positions = [5000, 5500, 7000]
        self.__peak_amplitudes = [10, 7, 7]
        self.__peak_sigmas = [100, 100, 150]
        for mu, amp, sig in zip(self.__peaks_positions, self.__peak_amplitudes,
                                self.__peak_sigmas):
            flux += gaussian(wave, amp, mu, sig)

        # store the peak indices and significances
        self.__peak_indices = [1002, 1500, 3000]
        self.__significances = [76.4883757, 53.48831078, 30.52738474]
        self.__peak_indices_smooth = [1039, 3000]
        self.__significances_smooth = [95.92189993, 131.96229093]

        # keep the noiseless spectrum
        self.__noiseless_spec = SimpleSpectrum(flux.copy(), ivar.copy(),
                                               wave.copy(), {})

    def test_peak_finder(self):
        """Basic test for the peak finder.

        Peak finder is run on a dummy noiseless spectrum with three
        peaks. All three peaks should be recovered.
        """
        config = ConfigParser()
        config.read_dict({"peak finder": {
            "width": 70,
            "min significance": 6,
        }})
        peak_finder = PeakFinder(config["peak finder"])
        indices, significances, best_fit = peak_finder.find_peaks(
            self.__noiseless_spec)

        self.assertTrue(indices.size == 3)
        self.assertTrue(np.allclose(indices, self.__peak_indices, atol=5))
        self.assertTrue(np.allclose(significances, self.__significances,
                                    atol=1))
        self.assertTrue(best_fit.size == 0)

    def test_significance_cut(self):
        """Test that the significance cut works.

        Peak finder is run on a dummy noiseless spectrum with three
        peaks. Only two of the peaks should be recovered, and one
        should be lost due to the high significance cut
        """
        config = ConfigParser()
        config.read_dict(
            {"peak finder": {
                "width": 70,
                "min significance": 40,
            }})
        peak_finder = PeakFinder(config["peak finder"])
        indices, significances, best_fit = peak_finder.find_peaks(
            self.__noiseless_spec)

        self.assertTrue(indices.size == 2)
        self.assertTrue(np.allclose(indices, self.__peak_indices[:-1], atol=5))
        self.assertTrue(
            np.allclose(significances, self.__significances[:-1], atol=1))
        self.assertTrue(best_fit.size == 0)

    def test_smoothing(self):
        """Test that the smoothing works.

        Peak finder is run on a dummy noiseless spectrum with three
        peaks. Only two of the peaks should be recovered, since the
        first two should be smoothed into a single peak
        """
        config = ConfigParser()
        config.read_dict(
            {"peak finder": {
                "width": 200,
                "min significance": 6,
            }})
        peak_finder = PeakFinder(config["peak finder"])
        indices, significances, best_fit = peak_finder.find_peaks(
            self.__noiseless_spec)

        self.assertTrue(indices.size == 2)
        self.assertTrue(np.allclose(indices, self.__peak_indices_smooth,
                                    atol=5))
        self.assertTrue(
            np.allclose(significances, self.__significances_smooth, atol=1))
        self.assertTrue(best_fit.size == 0)


if __name__ == '__main__':
    unittest.main()
