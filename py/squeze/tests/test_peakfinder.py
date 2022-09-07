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

class TestPeakFinder(unittest.TestCase):
    """Test the peak finder.

        CLASS: TestPeakFinder
        PURPOSE: Test the peak finder
        """
    def setUp(self):
        """Create dummy spectra to test."""
        def gaussian(wave, amp, mu, sig):
            """Return a peak with a Gaussian shape

            Parameters
            ----------
            wave : array of floats
            Wavelength array where the peak will be added

            amp : float
            Amplitude of the peak

            mu : float
            Peak center position

            sig : float
            Squared root of the Gaussian variance
            """
            return amp*np.exp(-(wave-mu)**2./(2*sig**2.))

        # create arrays
        wave = np.arange(4000, 9000, 1, dtype=float)
        flux = np.zeros_like(wave)
        ivar = np.ones_like(wave)

        # add three peaks
        self.__peaks_positions = [5000, 5500, 7000]
        self.__peak_amplitudes = [10, 7, 7]
        self.__peak_sigmas = [100, 150, 100]
        for mu, amp, sig in zip(self.__peaks_positions,
                                 self.__peak_amplitudes,
                                 self.__peak_sigmas):
            flux += gaussian(wave, amp, mu, sig)

        # keep the noiseless spectrum
        self.__noiseless_spec = SimpleSpectrum(flux.copy(), ivar.copy(),
                                               wave.copy(), {})

    def test_peak_finder(self):
        """Basic test for the peak finder.

        Peak finder is run on a dummy noiseless spectrum with three
        peaks. All three peaks should be recovered.
        """
        config = ConfigParser()
        config.read_dict({
            "peak finder": {
                "width": 70,
                "min significance": 6,
            }
        })
        peak_finder = PeakFinder(config["peak finder"])
        indexs, significances = peak_finder.find_peaks(self.__noiseless_spec)

        self.assertTrue(indexs.size == 3)

    def test_significance_cut(self):
        """Test that the significance cut works.

        Peak finder is run on a dummy noiseless spectrum with three
        peaks. Only two of the peaks should be recovered, and one
        should be lost due to the high significance cut
        """
        config = ConfigParser()
        config.read_dict({
            "peak finder": {
                "width": 70,
                "min significance": 40,
            }
        })
        peak_finder = PeakFinder(config["peak finder"])
        indexs, significances = peak_finder.find_peaks(self.__noiseless_spec)

        self.assertTrue(indexs.size == 2)

    def test_smoothing(self):
        """Test that the smoothing works.

        Peak finder is run on a dummy noiseless spectrum with three
        peaks. Only two of the peaks should be recovered, since the
        first two should be smoothed into a single peak
        """
        config = ConfigParser()
        config.read_dict({
            "peak finder": {
                "width": 200,
                "min significance": 6,
            }
        })
        peak_finder = PeakFinder(config["peak finder"])
        indexs, significances = peak_finder.find_peaks(self.__noiseless_spec)

        self.assertTrue(indexs.size == 2)

if __name__ == '__main__':
    unittest.main()
