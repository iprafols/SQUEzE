"""This file contains example spectra to test the peak finders."""
import numpy as np

from squeze.simple_spectrum import SimpleSpectrum
from squeze.tests.test_utils import gaussian

# create wavelength array
wave = np.arange(4000, 9000, 1, dtype=float)

# simple spectrum with three peaks and flat continuum
flux = np.zeros_like(wave)
ivar = np.ones_like(wave)
peak_positions = [5000, 5500, 7000]
peak_amplitudes = [10, 7, 7]
peak_sigmas = [100, 100, 150]
for mu, amp, sig in zip(peak_positions, peak_amplitudes, peak_sigmas):
    flux += gaussian(wave, amp, mu, sig)

# keep the spectrum
NOISELESS_FLAT_SPEC = SimpleSpectrum(
    flux, ivar, wave, {})
SMOOTHING_WIDTH = 70
MIN_SIGNIFICANCE = 6
PEAK_INDICES = [1000, 1500, 3000]
SIGNIFICANCES = [76.4883757, 53.48831078, 30.52738474]

# cutting at 40 should remove the last peak
SMOOTHING_WIDTH_CUT = SMOOTHING_WIDTH
MIN_SIGNIFICANCE_CUT = 40
PEAK_INDICES_CUT = PEAK_INDICES[:2]
SIGNIFICANCES_CUT = SIGNIFICANCES[:2]

# smoothing with a wide kernel should merge the first two peaks
SMOOTHING_WIDTH_SMOOTH = 200
MIN_SIGNIFICANCE_SMOOTH = MIN_SIGNIFICANCE
PEAK_INDICES_SMOOTH = [1039, 3000]
SIGNIFICANCES_SMOOTH = [95.92189993, 131.96229093]

# simple spectrum with three peaks and single power-law continuum
flux = 1e8 * wave**-2
ivar = np.ones_like(wave)
peak_positions = [5000, 5500, 7000]
peak_amplitudes = [10, 7, 7]
peak_sigmas = [50, 50, 100]
for mu, amp, sig in zip(peak_positions, peak_amplitudes, peak_sigmas):
    flux += gaussian(wave, amp, mu, sig)

# keep the spectrum
NOISELESS_POWER_LAW_SPEC = SimpleSpectrum(
    flux, ivar, wave, {})
MIN_SIGNIFICANCE_POWER_LAW = 2
PEAK_INDICES_POWER_LAW = [1000, 1500, 3000]
SIGNIFICANCES_POWER_LAW = [1132.29694149,  754.85128291, 1529.2492195]
BEST_FIT_POWER_LAW = [1.09605851e+08, 2.00703770e+00]
