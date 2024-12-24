"""
    SQUEzE
    ======

    This file contains functions used in the tests
"""
import numpy as np


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
    return amp * np.exp(-(wave - mu)**2. / (2 * sig**2.))


if __name__ == '__main__':
    pass
