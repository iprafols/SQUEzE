"""
    SQUEzE
    ======

    This file provides a peak finder to be used by SQUEzE
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"

import numpy as np

accepted_options = ["min significance", "width"]

defaults = {
    # This variable sets the width (in pixels) of the typical peak to be detected.
    "width": 70,
    # This variable sets the minimum signal-to-noise ratio of a peak.
    "min significance": 6,
}


class PeakFinder:
    """ Create and manage the peak finder used by SQUEzE

    CLASS: PeakFinder
    PURPOSE: Create and manage the peak finder used by SQUEzE. This
    peak finder looks for peaks in a smoothed spectrum by looking for
    points with values higher than their surroundings. It also computes
    the significance of the peaks and filters the results according to
    their significances.
    """

    def __init__(self, config):
        """ Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        self.width = config.getfloat("width")
        if self.width > 0:
            self.fwhm = int(2.355 * self.width)
        else:
            self.fwhm = 2
        self.half_fwhm = int(self.fwhm / 2)
        self.min_significance = config.getfloat("min significance")

    def __find_peak_significance(self, spectrum, index):
        """ Find the significance of the peak.

        The significance is computed by measuring the signal to noise of
        the difference between the peak and the continuum. The peak is
        measured using a window of size FWHM centered at the position of
        the specified index, where FWHM is the full width half maximum of the
        Gaussian used in the smoothing. The continuum is measured using two
        windows of size FWHM/2 at each side of the peak.

        Arguments
        ---------
        spectrum : Spectrum
        The spectrum where peaks are looked for
        """
        flux = spectrum.flux
        ivar = spectrum.ivar

        if index < self.fwhm or index + self.fwhm > flux.size:
            significance = np.nan
        else:
            peak = np.average(flux[index - self.half_fwhm:index +
                                   self.half_fwhm])
            cont = np.average(flux[index - self.fwhm:index - self.half_fwhm])
            cont += np.average(flux[index + self.half_fwhm:index + self.fwhm])
            cont = cont / 2.0
            ivar_diff = np.sum(ivar[index - self.fwhm:index + self.fwhm])
            if ivar_diff != 0.0:
                error = 1.0 / np.sqrt(ivar_diff)
                significance = (peak - cont) / error
            else:
                significance = np.nan

        return significance

    def find_peaks(self, spectrum):
        """ Find significant peaks in a given spectrum.

        Arguments
        ---------
        spectrum : Spectrum
        The spectrum where peaks are looked for

        Return
        ------
        peak_indices: array of int
        An array with the position of the peaks

        peak_significances: array of float
        An array with the significance of the peaks

        best_fit: empty array
        Unused. Included for compatibility with other peak finders.
        """
        # smooth the spectrum
        smoothed_data = spectrum.smooth(self.width)

        # find peaks
        peak_indexs = []
        significances = []
        for index, flux in enumerate(smoothed_data):
            if ((0 < index < smoothed_data.size - 1) and
                (flux > smoothed_data[index + 1]) and
                (flux > smoothed_data[index - 1])):
                # find significance of the peak
                significance = self.__find_peak_significance(spectrum, index)

                # add the peak to the list if the significance is large enough
                if significance >= self.min_significance:
                    peak_indexs.append(index)
                    significances.append(significance)

        # convert list to array
        peak_indexs = np.array(peak_indexs, dtype=int)
        significances = np.array(significances, dtype=float)

        # return
        return peak_indexs, significances, np.array(())
