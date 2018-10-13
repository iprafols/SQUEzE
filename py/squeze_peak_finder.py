"""
    SQUEzE
    ======
    
    This file provides a peak finder to be used by SQUEzE
    """

import numpy as np

class PeakFinder(object):
    """ Create and manage the peak finder used by SQUEzE
        
        CLASS: PeakFinder
        PURPOSE: Create and manage the peak finder used by SQUEzE. This
        peak finder looks for peaks in a smoothed spectrum by looking for
        points with values higher than their surroundings. It also computes
        the significance of the peaks and filters the results according to
        their significances.
        """
    
    def __init__(self, width, min_significance):
        """ Initialize class instance

            Parameters
            ----------
            width : float
            Width of the Gaussian used as a kernel for the convolution (see
            method 'smooth' from module 'squeze_spectrum' for details

            min_significance : float
            Minimum significance of the peak for it to be considered a valid peak
            """
        self.__width = width
        self.__fwhm = int(2.355*width)
        self.__half_fwhm = int(self.__fwhm/2)
        self.__min_significance = min_significance

    def __find_peak_significance(self, spectrum, index):
        """ Find the significance of the peak. The significance is computed by
            measuring the signal to noise of the difference between the peak and
            the continuum. The peak is measured using a window of size FWHM
            centered at the position of the specified index, where FWHM is the
            full width half maximum of the Gaussian used in the smoothing.
            The continuum is measured using two windows of size FWHM/2 at each
            side of the peak.
            
            Parameters
            ----------
            spectrum : Spectrum
            The spectrum where peaks are looked for
            
            
            """
        flux = spectrum.flux()
        ivar = spectrum.ivar()

        if index < self.__fwhm or index + self.__fwhm > flux.size:
            significance = np.nan
        else:
            peak = np.average(flux[index - self.__half_fwhm: index + self.__half_fwhm])
            cont = np.average(flux[index - self.__fwhm: index - self.__half_fwhm])
            cont += np.average(flux[index + self.__half_fwhm: index + self.__fwhm])
            cont /= 2.0
            ivar_diff = np.sum(ivar[index - self.__fwhm: index + self.__fwhm])
            if ivar_diff != 0.0:
                error = 1/np.sqrt(ivar_diff)
                significance = (peak-cont)/error
            else:
                significance = np.nan

        return significance

    def find_peaks(self, spectrum):
        """ Find significant peaks in a given spectrum.
            
            Parameters
            ----------
            spectrum : Spectrum
            The spectrum where peaks are looked for
            
            Returns
            -------
            An array with the position of the peaks
            """
        # smooth the spectrum
        smoothed_data = spectrum.smooth(self.__width)

        # find peaks
        peak_indexs = []
        for index, flux in enumerate(smoothed_data):
            if ((index > 0) and (index < smoothed_data.size -1) and
                (flux > smoothed_data[index + 1]) and (flux > smoothed_data[index - 1])):
                # find significance of the peak
                significance = self.__find_peak_significance(spectrum, index)

                # add the peak to the list if the significance is large enough
                if significance >= self.__min_significance:
                    peak_indexs.append(index)

        # convert list to array
        peak_indexs = np.array(peak_indexs, dtype=int)

        # return
        return peak_indexs


