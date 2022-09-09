"""
    SQUEzE
    ======

    This file defines some functions necessary for the Candidates class to work.
    For the most part, these are jit functions to optimize the construction of
    the sample.
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"

import os
from math import sqrt
from numba import prange, jit, vectorize
import numpy as np
from astropy.table import Table


@jit(nopython=True)
def compute_line_ratios(wave, flux, ivar, peak_indexs, significances, try_lines,
                        lines):
    """Compute the line ratios for the specified lines for a given spectrum

    See equations 1 to 3 of Perez-Rafols et al. 2020 for a detailed
    description of the metrics

    Arguments
    ---------
    wave : array of float
    The spectrum wavelength

    flux : array of float
    The spectrum flux

    ivar : array of float
    The spectrum inverse variance

    peak indexs : array of int
    The indexes on the wave array where peaks are found

    significances : array of float
    The significances of the peaks

    try_lines : array of ints
    Indexes of the lines to be considered as originators of the peaks

    lines : array of arrays of floats
    Information of the lines where the ratios need to be computed. Each
    of the arrays must be organised as follows:
    0 - wavelength of the line
    1 - wavelength of the start of the peak interval
    2 - wavelength of the end of the peak interval
    3 - wavelength of the start of the blue interval
    4 - wavelength of the end of the blue interval
    5 - wavelength of the start of the red interval
    6 - wavelength of the end of the red interval

    Return
    ------
    new_candidates : list
    Each element of the list contains the ratios of the lines, the trial
    redshift, the significance of the line and the index of the line
    assumed to be originating the emission peak
    """
    new_candidates = []
    # pylint: disable=not-an-iterable
    # prange is the numba equivalent to range
    for index1 in prange(peak_indexs.size):
        for index2 in prange(len(try_lines)):
            # compute redshift
            # 0=WAVE
            z_try = wave[peak_indexs[index1]]
            z_try = z_try / lines[try_lines[index2], 0]
            z_try = z_try - 1.0
            if z_try < 0.0:
                continue
            oneplusz = (1.0 + z_try)

            candidate_info = []

            # compute peak ratio for the different lines
            for index3 in prange(lines.shape[0]):
                # compute intervals
                # 1=START, 2=END
                pix_peak = np.where((wave >= oneplusz * lines[index3, 1]) &
                                    (wave <= oneplusz * lines[index3, 2]))[0]
                # 3=BLUE_START, 4=BLUE_END
                pix_blue = np.where((wave >= oneplusz * lines[index3, 3]) &
                                    (wave <= oneplusz * lines[index3, 4]))[0]
                # 5=RED_START, 6=RED_END
                pix_red = np.where((wave >= oneplusz * lines[index3, 5]) &
                                   (wave <= oneplusz * lines[index3, 6]))[0]

                # compute peak and continuum values
                compute_ratio = True
                if ((pix_blue.size == 0) or (pix_peak.size == 0) or
                    (pix_red.size == 0) or
                    (pix_blue.size < pix_peak.size // 2) or
                    (pix_red.size < pix_peak.size // 2)):
                    compute_ratio = False
                else:
                    peak = np.mean(flux[pix_peak])
                    cont_red = np.mean(flux[pix_red])
                    cont_blue = np.mean(flux[pix_blue])
                    cont_red_and_blue = cont_red + cont_blue
                    if cont_red_and_blue == 0.0:
                        compute_ratio = False

                # compute ratios
                if compute_ratio:
                    peak_ivar_sum = ivar[pix_peak].sum()
                    if peak_ivar_sum == 0.0:
                        peak_err_squared = np.nan
                    else:
                        peak_err_squared = 1.0 / peak_ivar_sum
                    blue_ivar_sum = ivar[pix_blue].sum()
                    red_ivar_sum = ivar[pix_red].sum()
                    if blue_ivar_sum == 0.0 or red_ivar_sum == 0.0:
                        cont_err_squared = np.nan
                    else:
                        cont_err_squared = (1.0 / blue_ivar_sum +
                                            1.0 / red_ivar_sum) / 4.0

                    ratio = 2.0 * peak / cont_red_and_blue
                    ratio2 = abs((cont_red - cont_blue) / cont_red_and_blue)
                    err_ratio = sqrt(4. * peak_err_squared + ratio * ratio *
                                     cont_err_squared) / abs(cont_red_and_blue)
                    ratio_sn = (ratio - 1.0) / err_ratio
                else:
                    ratio = np.nan
                    ratio2 = np.nan
                    ratio_sn = np.nan

                candidate_info.append(ratio)
                candidate_info.append(ratio_sn)
                candidate_info.append(ratio2)

            candidate_info.append(z_try)
            candidate_info.append(significances[index1])
            candidate_info.append(float(index2))

            # add candidate to the list
            new_candidates.append(candidate_info)

    return new_candidates


@jit(nopython=True)
def compute_pixel_metrics(wave, flux, ivar, peak_indexs, num_pixels, try_lines,
                          lines):
    """Compute pixel metrics.

    Basically keep the pixel close to each peak as a new set of metrics.
    For each pixel, keep the flux and the ivar. Use NaN for no coverage.

    Arguments
    ---------
    wave : array of float
    The spectrum wavelength

    flux : array of float
    The spectrum flux

    ivar : array of float
    The spectrum inverse variance

    peak indexs : array of int
    The indexes on the wave array where peaks are found

    num_pixels : int
    The number of pixels to keep to each side of the peak

    try_lines : array of ints
    Indexes of the lines to be considered as originators of the peaks

    lines : array of arrays of floats
    Information of the lines where the ratios need to be computed. Each
    of the arrays must be organised as follows:
    0 - wavelength of the line
    1 - wavelength of the start of the peak interval
    2 - wavelength of the end of the peak interval
    3 - wavelength of the start of the blue interval
    4 - wavelength of the end of the blue interval
    5 - wavelength of the start of the red interval
    6 - wavelength of the end of the red interval

    Return
    ------
    pixel_metrics : list
    Each element of the list contains the pixel metrics associated to each
    peak.
    """
    pixel_metrics = []
    # pylint: disable=not-an-iterable
    # prange is the numba equivalent to range
    for index1 in prange(peak_indexs.size):
        #for peak_index, significance in zip(peak_indexs, significances):

        candidate_info = [float(x) for x in range(0)]

        # compute pixel metrics
        peak_index = peak_indexs[index1]
        candidate_info = []
        for index2 in prange(-num_pixels, 0):
            if peak_index + index2 < 0:
                candidate_info.append(np.nan)
                candidate_info.append(np.nan)
            else:
                candidate_info.append(flux[peak_index + index2])
                candidate_info.append(ivar[peak_index + index2])
        for index2 in prange(0, num_pixels):
            if peak_index + index2 >= flux.size:
                candidate_info.append(np.nan)
                candidate_info.append(np.nan)
            else:
                candidate_info.append(flux[peak_index + index2])
                candidate_info.append(ivar[peak_index + index2])

        # pylint: disable=not-an-iterable
        # prange is the numba equivalent to range
        for index2 in prange(len(try_lines)):
            #for try_line in try_lines:

            # compute redshift
            # 0=WAVE
            z_try = wave[peak_indexs[index1]]
            z_try = z_try / lines[try_lines[index2], 0]
            z_try = z_try - 1.0
            if z_try < 0.0:
                continue

            # add pixel metrics to the list once per each candidate
            pixel_metrics.append(candidate_info)

    return pixel_metrics


def convert_dtype(dtype):
    """Convert datatype "O" to "15" to save in fits file.
    Other types are ignored return as they are

    Arguments
    ---------
    dtype: dtype
    Data type

    Return
    ------
    dtype: dtype
    Data type
    """
    if dtype == "O":
        return "15A"
    return dtype


@vectorize
def compute_is_correct(correct_redshift, class_person):
    """ Returns True if a candidate is a true quasar and False otherwise.

    A true candidate is defined as a candidate having an absolute value
    of Delta_z is lower or equal than self.z_precision.

    Arguments
    ---------
    correct_redshift : array of bool
    Array specifying if the redhsift is correct

    class_person : array of int
    Array specifying the actual classification. 3 and 30 stand for quasars
    and BAL quasars respectively. 1 stands for stars and 4 stands for galaxies.

    Return
    ------
    correct : array of bool
    For each element in the arrays, returns True if the candidate is a
    quasar and has the correct redshift assign to it
    """
    correct = bool(correct_redshift and class_person in [3, 30])
    return correct


@vectorize
def compute_is_correct_redshift(delta_z, class_person, z_precision):
    """ Returns True if a candidate has a correct redshift and False otherwise.

    A candidate is assumed to have a correct redshift if it has an absolute
    value of Delta_z lower than or equal to z_precision.
    If the object is a star (class_person = 1), then return False.

    Arguments
    ---------
    delta_z : array of float
    Differences between the trial redshift (Z_TRY) and the true redshift
    (Z_TRUE)

    class_person : array of int
    Array specifying the actual classification. 3 and 30 stand for quasars
    and BAL quasars respectively. 1 stands for stars and 4 stands for galaxies.

    z_precision : float
    Tolerance with which two redshifts are considerd equal

    Return
    ------
    correct_redshift : array of bool
    For each element, returns True if the candidate is not a star and the
    trial redshift is equal to the true redshift.
    """
    correct_redshift = bool((class_person != 1) and
                            (-z_precision <= delta_z <= z_precision))
    return correct_redshift


@jit(nopython=True)
def compute_is_line(is_correct, class_person, assumed_line_index, z_true, z_try,
                    z_precision, lines):
    """ Return True if the candidates corresponds to a valid quasar emission
    line and False otherwise.

    A quasar line is defined as a candidate whose
    trial redshift is correct or where any of the redshifts obtained
    assuming that the emission line is generated by any of the lines is
    equal to the true redshift

    Arguments
    ---------
    is_correct : array of bool
    Array specifying if the candidates with a correct trial redshift

    class_person : array of int
    Array specifying the actual classification. 3 and 30 stand for quasars
    and BAL quasars respectively. 1 stands for stars and 4 stands for galaxies.

    assumed_line_index : int
    Index of the line considered as originating the peaks

    z_true : float
    True redshift

    z_try : float
    Trial redshift

    z_precision : float
    Tolerance with which two redshifts are considerd equal

    lines : array of arrays of floats
    Information of the lines that can be originating the emission line peaks.
    Each of the arrays must be organised as follows:
    0 - wavelength of the line

    Return
    ------
    is_line : array of bool
    For each element, returns True if thr candidate is a quasar line and
    False otherwise.
    """
    is_line = np.zeros_like(is_correct)

    # correct identification
    # pylint: disable=not-an-iterable
    # prange is the numba equivalent to range
    for index1 in prange(is_correct.size):
        if is_correct[index1]:
            is_line[index1] = True
        # not a quasar
        elif not ((class_person[index1] == 3) or (class_person[index1] == 30)):
            continue
        # not a peak
        elif assumed_line_index[index1] == -1:
            continue
        else:
            for index2 in prange(lines.shape[0]):
                #for line in self.lines.index:
                if index2 == assumed_line_index[index1]:
                    continue
                # 0=WAVE
                z_try_line = (lines[assumed_line_index[index1]][0] /
                              lines[index2][0]) * (1 + z_try[index1]) - 1
                if ((z_try_line - z_true[index1] <= z_precision) and
                    (z_try_line - z_true[index1] >= -z_precision)):
                    is_line[index1] = True
    return is_line


def load_df(filename):
    """Read a candidates dataframe from file

    Arguments
    ---------
    filename: str
    The file to read

    Return
    ------
    candidates: pd.DataFrame
    The loaded dataframe
    """
    data = Table.read(os.path.expandvars(filename), format='fits')
    candidates = data.to_pandas()
    candidates.columns = candidates.columns.str.upper()
    return candidates
