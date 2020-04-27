"""
    SQUEzE - WEAVE
    ==============

    This file implements the class MySpectrum, that is a "fill the blanks"
    example of a class inheriting from Spectrum defined in squeze_spectrum.py.
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import pandas as pd

from squeze.squeze_error import Error
from squeze.squeze_spectrum import Spectrum

class WeaveSpectrum(Spectrum):
    """
        Load and format a WEAVE spectrum to be digested by SQUEzE

        CLASS: WeaveSpectrum
        PURPOSE: Load and format a WEAVE spectrum to be digested by
        SQUEzE
        """
    def __init__(self, spectrum_dict, wave, metadata):
        """ Initialize class instance

            Parameters
            ----------
            spectrum_dict : dict
            A dictionary with the flux, inverse variance, and wavelengths for the red CCD.
            Keys are "red_flux" and "red_ivar", and  with the flux, inverse variance, and
            wavelengths for the blue CCD. Keys are "blue_flux" and "blue_ivar".

            wave : dict
            Dictionary with the wavelength information. Must contain the arrays "red_wave"
            and "blue_wave", and the floats "red_delta_wave" and "blue_deltas_wave".

            metadata : dict
            A dictionary with the metadata. Keys should be strings
            """
        # check that "specid" is present in metadata
        if "specid" not in metadata.keys():
            raise Error("""The property "specid" must be present in metadata""")

        self._flux, self._ivar, self._wave = get_spectra(spectrum_dict, wave)
        self._metadata = metadata


def get_spectra(spectrum_dict, wave):
    """ Given the fluxes, inverse variances and wavelengths for the red and blue CCDs,
        return create a combined spectrum.
        Fill the flux in the inter-CCD gaps using a linear interpolation of the neighbouring
        pixels. This will not affect the computation of the ratios: they are weigthed using
        the inverse variance, which is zero in this section; and will stabilize the peak
        finding algorithm.

        Parameters
        ----------
        spectrum_dict : dict
        A dictionary with the flux, inverse variance, and wavelengths for the red CCD.
        Keys are "red_flux" and "red_ivar", and  with the flux, inverse variance, and
        wavelengths for the blue CCD. Keys are "blue_flux" and "blue_ivar".

        wave : dict
        Dictionary with the wavelength information. Must contain the arrays "red_wave"
        and "blue_wave", and the floats "red_delta_wave" and "blue_deltas_wave".

        Returns
        -------
        A tupple with the flux, inverse_variance, and wavelength
        """
    # nested function to drop leading and trailing zeros
    def drop_leading_trailing_zeros(data_frame):
        """ Drop leading and trailing zeros

            Parameters
            ----------
            data_frame : pd.DataFrame
            A dataframe with the flux and the inverse variance. Index of the DataFrame
            is the wavelength.

            Returns
            -------
            The modified DataFrame
            """
        non_zeros = data_frame[(data_frame["flux"] != 0.0) &
                               (data_frame["ivar"] != 0.0)]
        data_frame = data_frame[(data_frame.index >= non_zeros.iloc[1].name) &
                                (data_frame.index <= non_zeros.iloc[-2].name)]
        return data_frame

    # nested function to fill the gap
    def fill_gap(data_frame, gap):
        """ Fill the gap by linearly interpolating between the edges

            Parameters
            ----------
            data_frame : pd.DataFrame
            A dataframe with the flux and the inverse variance. Index of the DataFrame
            is the wavelength.

            gap : tuple
            A tuple with the initial and ending wavelengths for the gap

            Returns
            -------
            The modified DataFrame
            """
        gap_slope = (data_frame[data_frame.index == gap[0]]["flux"].values - \
            data_frame[data_frame.index == gap[1]]["flux"].values)/(gap[0]-gap[1])
        gap_startflux = data_frame[data_frame.index == gap[0]]["flux"]
        def interpolate_gap(row):
            """ Interpolate data within the gap by using linear interpolation
                between the edges """
            if row.name > gap[0] and row.name < gap[1]:
                row["flux"] = gap_slope*(row.name - gap[0]) + gap_startflux
            return row
        data_frame = data_frame.apply(interpolate_gap, axis=1)
        return data_frame

    # nested function fo find the gap
    def find_gap(data_frame, delta_wave):
        """ Find the gap in the spectrum

            Parameters
            ----------
            data_frame : pd.DataFrame
            A dataframe with the flux and the inverse variance. Index of the DataFrame
            is the wavelength.

            delta_wave : float
            Wavelength step for the wavelength array in the DataFrame

            Returns
            -------
            A tuple with the start and end of the gap
            """
        zeros = data_frame[(data_frame["flux"] == 0.0) & (data_frame["ivar"] == 0.0)]
        gap_start = zeros.iloc[0].name - 4.0*delta_wave
        gap_end = zeros.iloc[-1].name + 4.0*delta_wave
        return (gap_start, gap_end)

    # format red spectrum
    red_dict = {key[4:]: value for key, value in spectrum_dict.items() if "red" in key}
    red_df = pd.DataFrame(red_dict, index=wave.get("red_wave"))
    # locate leading and ending zeros
    red_df = drop_leading_trailing_zeros(red_df)
    # locate the gap
    gap = find_gap(red_df, wave.get("red_delta_wave"))
    # interpolate to fill the gap
    red_df = fill_gap(red_df, gap)

    # format blue spectrum
    blue_dict = {key[5:]: value for key, value in spectrum_dict.items() if "blue" in key}
    blue_df = pd.DataFrame(blue_dict, index=wave.get("blue_wave"))
    # locate leading and ending zeros
    blue_df = drop_leading_trailing_zeros(blue_df)
    # locate the gap
    gap = find_gap(blue_df, wave.get("blue_delta_wave"))
    # interpolate to fill the gap
    blue_df = fill_gap(blue_df, gap)

    # combine the two spectra
    comb = pd.concat((red_df, blue_df))
    comb = comb.groupby(comb.index).mean()

    return comb["flux"].values, comb["ivar"].values, comb.index.values


if __name__ == "__main__":
    pass
