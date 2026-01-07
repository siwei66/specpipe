# -*- coding: utf-8 -*-
"""
SpecPipe - Vegetation Indices - SWSIs (Salinity and Water Stress Indices)

Copyright (c) 2025 Siwei Luo. MIT License.
"""
import numpy as np
import pandas as pd
from typing import Annotated, Any

from ..specio import simple_type_validator, arraylike_validator

# # For local test
# from specpipe.specio import simple_type_validator, arraylike_validator


# %% Vegetation index function


@simple_type_validator
def swsi1(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute SWSI1 (Salinity and Water Stress Index 1) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        S. Hamzeh, A.A. Naseri, S.K. AlaviPanah, B. Mojaradi, H.M. Bartholomeus, J.G.P.W. Clevers, M. Behzad,
        Estimating salinity stress in sugarcane fields with spaceborne hyperspectral vegetation indices,
        International Journal of Applied Earth Observation and Geoinformation,
        Volume 21,
        2013,
        Pages 282-290,
        ISSN 1569-8432,
        https://doi.org/10.1016/j.jag.2012.07.002.

    Parameters
    ----------
    spec_array : 2D array-like, shape (n_samples, n_features)
        2D array-like of 1D spectral data series.
    wavelength : 1D array-like
        Wavelengths for the spectra data.
    axis : int, optional
        Axis of spectral data series index, must be 0 or 1. If 0, each row represents a sample spectrum.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing spectral indices of the samples.

    See Also
    --------
    vegeind_summary
    """
    # Validate input spectral data array
    spec_array = np.array(spec_array).astype('float32')
    if np.isnan(spec_array).any():
        raise ValueError('spec_array contains NaN values.')

    # Validate axis
    if axis == 1:
        spec_array = spec_array.T
    elif axis != 0:
        raise ValueError(f'axis must be 0 or 1, got: {axis}.')
    wavelength = np.array(wavelength).astype('float32')

    # Validate data and wavelengths
    if len(wavelength) != spec_array.shape[1]:
        raise ValueError(
            f'Length of given wavelength does not match the band number of spec_array, \
                got wavelength length: {len(wavelength)}, \
                got band number of given spec_array: {spec_array.shape[1]}'
        )
    band1 = 681
    band2 = 803
    band3 = 905
    band4 = 972
    if (min(wavelength) > band1) | (max(wavelength) < band4):
        raise ValueError(
            f'Index requires wavelength range ({band1}, {band4}) '
            + f'exceeds given wavelength range ({min(wavelength)}, {max(wavelength)}).'
        )

    # Get indices of the wavelength
    band1_ind = (np.abs(wavelength - band1)).argmin()
    band2_ind = (np.abs(wavelength - band2)).argmin()
    band3_ind = (np.abs(wavelength - band3)).argmin()
    band4_ind = (np.abs(wavelength - band4)).argmin()

    # Band values
    band1v = spec_array[:, band1_ind]
    band2v = spec_array[:, band2_ind]
    band3v = spec_array[:, band3_ind]
    band4v = spec_array[:, band4_ind]

    band1v[band1v == 0] = 1e-15
    band2v[band2v == 0] = 1e-15
    band3v[band3v == 0] = 1e-15
    band4v[band4v == 0] = 1e-15

    # Compute vegetation indices
    vi_values = (band2v - band1v) / (band3v + band4v) ** 0.5
    vi_values = np.array([vi_values])

    if axis == 0:
        df_vi = pd.DataFrame(vi_values.T, columns=['SWSI1'])
    else:
        df_vi = pd.DataFrame(vi_values, index=['SWSI1'])

    return df_vi


@simple_type_validator
def swsi2(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute SWSI2 (Salinity and Water Stress Index 2) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        S. Hamzeh, A.A. Naseri, S.K. AlaviPanah, B. Mojaradi, H.M. Bartholomeus, J.G.P.W. Clevers, M. Behzad,
        Estimating salinity stress in sugarcane fields with spaceborne hyperspectral vegetation indices,
        International Journal of Applied Earth Observation and Geoinformation,
        Volume 21,
        2013,
        Pages 282-290,
        ISSN 1569-8432,
        https://doi.org/10.1016/j.jag.2012.07.002.

    Parameters
    ----------
    spec_array : 2D array-like, shape (n_samples, n_features)
        2D array-like of 1D spectral data series.
    wavelength : 1D array-like
        Wavelengths for the spectra data.
    axis : int, optional
        Axis of spectral data series index, must be 0 or 1. If 0, each row represents a sample spectrum.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing spectral indices of the samples.

    See Also
    --------
    vegeind_summary
    """
    # Validate input spectral data array
    spec_array = np.array(spec_array).astype('float32')
    if np.isnan(spec_array).any():
        raise ValueError('spec_array contains NaN values.')

    # Validate axis
    if axis == 1:
        spec_array = spec_array.T
    elif axis != 0:
        raise ValueError(f'axis must be 0 or 1, got: {axis}.')
    wavelength = np.array(wavelength).astype('float32')

    # Validate data and wavelengths
    if len(wavelength) != spec_array.shape[1]:
        raise ValueError(
            f'Length of given wavelength does not match the band number of spec_array, \
                got wavelength length: {len(wavelength)}, \
                got band number of given spec_array: {spec_array.shape[1]}'
        )
    band1 = 681
    band2 = 803
    band3 = 1326
    band4 = 11507
    if (min(wavelength) > band1) | (max(wavelength) < band4):
        raise ValueError(
            f'Index requires wavelength range ({band1}, {band4}) '
            + f'exceeds given wavelength range ({min(wavelength)}, {max(wavelength)}).'
        )

    # Get indices of the wavelength
    band1_ind = (np.abs(wavelength - band1)).argmin()
    band2_ind = (np.abs(wavelength - band2)).argmin()
    band3_ind = (np.abs(wavelength - band3)).argmin()
    band4_ind = (np.abs(wavelength - band4)).argmin()

    # Band values
    band1v = spec_array[:, band1_ind]
    band2v = spec_array[:, band2_ind]
    band3v = spec_array[:, band3_ind]
    band4v = spec_array[:, band4_ind]

    band1v[band1v == 0] = 1e-15
    band2v[band2v == 0] = 1e-15
    band3v[band3v == 0] = 1e-15
    band4v[band4v == 0] = 1e-15

    # Compute vegetation indices
    vi_values = (band2v - band1v) / (band3v + band4v) ** 0.5
    vi_values = np.array([vi_values])

    if axis == 0:
        df_vi = pd.DataFrame(vi_values.T, columns=['SWSI2'])
    else:
        df_vi = pd.DataFrame(vi_values, index=['SWSI2'])

    return df_vi


@simple_type_validator
def swsi3(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute SWSI3 (Salinity and Water Stress Index 3) of 2D array-like of 1D spectra data series.

    The calculation is based on:

        S. Hamzeh, A.A. Naseri, S.K. AlaviPanah, B. Mojaradi, H.M. Bartholomeus, J.G.P.W. Clevers, M. Behzad,
        Estimating salinity stress in sugarcane fields with spaceborne hyperspectral vegetation indices,
        International Journal of Applied Earth Observation and Geoinformation,
        Volume 21,
        2013,
        Pages 282-290,
        ISSN 1569-8432,
        https://doi.org/10.1016/j.jag.2012.07.002.

    Parameters
    ----------
    spec_array : 2D array-like, shape (n_samples, n_features)
        2D array-like of 1D spectral data series.
    wavelength : 1D array-like
        Wavelengths for the spectra data.
    axis : int, optional
        Axis of spectral data series index, must be 0 or 1. If 0, each row represents a sample spectrum.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing spectral indices of the samples.

    See Also
    --------
    vegeind_summary
    """
    # Validate input spectral data array
    spec_array = np.array(spec_array).astype('float32')
    if np.isnan(spec_array).any():
        raise ValueError('spec_array contains NaN values.')

    # Validate axis
    if axis == 1:
        spec_array = spec_array.T
    elif axis != 0:
        raise ValueError(f'axis must be 0 or 1, got: {axis}.')
    wavelength = np.array(wavelength).astype('float32')

    # Validate data and wavelengths
    if len(wavelength) != spec_array.shape[1]:
        raise ValueError(
            f'Length of given wavelength does not match the band number of spec_array, \
                got wavelength length: {len(wavelength)}, \
                got band number of given spec_array: {spec_array.shape[1]}'
        )
    band1 = 681
    band2 = 803
    band3 = 972
    band4 = 1174
    if (min(wavelength) > band1) | (max(wavelength) < band4):
        raise ValueError(
            f'Index requires wavelength range ({band1}, {band4}) '
            + f'exceeds given wavelength range ({min(wavelength)}, {max(wavelength)}).'
        )

    # Get indices of the wavelength
    band1_ind = (np.abs(wavelength - band1)).argmin()
    band2_ind = (np.abs(wavelength - band2)).argmin()
    band3_ind = (np.abs(wavelength - band3)).argmin()
    band4_ind = (np.abs(wavelength - band4)).argmin()

    # Band values
    band1v = spec_array[:, band1_ind]
    band2v = spec_array[:, band2_ind]
    band3v = spec_array[:, band3_ind]
    band4v = spec_array[:, band4_ind]

    band1v[band1v == 0] = 1e-15
    band2v[band2v == 0] = 1e-15
    band3v[band3v == 0] = 1e-15
    band4v[band4v == 0] = 1e-15

    # Compute vegetation indices
    vi_values = (band2v - band1v) / (band3v + band4v) ** 0.5
    vi_values = np.array([vi_values])

    if axis == 0:
        df_vi = pd.DataFrame(vi_values.T, columns=['SWSI3'])
    else:
        df_vi = pd.DataFrame(vi_values, index=['SWSI3'])

    return df_vi


# %% Local test

# from specpipe.vegeind.demo_data import create_vegeind_demo_data

# specdata = create_vegeind_demo_data(nband=12000)
# vidata = swsi1(specdata, wavelength=specdata.columns)
# vidata = swsi2(specdata, wavelength=specdata.columns)
# vidata = swsi3(specdata, wavelength=specdata.columns)
