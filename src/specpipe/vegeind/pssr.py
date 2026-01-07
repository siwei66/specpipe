# -*- coding: utf-8 -*-
"""
SpecPipe - Vegetation Indices - PSSR (Pigment-specific Simple Ratio)

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
def pssr(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute PSSR (Pigment-specific Simple Ratio) R810/R676 and R810/R682 of 2D array-like of 1D spectra data series.
    The calculation is based on:

        George Alan Blackburn,
        Quantifying Chlorophylls and Caroteniods at Leaf and Canopy Scales: An Evaluation of Some Hyperspectral Approaches,
        Remote Sensing of Environment,
        Volume 66, Issue 3,
        1998,
        Pages 273-285,
        ISSN 0034-4257,
        https://doi.org/10.1016/S0034-4257(98)00059-5.

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
    """  # noqa: E501
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

    # Band indices
    # Band closest to the required wavelength
    band1_ind = (np.abs(wavelength - 810.4)).argmin()
    band2_ind = (np.abs(wavelength - 676.0)).argmin()
    band3_ind = (np.abs(wavelength - 681.9)).argmin()

    # Band values
    band1 = spec_array[:, band1_ind]
    band2 = spec_array[:, band2_ind]
    band3 = spec_array[:, band3_ind]
    band1[band1 == 0] = 1e-15
    band2[band2 == 0] = 1e-15
    band3[band3 == 0] = 1e-15

    # Compute vegetation indices
    pssr1_values = band1 / band2
    pssr2_values = band1 / band3
    vi_arr = np.array(
        [
            pssr1_values,
            pssr2_values,
        ]
    )
    vi_names = [
        'PSSR_R810_to_R676',
        'PSSR_R810_to_R682',
    ]
    if axis == 0:
        df_vi = pd.DataFrame(vi_arr.T, columns=vi_names)
    else:
        df_vi = pd.DataFrame(vi_arr, index=vi_names)
    return df_vi


@simple_type_validator
def pssr1(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute PSSR (Pigment-specific Simple Ratio) R810/R676 of 2D array-like of 1D spectra data series.
    The calculation is based on:

        George Alan Blackburn,
        Quantifying Chlorophylls and Caroteniods at Leaf and Canopy Scales: An Evaluation of Some Hyperspectral Approaches,
        Remote Sensing of Environment,
        Volume 66, Issue 3,
        1998,
        Pages 273-285,
        ISSN 0034-4257,
        https://doi.org/10.1016/S0034-4257(98)00059-5.

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
    """  # noqa: E501
    result = pssr(spec_array=spec_array, wavelength=wavelength, axis=axis)
    if axis == 0:
        return result.loc[:, ['PSSR_R810_to_R676']]
    else:
        return result.loc[['PSSR_R810_to_R676'], :]


@simple_type_validator
def pssr2(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute PSSR (Pigment-specific Simple Ratio) R810/R682 of 2D array-like of 1D spectra data series.
    The calculation is based on:

        George Alan Blackburn,
        Quantifying Chlorophylls and Caroteniods at Leaf and Canopy Scales: An Evaluation of Some Hyperspectral Approaches,
        Remote Sensing of Environment,
        Volume 66, Issue 3,
        1998,
        Pages 273-285,
        ISSN 0034-4257,
        https://doi.org/10.1016/S0034-4257(98)00059-5.

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
    """  # noqa: E501
    result = pssr(spec_array=spec_array, wavelength=wavelength, axis=axis)
    if axis == 0:
        return result.loc[:, ['PSSR_R810_to_R682']]
    else:
        return result.loc[['PSSR_R810_to_R682'], :]


# %% Local test

# from specpipe.vegeind.demo_data import create_vegeind_demo_data

# specdata = create_vegeind_demo_data()
# vidata = pssr(specdata, wavelength=specdata.columns)
