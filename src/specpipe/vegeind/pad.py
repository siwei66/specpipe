# -*- coding: utf-8 -*-
"""
SpecPipe - Vegetation Indices - PAD (Derivative Pseudo-Absorption)

Copyright (c) 2025 Siwei Luo. MIT License.
"""
import numpy as np
import pandas as pd
from typing import Annotated, Any, Union

from ..specio import simple_type_validator, arraylike_validator
from ..roistats import nderiv

# # For local test
# from specpipe.specio import simple_type_validator, arraylike_validator
# from specpipe.roistats import nderiv


# %% Vegetation index function


@simple_type_validator
def pad(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    order: int,
    padding: Union[int, float, str, None] = 'edge',
    axis: int = 0,
) -> np.ndarray:
    """
    Compute PAD (Derivative Pseudo-Absorption) of a 2D array-like collection of 1D spectral data series.
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
    order : int
        Order of the derivative.
    padding : int or float or str or None, optional
        Boundary padding strategy for derivative arrays. Choose between:

        - 'nan' – pad with NaN.
        - 'edge' – pad with edge values.
        - numeric – pad with the specified constant.
        - None – no padding.

        If None is used, or for n-th order derivatives, the output length will be reduced by 2n along the computation axis.
        The default is 'edge'.
    axis : int, optional
        Axis representing the spectral dimension. Must be 0 or 1.

        If 0, each row represents a sample spectrum.
        The default is 0.

    Returns
    -------
    np.ndarray
        Derivative pseudo-absorptions.

    See Also
    --------
    vegeind_summary
    """  # noqa: E501
    # Validate input spectral data array
    spec_array = np.array(spec_array).astype('float32')
    if np.isnan(spec_array).any():
        raise ValueError('spec_array contains NaN values.')

    # Validate order
    if order < 1:
        raise ValueError(f'order must be at least 1, but got: {order}')

    # Validate axis
    if axis == 1:
        spec_array = spec_array.T
    elif axis != 0:
        raise ValueError(f'axis must be 0 or 1, got: {axis}.')

    # Compute PAD
    spec_array[spec_array == 0] = 1e-15
    pseudo_absorbance = np.log10(1 / spec_array)
    pad_values = nderiv(pseudo_absorbance, n=order, axis=1, padding=padding)

    if axis == 1:
        pad_values = pad_values.T

    return np.array(pad_values)


@simple_type_validator
def padvi(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute PAD vegetation indices (Derivative Pseudo-Absorption Vegetation Indices) of 2D array-like of 1D spectra data series.
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

    # Compute PAD
    pad1_values = pad(spec_array, order=1, padding='edge', axis=0)
    pad2_values = pad(spec_array, order=2, padding='edge', axis=0)

    # Band indices
    # Band closest to the required wavelength
    r453_ind = (np.abs(wavelength - 452.6)).argmin()
    r717_ind = (np.abs(wavelength - 717.4)).argmin()
    r729_ind = (np.abs(wavelength - 729.3)).argmin()
    r732_ind = (np.abs(wavelength - 732.3)).argmin()
    r664_ind = (np.abs(wavelength - 664.3)).argmin()
    r726_ind = (np.abs(wavelength - 726.3)).argmin()

    # Band values
    pad1_453 = pad1_values[:, r453_ind]
    pad1_717 = pad1_values[:, r717_ind]
    pad1_729 = pad1_values[:, r729_ind]
    pad1_732 = pad1_values[:, r732_ind]
    pad2_664 = pad2_values[:, r664_ind]
    pad2_726 = pad2_values[:, r726_ind]

    vi_arr = np.array(
        [
            pad1_453,
            pad1_717,
            pad1_729,
            pad1_732,
            pad2_664,
            pad2_726,
        ]
    )
    vi_names = [
        'PAD1_R453',
        'PAD1_R717',
        'PAD1_R729',
        'PAD1_R732',
        'PAD2_R664',
        'PAD2_R726',
    ]
    if axis == 0:
        df_vi = pd.DataFrame(vi_arr.T, columns=vi_names)
    else:
        df_vi = pd.DataFrame(vi_arr, index=vi_names)
    return df_vi


@simple_type_validator
def pad1_r729(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute PAD1 R729 (First derivative pseudo-absorption R729) of 2D array-like of 1D spectra data series.
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
    result = padvi(spec_array=spec_array, wavelength=wavelength, axis=axis)
    if axis == 0:
        return result.loc[:, ['PAD1_R729']]
    else:
        return result.loc[['PAD1_R729'], :]


@simple_type_validator
def pad1_r732(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute PAD1 R732 (First derivative pseudo-absorption R732) of 2D array-like of 1D spectra data series.
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
    result = padvi(spec_array=spec_array, wavelength=wavelength, axis=axis)
    if axis == 0:
        return result.loc[:, ['PAD1_R732']]
    else:
        return result.loc[['PAD1_R732'], :]


@simple_type_validator
def pad2_r664(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute PAD2 R664 (Second derivative pseudo-absorption R664) of 2D array-like of 1D spectra data series.
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
    result = padvi(spec_array=spec_array, wavelength=wavelength, axis=axis)
    if axis == 0:
        return result.loc[:, ['PAD2_R664']]
    else:
        return result.loc[['PAD2_R664'], :]


# %% Local test

# from specpipe.vegeind.demo_data import create_vegeind_demo_data

# specdata = create_vegeind_demo_data()
# padv = pad(specdata, order=1)
# vidata = padvi(specdata, wavelength=specdata.columns)
