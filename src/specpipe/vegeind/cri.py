# -*- coding: utf-8 -*-
"""
SpecPipe - Vegetation Indices - CRI (Carotenoid Reflectance Index)

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
def cri(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute CRI (Carotenoid Reflectance Index) and RNIR/R510 of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Gitelson, A.A.; Zur, Y.; Chivkunova, O.B.; Merzlyak, M.N.
        Assessing Carotenoid Content in Plant Leaves with Reflectance Spectroscopy.
        Photochem. Photobiol. 2007, 75, 272–281,
        https://doi.org/10.1562/0031-8655(2002)0750272accipl2.0.co2.

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
    nir_range = (750, 800)
    if (min(wavelength) > 510) | (max(wavelength) < 800):
        raise ValueError(
            'Index requires wavelength range (510, 800) '
            + f'exceeds given wavelength range ({min(wavelength)}, {max(wavelength)}).'
        )
    nir_bands_ind = (wavelength >= nir_range[0]) & (wavelength <= nir_range[1])
    if sum(nir_bands_ind) < 1:
        raise ValueError('One or more required wavelength bands are missing from the dataset.')

    rnir = np.nanmean(spec_array[:, nir_bands_ind], axis=1)
    if np.prod(rnir != 0) == 0:
        raise ValueError('Band values are all zero in the required wavelength range.')
    rnir[rnir == 0] = 1e-15

    # Spectra indices of the wavelength
    r510_ind = (np.abs(wavelength - 510)).argmin()
    r550_ind = (np.abs(wavelength - 550)).argmin()
    r700_ind = (np.abs(wavelength - 700)).argmin()
    r510 = spec_array[:, r510_ind]
    r550 = spec_array[:, r550_ind]
    r700 = spec_array[:, r700_ind]
    r510[r510 == 0] = 1e-15
    r550[r550 == 0] = 1e-15
    r700[r700 == 0] = 1e-15

    # Compute vegetation indices
    cri550 = (1 / r510 + 1 / rnir) - 0.75 * (1 / r550 - 1 / rnir)
    cri700 = (1 / r510 + 1 / rnir) - 0.75 * (1 / r700 - 1 / rnir)
    rnir_to_r510 = rnir / r510

    vi_arr = np.array(
        [
            cri550,
            cri700,
            rnir_to_r510,
        ]
    )
    vi_names = ['CRI550', 'CRI700', 'RNIR_to_R510']

    if axis == 0:
        df_vi = pd.DataFrame(vi_arr.T, columns=vi_names)
    else:
        df_vi = pd.DataFrame(vi_arr, index=vi_names)

    return df_vi


@simple_type_validator
def rnir_to_r510(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute RNIR / R510 (Simple Ratio RNIR / R510) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Gitelson, A.A.; Zur, Y.; Chivkunova, O.B.; Merzlyak, M.N.
        Assessing Carotenoid Content in Plant Leaves with Reflectance Spectroscopy.
        Photochem. Photobiol. 2007, 75, 272–281,
        https://doi.org/10.1562/0031-8655(2002)0750272accipl2.0.co2.

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
    if axis == 0:
        result = cri(spec_array, wavelength=wavelength, axis=axis).loc[:, ['RNIR_to_R510']]
    else:
        result = cri(spec_array, wavelength=wavelength, axis=axis).loc[['RNIR_to_R510'], :]
    return result


@simple_type_validator
def cri550(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute CRI550 (Carotenoid Reflectance Index 550) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Gitelson, A.A.; Zur, Y.; Chivkunova, O.B.; Merzlyak, M.N.
        Assessing Carotenoid Content in Plant Leaves with Reflectance Spectroscopy.
        Photochem. Photobiol. 2007, 75, 272–281,
        https://doi.org/10.1562/0031-8655(2002)0750272accipl2.0.co2.

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
    if axis == 0:
        result = cri(spec_array, wavelength=wavelength, axis=axis).loc[:, ['CRI550']]
    else:
        result = cri(spec_array, wavelength=wavelength, axis=axis).loc[['CRI550'], :]
    return result


@simple_type_validator
def cri700(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute CRI700 (Carotenoid Reflectance Index 700) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Gitelson, A.A.; Zur, Y.; Chivkunova, O.B.; Merzlyak, M.N.
        Assessing Carotenoid Content in Plant Leaves with Reflectance Spectroscopy.
        Photochem. Photobiol. 2007, 75, 272–281,
        https://doi.org/10.1562/0031-8655(2002)0750272accipl2.0.co2.

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
    if axis == 0:
        result = cri(spec_array, wavelength=wavelength, axis=axis).loc[:, ['CRI700']]
    else:
        result = cri(spec_array, wavelength=wavelength, axis=axis).loc[['CRI700'], :]
    return result


# %% Local test

# from specpipe.vegeind.demo_data import create_vegeind_demo_data

# specdata = create_vegeind_demo_data()
# vidata = cri(specdata, wavelength=specdata.columns)
