# -*- coding: utf-8 -*-
"""
SpecPipe - Vegetation Indices - SIPI (Structure Insensitive Pigment Index)

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
def sipi(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute SIPI (Structure Insensitive Pigment Index) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Peñuelas,J. and Baret,F. and Filella,I.(1995)
        Semi-empirical indices to assess carotenoids/chlorophyll a ratio from leaf spectral reflectance., (1995)
        Photosynthetica, 31, (2), 221–230

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
    if (min(wavelength) > 446) | (max(wavelength) < 799):
        raise ValueError(
            'Index requires wavelength range (445, 800) '
            + f'exceeds given wavelength range ({min(wavelength)}, {max(wavelength)}).'
        )

    # Spectra indices of the wavelength
    r445_ind = (np.abs(wavelength - 445)).argmin()
    r680_ind = (np.abs(wavelength - 680)).argmin()
    r800_ind = (np.abs(wavelength - 800)).argmin()
    r445 = spec_array[:, r445_ind]
    r680 = spec_array[:, r680_ind]
    r800 = spec_array[:, r800_ind]

    # Compute vegetation indices
    sipi_denominator = r800 - r680
    sipi_denominator[sipi_denominator == 0] = 1e-15
    sipi_values = (r800 - r445) / sipi_denominator
    sipi_values = np.array([sipi_values])

    if axis == 0:
        df_vi = pd.DataFrame(sipi_values.T, columns=['SIPI'])
    else:
        df_vi = pd.DataFrame(sipi_values, index=['SIPI'])

    return df_vi


# %% Local test

# from specpipe.vegeind.demo_data import create_vegeind_demo_data

# specdata = create_vegeind_demo_data()
# vidata = sipi(specdata, wavelength=specdata.columns)
