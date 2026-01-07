# -*- coding: utf-8 -*-
"""
SpecPipe - Vegetation Indices - PRI (Photochemical Reflectance Index)

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
def pri(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute PRI (Photochemical Reflectance Index) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        J.A. Gamon, J. Peñuelas, C.B. Field,
        A narrow-waveband spectral index that tracks diurnal changes in photosynthetic efficiency,
        Remote Sensing of Environment,
        Volume 41, Issue 1,
        1992,
        Pages 35-44,
        ISSN 0034-4257,
        https://doi.org/10.1016/0034-4257(92)90059-S.

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
    if (min(wavelength) > 532) | (max(wavelength) < 569):
        raise ValueError(
            'Index requires wavelength range (531, 570) '
            + f'exceeds given wavelength range ({min(wavelength)}, {max(wavelength)}).'
        )

    # Get band radiance
    r531_ind = (np.abs(wavelength - 531)).argmin()
    reference_ind = (np.abs(wavelength - 570)).argmin()
    r531 = spec_array[:, r531_ind]
    rref = spec_array[:, reference_ind]
    pri_denominator = rref + r531
    pri_denominator[pri_denominator == 0] = 1e-15

    # Compute index
    pri = (rref - r531) / pri_denominator

    # Convert to df
    df_vi = pd.DataFrame(pri.reshape(-1, 1), columns=['PRI'])

    if axis == 1:
        df_vi = df_vi.T

    return df_vi


# %% Local test

# from specpipe.vegeind.demo_data import create_vegeind_demo_data

# specdata = create_vegeind_demo_data()
# vidata = pri(specdata, wavelength=specdata.columns)
