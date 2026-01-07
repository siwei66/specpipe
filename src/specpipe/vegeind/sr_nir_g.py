# -*- coding: utf-8 -*-
"""
SpecPipe - Vegetation Indices - SR NIR/Green (Simple Ratio NIR/Green)

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
def sr_nir_g(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute SR NIR/Green (Simple Ratio NIR/Green) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        C.S.T Daughtry, C.L Walthall, M.S Kim, E.Brown de Colstoun, J.E McMurtrey,
        Estimating Corn Leaf Chlorophyll Concentration from Leaf and Canopy Reflectance,
        Remote Sensing of Environment,
        Volume 74, Issue 2,
        2000,
        Pages 229-239,
        ISSN 0034-4257,
        https://doi.org/10.1016/S0034-4257(00)00113-9.

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

    # Band indices
    # Band closest to the required wavelength
    green_ind = (np.abs(wavelength - 550.0)).argmin()
    # red_ind = (np.abs(wavelength - 670.0)).argmin()
    nir_ind = (np.abs(wavelength - 801.0)).argmin()

    # Band values
    green = spec_array[:, green_ind]
    # red = spec_array[:, red_ind]
    nir = spec_array[:, nir_ind]
    green[green == 0] = 1e-15
    # red[red == 0] = 1e-15
    nir[nir == 0] = 1e-15

    # Compute vegetation indices
    # sr_nir_red = nir / red
    sr_nir_green = nir / green
    vi_arr = np.array(
        [
            # sr_nir_red,
            sr_nir_green,
        ]
    )
    vi_names = [
        # 'SR_NIR_to_R',
        'SR_NIR_to_G',
    ]
    if axis == 0:
        df_vi = pd.DataFrame(vi_arr.T, columns=vi_names).loc[:, ['SR_NIR_to_G']]
    else:
        df_vi = pd.DataFrame(vi_arr, index=vi_names).loc[['SR_NIR_to_G'], :]
    return df_vi


# %% Local test

# from specpipe.vegeind.demo_data import create_vegeind_demo_data

# specdata = create_vegeind_demo_data()
# vidata = sr_nir_g(specdata, wavelength=specdata.columns)
