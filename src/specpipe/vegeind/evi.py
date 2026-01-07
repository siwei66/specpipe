# -*- coding: utf-8 -*-
"""
SpecPipe - Vegetation Indices - EVI (Enhanced Vegetation Index)

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
def evi(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute EVI (Enhanced Vegetation Index) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        A Huete, K Didan, T Miura, E.P Rodriguez, X Gao, L.G Ferreira,
        Overview of the radiometric and biophysical performance of the MODIS vegetation indices,
        Remote Sensing of Environment,
        Volume 83, Issues 1–2,
        2002,
        Pages 195-213,
        ISSN 0034-4257,
        https://doi.org/10.1016/S0034-4257(02)00096-2.

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
    blue_range = (459, 479)
    red_range = (620, 670)
    ir_range = (841, 876)
    if (min(wavelength) > blue_range[0]) | (max(wavelength) < ir_range[1]):
        raise ValueError(
            f'Index requires wavelength range ({blue_range[0]}, {ir_range[1]}) '
            + f'exceeds given wavelength range ({min(wavelength)}, {max(wavelength)}).'
        )
    blue_bands_ind = (wavelength >= blue_range[0]) & (wavelength <= blue_range[1])
    red_bands_ind = (wavelength >= red_range[0]) & (wavelength <= red_range[1])
    ir_bands_ind = (wavelength >= ir_range[0]) & (wavelength <= ir_range[1])
    if (sum(red_bands_ind) < 1) | (sum(ir_bands_ind) < 1) | (sum(blue_bands_ind) < 1):
        raise ValueError('One or more required wavelength bands are missing from the dataset.')
    radiance_blue = np.nanmean(spec_array[:, blue_bands_ind], axis=1)
    radiance_red = np.nanmean(spec_array[:, red_bands_ind], axis=1)
    radiance_ir = np.nanmean(spec_array[:, ir_bands_ind], axis=1)
    if np.prod(radiance_blue != 0) * np.prod(radiance_red != 0) * np.prod(radiance_ir != 0) == 0:
        raise ValueError('Band values are all zero in the required wavelength range.')

    # Compute EVI
    evi_d = radiance_ir + 6 * radiance_red - 7.5 * radiance_blue + 1
    evi_d[evi_d == 0] = 1e-15
    evi = 2.5 * (radiance_ir - radiance_red) / evi_d

    # Convert to df
    df_vi = pd.DataFrame(evi.reshape(-1, 1), columns=['EVI'])

    if axis == 1:
        df_vi = df_vi.T

    return df_vi


# %% Local test

# from specpipe.vegeind.demo_data import create_vegeind_demo_data

# specdata = create_vegeind_demo_data()
# vidata = evi(specdata, wavelength=specdata.columns)
