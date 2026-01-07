# -*- coding: utf-8 -*-
"""
SpecPipe - Vegetation Indices - GDNVI (Green Normalized Difference Vegetation Index)

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
# GNDVI
def gndvi(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute GDNVI (Green Normalized Difference Vegetation Index) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Anatoly A. Gitelson, Yoram J. Kaufman, Mark N. Merzlyak,
        Use of a green channel in remote sensing of global vegetation from EOS-MODIS,
        Remote Sensing of Environment,
        Volume 58, Issue 3,
        1996,
        Pages 289-298,
        ISSN 0034-4257,
        https://doi.org/10.1016/S0034-4257(96)00072-7.

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
    # blue_range = (459, 479)
    green_range = (545, 565)
    # red_range = (620, 670)
    ir_range = (841, 876)
    if (min(wavelength) > green_range[0]) | (max(wavelength) < ir_range[1]):
        raise ValueError(
            f'Index requires wavelength range ({green_range[0]}, {ir_range[1]}) '
            + f'exceeds given wavelength range ({min(wavelength)}, {max(wavelength)}).'
        )
    green_bands_ind = (wavelength >= green_range[0]) & (wavelength <= green_range[1])
    ir_bands_ind = (wavelength >= ir_range[0]) & (wavelength <= ir_range[1])
    if (sum(green_bands_ind) < 1) | (sum(ir_bands_ind) < 1):
        raise ValueError('One or more required wavelength bands are missing from the dataset.')
    radiance_green = np.nanmean(spec_array[:, green_bands_ind], axis=1)
    radiance_ir = np.nanmean(spec_array[:, ir_bands_ind], axis=1)
    if np.prod(radiance_green != 0) * np.prod(radiance_ir != 0) == 0:
        raise ValueError('Band values are all zero in the required wavelength range.')

    # Compute GNDVI
    gndvi = (radiance_ir - radiance_green) / (radiance_ir + radiance_green)

    # Convert to df
    df_vi = pd.DataFrame(gndvi.reshape(-1, 1), columns=['GNDVI'])

    if axis == 1:
        df_vi = df_vi.T

    return df_vi


# %% Local test

# from specpipe.vegeind.demo_data import create_vegeind_demo_data

# specdata = create_vegeind_demo_data()
# vidata = gndvi(specdata, wavelength=specdata.columns)
