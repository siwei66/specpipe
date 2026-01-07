# -*- coding: utf-8 -*-
"""
SpecPipe - Vegetation Indices - SR (simple ratio, simple ratio of R800 / R675)

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
def sr(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute SR (simple ratio, simple ratio of R800 / R675) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Jordan, C. F. (1969).
        Derivation of Leaf-Area Index from Quality of Light on the Forest Floor.
        Ecology, 50(4), 663–666.
        https://doi.org/10.2307/1936256

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
    if (min(wavelength) > 675) | (max(wavelength) < 800):
        raise ValueError(
            'Index requires wavelength range (675, 800) '
            + f'exceeds given wavelength range ({min(wavelength)}, {max(wavelength)}).'
        )
    # Spectra indices of the wavelength
    r800_ind = (np.abs(wavelength - 800)).argmin()
    r675_ind = (np.abs(wavelength - 675)).argmin()
    r800 = spec_array[:, r800_ind]
    r675 = spec_array[:, r675_ind]
    r800[r800 == 0] = 1e-15
    r675[r675 == 0] = 1e-15
    # Compute vegetation indices
    sr_values = r800 / r675
    sr_values = np.array([sr_values])
    if axis == 0:
        df_vi = pd.DataFrame(sr_values.T, columns=['SR'])
    else:
        df_vi = pd.DataFrame(sr_values, index=['SR'])

    return df_vi


# %% Local test

# from specpipe.vegeind.demo_data import create_vegeind_demo_data

# specdata = create_vegeind_demo_data()
# vidata = sr(specdata, wavelength=specdata.columns)
