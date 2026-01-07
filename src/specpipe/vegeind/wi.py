# -*- coding: utf-8 -*-
"""
SpecPipe - Vegetation Indices - WI (Water Index)

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
def wi(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute WI (Water Index) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Penuelas, J., Pinol, J., Ogaya, R., & Filella, I. (1997).
        Estimation of plant water concentration by the reflectance Water Index WI (R900/R970).
        International Journal of Remote Sensing, 18(13), 2869–2875.
        https://doi.org/10.1080/014311697217396

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
    if (min(wavelength) > 901) | (max(wavelength) < 969):
        raise ValueError(
            'Index requires wavelength range (900, 970) '
            + f'exceeds given wavelength range ({min(wavelength)}, {max(wavelength)}).'
        )

    # Spectra indices of the wavelength
    r900_ind = (np.abs(wavelength - 900)).argmin()
    r970_ind = (np.abs(wavelength - 970)).argmin()
    r900 = spec_array[:, r900_ind]
    r970 = spec_array[:, r970_ind]
    r970[r970 == 0] = 1e-15

    # Compute vegetation indices
    wi_values = r900 / r970

    # Convert to df and output
    wi_values = np.array([wi_values])
    if axis == 0:
        df_vi = pd.DataFrame(wi_values.T, columns=['WI'])
    else:
        df_vi = pd.DataFrame(wi_values, index=['WI'])

    return df_vi


# %% Local test

# from specpipe.vegeind.demo_data import create_vegeind_demo_data

# specdata = create_vegeind_demo_data()
# vidata = sr(specdata, wavelength=specdata.columns)
