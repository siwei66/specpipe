# -*- coding: utf-8 -*-
"""
SpecPipe - Vegetation Indices - NDRE (Normalized Difference Red Edge Index)

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
def ndre(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute NDRE (Normalized Difference Red Edge Index) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Narmilan, A.; Gonzalez, F.; Salgadoe, A.S.A.; Kumarasiri, U.W.L.M.; Weerasinghe, H.A.S.; Kulasekara, B.R.
        Predicting Canopy Chlorophyll Content in Sugarcane Crops Using Machine Learning Algorithms and Spectral Vegetation Indices Derived from UAV Multispectral Imagery.
        Remote Sens. 2022, 14, 1140.
        https://doi.org/10.3390/rs14051140

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
    # blue_ind = (np.abs(wavelength - 450.0)).argmin()
    # green_ind = (np.abs(wavelength - 560.0)).argmin()
    # red_ind = (np.abs(wavelength - 650.0)).argmin()
    re_ind = (np.abs(wavelength - 730.0)).argmin()
    nir_ind = (np.abs(wavelength - 840.0)).argmin()

    # Band values
    # blue = spec_array[:, blue_ind]
    # green = spec_array[:, green_ind]
    # red = spec_array[:, red_ind]
    re = spec_array[:, re_ind]
    nir = spec_array[:, nir_ind]
    # blue[blue == 0] = 1e-15
    # green[green == 0] = 1e-15
    # red[red == 0] = 1e-15
    re[re == 0] = 1e-15
    nir[nir == 0] = 1e-15

    # Compute vegetation indices
    ndre_values = (nir - re) / (nir + re)
    # lci_values = (nir - re) / (nir + red)
    vi_arr = np.array(
        [
            ndre_values,
            # lci_values,
        ]
    )
    vi_names = [
        'NDRE',
        # 'LCI',
    ]
    if axis == 0:
        df_vi = pd.DataFrame(vi_arr.T, columns=vi_names).loc[:, ['NDRE']]
    else:
        df_vi = pd.DataFrame(vi_arr, index=vi_names).loc[['NDRE'], :]
    return df_vi


# %% Local test

# from specpipe.vegeind.demo_data import create_vegeind_demo_data

# specdata = create_vegeind_demo_data()
# vidata = ndre(specdata, wavelength=specdata.columns)
