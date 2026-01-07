# -*- coding: utf-8 -*-
"""
SpecPipe - Vegetation Indices - CVI (Chlorophyll Vegetation Index)

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
def cvi(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute CVI (Chlorophyll Vegetation Index) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Si Gao, Kai Yan, Jinxiu Liu, Jiabin Pu, Dongxiao Zou, Jianbo Qi, Xihan Mu, Guangjian Yan,
        Assessment of remote-sensed vegetation indices for estimating forest chlorophyll concentration,
        Ecological Indicators,
        Volume 162,
        2024,
        112001,
        ISSN 1470-160X,
        https://doi.org/10.1016/j.ecolind.2024.112001.

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

    # Sentinel-2 wavelength ranges
    # blue_range = (458, 523)
    green_range = (543, 578)
    red_range = (650, 680)
    # red_edge1 = (698, 713)
    # red_edge2 = (733, 748)
    # red_edge3 = (773, 793)
    nir_range = (785, 900)
    if (min(wavelength) > green_range[0]) | (max(wavelength) < nir_range[1]):  # Adjust according to applied VI
        raise ValueError(
            f'Index requires wavelength range ({green_range[0]}, {nir_range[1]}) '  # Adjust according to applied VI
            + f'exceeds given wavelength range ({min(wavelength)}, {max(wavelength)}).'
        )

    # Spectra indices within the ranges
    # blue_bands_ind = (wavelength >= blue_range[0]) & (wavelength <= blue_range[1])
    green_bands_ind = (wavelength >= green_range[0]) & (wavelength <= green_range[1])
    red_bands_ind = (wavelength >= red_range[0]) & (wavelength <= red_range[1])
    # re1_bands_ind = (wavelength >= red_edge1[0]) & (wavelength <= red_edge1[1])
    # re2_bands_ind = (wavelength >= red_edge2[0]) & (wavelength <= red_edge2[1])
    # re3_bands_ind = (wavelength >= red_edge3[0]) & (wavelength <= red_edge3[1])
    nir_bands_ind = (wavelength >= nir_range[0]) & (wavelength <= nir_range[1])
    if (
        # (sum(blue_bands_ind) < 1) |
        (sum(green_bands_ind) < 1)
        | (sum(red_bands_ind) < 1)
        |
        # (sum(re1_bands_ind) < 1) |
        # (sum(re2_bands_ind) < 1) |
        # (sum(re3_bands_ind) < 1) |
        (sum(nir_bands_ind) < 1)
        | False
    ):
        raise ValueError('One or more required wavelength bands are missing from the dataset.')

    # Mean radiance within the ranges
    # blue = np.nanmean(spec_array[:, blue_bands_ind], axis=1)
    green = np.nanmean(spec_array[:, green_bands_ind], axis=1)
    red = np.nanmean(spec_array[:, red_bands_ind], axis=1)
    # re1 = np.nanmean(spec_array[:, re1_bands_ind], axis=1)
    # re2 = np.nanmean(spec_array[:, re2_bands_ind], axis=1)
    # re3 = np.nanmean(spec_array[:, re3_bands_ind], axis=1)
    nir = np.nanmean(spec_array[:, nir_bands_ind], axis=1)
    if (
        # np.prod(blue != 0) *
        np.prod(green != 0) * np.prod(red != 0) *
        # np.prod(re1 != 0) *
        # np.prod(re2 != 0) *
        # np.prod(re3 != 0) *
        np.prod(nir != 0) * 1
        == 0
    ):
        raise ValueError('Band values are all zero in the required wavelength range.')

    # Vegetation indices
    # # Sentinel-2 Red Edge Position
    # s2rep_d = re2 - re1
    # s2rep_d[s2rep_d == 0] = 1e-10
    # s2rep_values = 705 + 35 * ((re3 + red) / 2 - re1) / s2rep_d

    # # MERIS Terrestrial Chlorophyll Index
    # mtci_d = re1 - red
    # mtci_d[mtci_d == 0] = 1e-10
    # mtci_values = (re2 - re1) / mtci_d

    # Chlorophyll Vegetation Index
    cvi_values = nir * red / green**2

    # # Triangular Greenness Index
    # tgi_values = - 0.5 * ((sum(red_range) / 2 - sum(blue_range) / 2) * (red - green) - (sum(red_range) / 2 - sum(green_range) / 2) * (red - green))  # noqa: E501

    # # Modified Chlorophyll Absorption Reflectance Index
    # mcari_values = (re1 - red) - 0.2 * (re1 - green) * re1 / red

    vi_arr = np.array(
        [
            # s2rep_values,
            # mtci_values,
            cvi_values,
            # tgi_values,
            # mcari_values,
        ]
    )
    vi_names = [
        # 'S2REP',
        # 'MTCI',
        'CVI',
        # 'TGI',
        # 'MCARI',
    ]
    if axis == 0:
        df_vi = pd.DataFrame(vi_arr.T, columns=vi_names).loc[:, ['CVI']]
    else:
        df_vi = pd.DataFrame(vi_arr, index=vi_names).loc[['CVI'], :]

    return df_vi


# %% Local test

# from specpipe.vegeind.demo_data import create_vegeind_demo_data

# specdata = create_vegeind_demo_data()
# vidata = cvi(specdata, wavelength=specdata.columns)
