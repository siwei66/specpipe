# -*- coding: utf-8 -*-
"""
SpecPipe - Vegetation Indices - SASIs (Soil Adjusted Salinity Indices)

Copyright (c) 2025 Siwei Luo. MIT License.
"""
import numpy as np
import pandas as pd
from typing import Annotated, Any, Union

from ..specio import simple_type_validator, arraylike_validator

# # For local test
# from specpipe.specio import simple_type_validator, arraylike_validator


# %% Vegetation index function


@simple_type_validator
def _sasi(  # noqa: C901
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
    L: Union[float, int] = 0.5,  # noqa: N803
    bands_long: tuple[Union[float, int], Union[float, int]] = (785, 900),
    bands_short: tuple[Union[float, int], Union[float, int]] = (650, 680),
    index_name: str = 'SAVI',
) -> pd.DataFrame:
    """
    Helper to compute arbitrary SASI (Soil Adjusted Salinity Index) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Zhang, Ting-Ting, et al.
        Using hyperspectral vegetation indices as a proxy to monitor soil salinity.
        Ecological Indicators 11.6 (2011): 1552-1562.
        DOI: 10.1016/j.ecolind.2011.03.025

    Parameters
    ----------
    spec_array : 2D array-like, shape (n_samples, n_features)
        2D array-like of 1D spectral data series.
    wavelength : 1D array-like
        Wavelengths for the spectra data.
    axis : int, optional
        Axis of spectral data series index, must be 0 or 1. If 0, each row represents a sample spectrum.
    L : float or int
        Soil influence adjustment parameter. The default is 0.5.
    bands_long : tuple of (float or int, float or int)
        Range of longer band item. The default is SAVI's NIR range.
    bands_short : tuple of (float or int, float or int)
        Range of shorter band item. The default is SAVI's red range.

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

    # Validate band range
    if bands_long[0] >= bands_long[1]:
        raise ValueError(f"Invalid range of 'bands_long': {bands_long}")
    if bands_short[0] >= bands_short[1]:
        raise ValueError(f"Invalid range of 'bands_short': {bands_short}")
    bands_1 = bands_long
    bands_2 = bands_short

    # Validate wavelength and band range
    if (min(wavelength) > bands_1[0]) | (max(wavelength) < bands_2[1]):
        raise ValueError(
            f'Index requires wavelength range ({bands_1[0]}, {bands_2[1]}) '
            + f'exceeds given wavelength range ({min(wavelength)}, {max(wavelength)}).'
        )
    bands_1_ind = (wavelength >= bands_1[0]) & (wavelength <= bands_1[1])
    bands_2_ind = (wavelength >= bands_2[0]) & (wavelength <= bands_2[1])
    if (sum(bands_1_ind) < 1) | (sum(bands_2_ind) < 1):
        raise ValueError('One or more required wavelength bands are missing from the dataset.')

    # Related band radiance
    radiance_bands_1 = np.nanmean(spec_array[:, bands_1_ind], axis=1)
    radiance_bands_2 = np.nanmean(spec_array[:, bands_2_ind], axis=1)
    if np.prod(radiance_bands_1 != 0) * np.prod(radiance_bands_2 != 0) == 0:
        raise ValueError('Band values are all zero in the required wavelength range.')

    # Compute vegetation indices
    sasi_values = ((radiance_bands_1 - radiance_bands_2) / (radiance_bands_1 + radiance_bands_2 + L)) * (1 + L)

    # Convert to df
    df_vi = pd.DataFrame(sasi_values.reshape(-1, 1), columns=[index_name])

    if axis == 1:
        df_vi = df_vi.T

    return df_vi


@simple_type_validator
def sasi1(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
    L: Union[float, int] = 0.5,  # noqa: N803
) -> pd.DataFrame:
    """
    Compute SASI1 (Soil Adjusted Salinity Index 1) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Zhang, Ting-Ting, et al.
        Using hyperspectral vegetation indices as a proxy to monitor soil salinity.
        Ecological Indicators 11.6 (2011): 1552-1562.
        DOI: 10.1016/j.ecolind.2011.03.025

    Parameters
    ----------
    spec_array : 2D array-like, shape (n_samples, n_features)
        2D array-like of 1D spectral data series.
    wavelength : 1D array-like
        Wavelengths for the spectra data.
    axis : int, optional
        Axis of spectral data series index, must be 0 or 1. If 0, each row represents a sample spectrum.
    L : float or int
        Soil influence adjustment parameter. The default is 0.5.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing spectral indices of the samples.

    See Also
    --------
    vegeind_summary
    """
    index_name = "SASI1_L" + str(L).replace('.', 'd')
    result = _sasi(
        spec_array=spec_array,
        wavelength=wavelength,
        axis=axis,
        L=L,
        bands_long=(560, 590),
        bands_short=(546, 575),
        index_name=index_name,
    )
    return result


@simple_type_validator
def sasi2(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
    L: Union[float, int] = 0.5,  # noqa: N803
) -> pd.DataFrame:
    """
    Compute SASI2 (Soil Adjusted Salinity Index 2) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Zhang, Ting-Ting, et al.
        Using hyperspectral vegetation indices as a proxy to monitor soil salinity.
        Ecological Indicators 11.6 (2011): 1552-1562.
        DOI: 10.1016/j.ecolind.2011.03.025

    Parameters
    ----------
    spec_array : 2D array-like, shape (n_samples, n_features)
        2D array-like of 1D spectral data series.
    wavelength : 1D array-like
        Wavelengths for the spectra data.
    axis : int, optional
        Axis of spectral data series index, must be 0 or 1. If 0, each row represents a sample spectrum.
    L : float or int
        Soil influence adjustment parameter. The default is 0.5.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing spectral indices of the samples.

    See Also
    --------
    vegeind_summary
    """
    index_name = "SASI2_L" + str(L).replace('.', 'd')
    result = _sasi(
        spec_array=spec_array,
        wavelength=wavelength,
        axis=axis,
        L=L,
        bands_long=(690, 780),
        bands_short=(655, 764),
        index_name=index_name,
    )
    return result


@simple_type_validator
def sasi3(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
    L: Union[float, int] = 0.5,  # noqa: N803
) -> pd.DataFrame:
    """
    Compute SASI3 (Soil Adjusted Salinity Index 3) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Zhang, Ting-Ting, et al.
        Using hyperspectral vegetation indices as a proxy to monitor soil salinity.
        Ecological Indicators 11.6 (2011): 1552-1562.
        DOI: 10.1016/j.ecolind.2011.03.025

    Parameters
    ----------
    spec_array : 2D array-like, shape (n_samples, n_features)
        2D array-like of 1D spectral data series.
    wavelength : 1D array-like
        Wavelengths for the spectra data.
    axis : int, optional
        Axis of spectral data series index, must be 0 or 1. If 0, each row represents a sample spectrum.
    L : float or int
        Soil influence adjustment parameter. The default is 0.5.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing spectral indices of the samples.

    See Also
    --------
    vegeind_summary
    """
    index_name = "SASI3_L" + str(L).replace('.', 'd')
    result = _sasi(
        spec_array=spec_array,
        wavelength=wavelength,
        axis=axis,
        L=L,
        bands_long=(800, 868),
        bands_short=(655, 764),
        index_name=index_name,
    )
    return result


@simple_type_validator
def sasi4(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
    L: Union[float, int] = 0.5,  # noqa: N803
) -> pd.DataFrame:
    """
    Compute SASI4 (Soil Adjusted Salinity Index 4) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Zhang, Ting-Ting, et al.
        Using hyperspectral vegetation indices as a proxy to monitor soil salinity.
        Ecological Indicators 11.6 (2011): 1552-1562.
        DOI: 10.1016/j.ecolind.2011.03.025

    Parameters
    ----------
    spec_array : 2D array-like, shape (n_samples, n_features)
        2D array-like of 1D spectral data series.
    wavelength : 1D array-like
        Wavelengths for the spectra data.
    axis : int, optional
        Axis of spectral data series index, must be 0 or 1. If 0, each row represents a sample spectrum.
    L : float or int
        Soil influence adjustment parameter. The default is 0.5.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing spectral indices of the samples.

    See Also
    --------
    vegeind_summary
    """
    index_name = "SASI4_L" + str(L).replace('.', 'd')
    result = _sasi(
        spec_array=spec_array,
        wavelength=wavelength,
        axis=axis,
        L=L,
        bands_long=(889, 903),
        bands_short=(655, 764),
        index_name=index_name,
    )
    return result


# %% Local test

# from specpipe.vegeind.demo_data import create_vegeind_demo_data

# specdata = create_vegeind_demo_data()
# vidata = sasi1(specdata, wavelength=specdata.columns)
