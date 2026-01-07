# -*- coding: utf-8 -*-
"""
SpecPipe - Vegetation Indices - SIs (Salinity Indices)

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
def si(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute SIs (Salinity Indices) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Nguyen, KA., Liou, YA., Tran, HP. et al.
        Soil salinity assessment by using near-infrared channel and Vegetation Soil Salinity Index derived from Landsat 8 OLI data: a case study in the Tra Vinh Province, Mekong Delta, Vietnam.
        Prog Earth Planet Sci 7, 1 (2020).
        https://doi.org/10.1186/s40645-019-0311-0

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
    # Landset-8 wavelength ranges
    # violetblue_range = (433, 453)
    blue_range = (450, 515)
    green_range = (525, 600)
    red_range = (630, 680)
    nir_range = (845, 885)
    if (min(wavelength) > blue_range[0]) | (max(wavelength) < nir_range[1]):
        raise ValueError(
            f'Index requires wavelength range ({blue_range[0]}, {nir_range[1]}) '
            + f'exceeds given wavelength range ({min(wavelength)}, {max(wavelength)}).'
        )

    # Get band ind of the range
    blue_bands_ind = (wavelength >= blue_range[0]) & (wavelength <= blue_range[1])
    green_bands_ind = (wavelength >= green_range[0]) & (wavelength <= green_range[1])
    red_bands_ind = (wavelength >= red_range[0]) & (wavelength <= red_range[1])
    nir_bands_ind = (wavelength >= nir_range[0]) & (wavelength <= nir_range[1])
    if (
        (sum(blue_bands_ind) < 1)
        | (sum(green_bands_ind) < 1)
        | (sum(red_bands_ind) < 1)
        | (sum(nir_bands_ind) < 1)
        | False
    ):
        raise ValueError('One or more required wavelength bands are missing from the dataset.')

    # Related band radiance
    blue = np.nanmean(spec_array[:, blue_bands_ind], axis=1)
    green = np.nanmean(spec_array[:, green_bands_ind], axis=1)
    red = np.nanmean(spec_array[:, red_bands_ind], axis=1)
    nir = np.nanmean(spec_array[:, nir_bands_ind], axis=1)
    if (np.prod(blue != 0) * np.prod(green != 0) * np.prod(red != 0) * np.prod(nir != 0) * 1) == 0:
        raise ValueError('Band values are all zero in at least one of the required wavelength ranges.')

    # Compute vegetation indices
    # Salinity index
    si1_values = (green**2 + red**2) ** 0.5
    si2_values = (green * red) ** 0.5
    si3_values = (blue * red) ** 0.5
    si4_values = red * nir / green
    si5_values = blue / red
    # Normalized salinity index
    ndsi_values = red - nir / (red + nir)
    # Vegetation soil salinity index
    vssi_values = 2 * green - 5 * (red + nir)

    vi_arr = np.array(
        [
            si1_values,
            si2_values,
            si3_values,
            si4_values,
            si5_values,
            ndsi_values,
            vssi_values,
        ]
    )

    vi_names = [
        "SI1",
        "SI2",
        "SI3",
        "SI4",
        "SI5",
        "NDSI",
        "VSSI",
    ]

    # Convert to df
    if axis == 0:
        df_vi = pd.DataFrame(vi_arr.T, columns=vi_names)
    else:
        df_vi = pd.DataFrame(vi_arr, index=vi_names)

    return df_vi


@simple_type_validator
def si1(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute SI1 (Salinity Index 1) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Nguyen, KA., Liou, YA., Tran, HP. et al.
        Soil salinity assessment by using near-infrared channel and Vegetation Soil Salinity Index derived from Landsat 8 OLI data: a case study in the Tra Vinh Province, Mekong Delta, Vietnam.
        Prog Earth Planet Sci 7, 1 (2020).
        https://doi.org/10.1186/s40645-019-0311-0

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
    result = si(spec_array=spec_array, wavelength=wavelength, axis=axis)
    if axis == 0:
        result = result.loc[:, ['SI1']]
    else:
        result = result.loc[['SI1'], :]
    return result


@simple_type_validator
def si2(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute SI2 (Salinity Index 2) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Nguyen, KA., Liou, YA., Tran, HP. et al.
        Soil salinity assessment by using near-infrared channel and Vegetation Soil Salinity Index derived from Landsat 8 OLI data: a case study in the Tra Vinh Province, Mekong Delta, Vietnam.
        Prog Earth Planet Sci 7, 1 (2020).
        https://doi.org/10.1186/s40645-019-0311-0

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
    result = si(spec_array=spec_array, wavelength=wavelength, axis=axis)
    if axis == 0:
        result = result.loc[:, ['SI2']]
    else:
        result = result.loc[['SI2'], :]
    return result


@simple_type_validator
def si3(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute SI3 (Salinity Index 3) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Nguyen, KA., Liou, YA., Tran, HP. et al.
        Soil salinity assessment by using near-infrared channel and Vegetation Soil Salinity Index derived from Landsat 8 OLI data: a case study in the Tra Vinh Province, Mekong Delta, Vietnam.
        Prog Earth Planet Sci 7, 1 (2020).
        https://doi.org/10.1186/s40645-019-0311-0

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
    result = si(spec_array=spec_array, wavelength=wavelength, axis=axis)
    if axis == 0:
        result = result.loc[:, ['SI3']]
    else:
        result = result.loc[['SI3'], :]
    return result


@simple_type_validator
def si4(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute SI4 (Salinity Index 4) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Nguyen, KA., Liou, YA., Tran, HP. et al.
        Soil salinity assessment by using near-infrared channel and Vegetation Soil Salinity Index derived from Landsat 8 OLI data: a case study in the Tra Vinh Province, Mekong Delta, Vietnam.
        Prog Earth Planet Sci 7, 1 (2020).
        https://doi.org/10.1186/s40645-019-0311-0

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
    result = si(spec_array=spec_array, wavelength=wavelength, axis=axis)
    if axis == 0:
        result = result.loc[:, ['SI4']]
    else:
        result = result.loc[['SI4'], :]
    return result


@simple_type_validator
def si5(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute SI5 (Salinity Index 5) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Nguyen, KA., Liou, YA., Tran, HP. et al.
        Soil salinity assessment by using near-infrared channel and Vegetation Soil Salinity Index derived from Landsat 8 OLI data: a case study in the Tra Vinh Province, Mekong Delta, Vietnam.
        Prog Earth Planet Sci 7, 1 (2020).
        https://doi.org/10.1186/s40645-019-0311-0

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
    result = si(spec_array=spec_array, wavelength=wavelength, axis=axis)
    if axis == 0:
        result = result.loc[:, ['SI5']]
    else:
        result = result.loc[['SI5'], :]
    return result


@simple_type_validator
def ndsi(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute NDSI (Normalized Difference Salinity Index) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Nguyen, KA., Liou, YA., Tran, HP. et al.
        Soil salinity assessment by using near-infrared channel and Vegetation Soil Salinity Index derived from Landsat 8 OLI data: a case study in the Tra Vinh Province, Mekong Delta, Vietnam.
        Prog Earth Planet Sci 7, 1 (2020).
        https://doi.org/10.1186/s40645-019-0311-0

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
    result = si(spec_array=spec_array, wavelength=wavelength, axis=axis)
    if axis == 0:
        result = result.loc[:, ['NDSI']]
    else:
        result = result.loc[['NDSI'], :]
    return result


@simple_type_validator
def vssi(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute VSSI (Vegetation Soil Salinity Index) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        Nguyen, KA., Liou, YA., Tran, HP. et al.
        Soil salinity assessment by using near-infrared channel and Vegetation Soil Salinity Index derived from Landsat 8 OLI data: a case study in the Tra Vinh Province, Mekong Delta, Vietnam.
        Prog Earth Planet Sci 7, 1 (2020).
        https://doi.org/10.1186/s40645-019-0311-0

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
    result = si(spec_array=spec_array, wavelength=wavelength, axis=axis)
    if axis == 0:
        result = result.loc[:, ['VSSI']]
    else:
        result = result.loc[['VSSI'], :]
    return result


# %% Local test

# from specpipe.vegeind.demo_data import create_vegeind_demo_data

# specdata = create_vegeind_demo_data()
# vidata = si(specdata, wavelength=specdata.columns)
