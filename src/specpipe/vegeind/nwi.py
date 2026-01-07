# -*- coding: utf-8 -*-
"""
SpecPipe - Vegetation Indices - NWI (Normalized Water Index)

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
def nwi(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute NWI (Normalized Water Index) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        K.K. Bandyopadhyay, S. Pradhan, R.N. Sahoo, Ravender Singh, V.K. Gupta, D.K. Joshi, A.K. Sutradhar,
        Characterization of water stress and prediction of yield of wheat using spectral indices under varied water and nitrogen management practices,
        Agricultural Water Management,
        Volume 146,
        2014,
        Pages 115-123,
        ISSN 0378-3774,
        https://doi.org/10.1016/j.agwat.2014.07.017.

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
    if (min(wavelength) > 850) | (max(wavelength) < 969):
        raise ValueError(
            'Index requires wavelength range (850, 970) '
            + f'exceeds given wavelength range ({min(wavelength)}, {max(wavelength)}).'
        )

    # Spectra indices of the wavelength
    r850_ind = (np.abs(wavelength - 850)).argmin()
    r880_ind = (np.abs(wavelength - 880)).argmin()
    r900_ind = (np.abs(wavelength - 900)).argmin()
    r920_ind = (np.abs(wavelength - 920)).argmin()
    r970_ind = (np.abs(wavelength - 970)).argmin()
    r850 = spec_array[:, r850_ind]
    r880 = spec_array[:, r880_ind]
    r900 = spec_array[:, r900_ind]
    r920 = spec_array[:, r920_ind]
    r970 = spec_array[:, r970_ind]

    # Compute vegetation indices
    nwi1_denominator = r970 + r900
    nwi2_denominator = r970 + r850
    nwi3_denominator = r970 + r920
    nwi4_denominator = r970 + r880
    nwi1_denominator[nwi1_denominator == 0] = 1e-15
    nwi2_denominator[nwi2_denominator == 0] = 1e-15
    nwi3_denominator[nwi3_denominator == 0] = 1e-15
    nwi4_denominator[nwi4_denominator == 0] = 1e-15

    nwi1_values = (r970 - r900) / nwi1_denominator
    nwi2_values = (r970 - r850) / nwi2_denominator
    nwi3_values = (r970 - r920) / nwi3_denominator
    nwi4_values = (r970 - r880) / nwi4_denominator

    # Convert values to df and output
    vi_arr = np.array(
        [
            nwi1_values,
            nwi2_values,
            nwi3_values,
            nwi4_values,
        ]
    )
    vi_names = [
        'NWI1',
        'NWI2',
        'NWI3',
        'NWI4',
    ]
    if axis == 0:
        df_vi = pd.DataFrame(vi_arr.T, columns=vi_names)
    else:
        df_vi = pd.DataFrame(vi_arr, index=vi_names)

    return df_vi


@simple_type_validator
def nwi1(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute NWI1 (Normalized Water Index - 1) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        K.K. Bandyopadhyay, S. Pradhan, R.N. Sahoo, Ravender Singh, V.K. Gupta, D.K. Joshi, A.K. Sutradhar,
        Characterization of water stress and prediction of yield of wheat using spectral indices under varied water and nitrogen management practices,
        Agricultural Water Management,
        Volume 146,
        2014,
        Pages 115-123,
        ISSN 0378-3774,
        https://doi.org/10.1016/j.agwat.2014.07.017.

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
    if axis == 0:
        df_vi = nwi(spec_array=spec_array, wavelength=wavelength, axis=axis).loc[:, ['NWI1']]
    else:
        df_vi = nwi(spec_array=spec_array, wavelength=wavelength, axis=axis).loc[['NWI1'], :]
    return df_vi


@simple_type_validator
def nwi2(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute NWI1 (Normalized Water Index - 2) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        K.K. Bandyopadhyay, S. Pradhan, R.N. Sahoo, Ravender Singh, V.K. Gupta, D.K. Joshi, A.K. Sutradhar,
        Characterization of water stress and prediction of yield of wheat using spectral indices under varied water and nitrogen management practices,
        Agricultural Water Management,
        Volume 146,
        2014,
        Pages 115-123,
        ISSN 0378-3774,
        https://doi.org/10.1016/j.agwat.2014.07.017.

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
    if axis == 0:
        df_vi = nwi(spec_array=spec_array, wavelength=wavelength, axis=axis).loc[:, ['NWI2']]
    else:
        df_vi = nwi(spec_array=spec_array, wavelength=wavelength, axis=axis).loc[['NWI2'], :]
    return df_vi


@simple_type_validator
def nwi3(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute NWI1 (Normalized Water Index - 3) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        K.K. Bandyopadhyay, S. Pradhan, R.N. Sahoo, Ravender Singh, V.K. Gupta, D.K. Joshi, A.K. Sutradhar,
        Characterization of water stress and prediction of yield of wheat using spectral indices under varied water and nitrogen management practices,
        Agricultural Water Management,
        Volume 146,
        2014,
        Pages 115-123,
        ISSN 0378-3774,
        https://doi.org/10.1016/j.agwat.2014.07.017.

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
    if axis == 0:
        df_vi = nwi(spec_array=spec_array, wavelength=wavelength, axis=axis).loc[:, ['NWI3']]
    else:
        df_vi = nwi(spec_array=spec_array, wavelength=wavelength, axis=axis).loc[['NWI3'], :]
    return df_vi


@simple_type_validator
def nwi4(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Compute NWI1 (Normalized Water Index - 4) of 2D array-like of 1D spectra data series.
    The calculation is based on:

        K.K. Bandyopadhyay, S. Pradhan, R.N. Sahoo, Ravender Singh, V.K. Gupta, D.K. Joshi, A.K. Sutradhar,
        Characterization of water stress and prediction of yield of wheat using spectral indices under varied water and nitrogen management practices,
        Agricultural Water Management,
        Volume 146,
        2014,
        Pages 115-123,
        ISSN 0378-3774,
        https://doi.org/10.1016/j.agwat.2014.07.017.

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
    if axis == 0:
        df_vi = nwi(spec_array=spec_array, wavelength=wavelength, axis=axis).loc[:, ['NWI4']]
    else:
        df_vi = nwi(spec_array=spec_array, wavelength=wavelength, axis=axis).loc[['NWI4'], :]
    return df_vi


# %% Local test

# from specpipe.vegeind.demo_data import create_vegeind_demo_data

# specdata = create_vegeind_demo_data()
# vidata = nwi(specdata, wavelength=specdata.columns)
