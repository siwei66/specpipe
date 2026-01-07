# -*- coding: utf-8 -*-
"""
SpecPipe - Vegetation Indices - Demo data creator for spectral indices

Copyright (c) 2025 Siwei Luo. MIT License.
"""
import numpy as np
import pandas as pd
from typing import Optional, Union

from ..specio import simple_type_validator

# # For local test
# from specpipe.specio import simple_type_validator


# %% Demo spectral data


# Demo spectral data
@simple_type_validator
def create_specind_demo_data(
    nsample: int = 20,
    nband: int = 462,
    wavelength_range: tuple[Union[int, float], Union[int, float]] = (387.6, 1005.6),
    *,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate demostration 2D spectral data array with column names as wavelengths.

    The function is for test and demonstration purpose of spectral index functions.

    Parameters
    ----------
    nsample : int, optional
        Number of samples. The default is 20.
    nband : int, optional
        Number of bands. The default is 462.
    wavelength_range: tuple of (int or float, int or float)
        Range of wavelength.
    seed : int or None, optional
        Seed for random data. The default is None.

    Returns
    -------
    pandas.DataFrame
        Exemplary spectral data table with wavelengths for spectral index computation.

    Examples
    --------
    Use default settings::

        >>> df_spec = create_vegeind_demo_data()

    Customize sample number, number of bands and wavelength range::

        >>> df_spec = create_vegeind_demo_data(20, 1000, (50, 15000))
    """
    if nsample < 1:
        raise ValueError(f"nsample must be at least 1, got: {nsample}")
    if nband < 3:
        raise ValueError(f"nband must be at least 3, got: {nband}")
    spec_res = (max(wavelength_range) - min(wavelength_range)) / (nband - 1)
    wavelengths_raw = [(387.6 + spec_res * i) for i in range(nband)]
    wavelengths = np.asarray(np.round(wavelengths_raw, 1))

    # Spectral data
    if seed is not None:
        np.random.seed(seed)
    spec_arr = np.random.randint(0, 10000, (nsample, len(wavelengths)))
    df_spec = pd.DataFrame(spec_arr, columns=wavelengths, dtype='int16')

    return df_spec
