# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 00:07:06 2025

@author: Siwei
"""
import numpy as np
import pandas as pd
from typing import Optional

from ..specio import simple_type_validator

# # For local test
# from specpipe.specio import simple_type_validator


# %% Demo spectral data


# Demo spectral data
@simple_type_validator
def create_vegeind_demo_data(nsample: int = 20, nband: int = 462, *, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate demostration 2D spectral data array with column names as wavelengths.

    Parameters
    ----------
    nsample : int, optional
        Number of samples. The default is 20.
    nband : int, optional
        Number of bands. The default is 462.
    seed : Optional[int], optional
        Seed for random data. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    df_spec : TYPE
        DESCRIPTION.

    """
    if nsample < 1:
        raise ValueError(f"nsample must be at least 1, got: {nsample}")
    wavelengths_raw = [(387.6 + 1.34 * i) for i in range(nband)]
    wavelengths = np.array(wavelengths_raw)

    # Spectral data
    if seed is not None:
        np.random.seed(seed)
    spec_arr = np.random.randint(0, 10000, (nsample, len(wavelengths)))
    df_spec = pd.DataFrame(spec_arr, columns=wavelengths, dtype='int16')

    return df_spec
