# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 12:58:21 2025

@author: Siwei
"""

import numpy as np
import pandas as pd

from typing import Annotated, Any
import inspect

from ..specio import simple_type_validator, arraylike_validator

from .ndvi import ndvi
from .sr import sr
from .dvi import dvi
from .evi import evi
from .gndvi import gndvi
from .cvi import cvi
from .mcari import mcari
from .mtci import mtci
from .s2rep import s2rep
from .tgi import tgi
from .ndre import ndre
from .lci import lci
from .sr_nir_r import sr_nir_r
from .sr_nir_g import sr_nir_g
from .pssr import pssr
from .pad import padvi
from .pri import pri
from .cci import cci
from .cri import cri
from .sipi import sipi
from .wi import wi
from .nwi import nwi
from .savi import savis
from .sasi import sasi1, sasi2, sasi3, sasi4
from .si import si
from .sr_salinity import sr_salinity
from .vog1 import vog1
from .ari import ari
from .osavi import osavi
from .msavi import msavi
from .swsi import swsi1, swsi2, swsi3


# %% Summarize available vegetation indices


@simple_type_validator
def vegeind_summary(
    spec_array: Annotated[Any, arraylike_validator(ndim=2)],
    wavelength: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
) -> pd.DataFrame:
    """
    Calculate a collection of common vegetation indices.

    Parameters
    ----------
    spec_array : 2D array-like, shape (n_samples, n_features)
        2D array-like of 1D spectral data series.
    wavelength : 1D array-like
        Wavelengths in nanometer for the spectra data.
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
    ndvi: Normalized Difference Vegetation Index
    sr: Simple ratio, simple ratio of R800 / R675
    dvi: Difference Vegetation Index
    evi: Enhanced Vegetation Index
    gndvi: Green Normalized Difference Vegetation Index
    cvi: Chlorophyll Vegetation Index
    mcari: Modified Chlorophyll Absorption Reflectance Index
    mtci: MERIS Terrestrial Chlorophyll Index
    s2rep: Sentinel-2 Red Edge Position
    tgi: Triangular Greenness Index
    ndre: Normalized Difference Red Edge Index
    lci: Leaf Chlorophyll Index
    sr_nir_r: Simple Ratio NIR/Red
    sr_nir_g: Simple Ratio NIR/Green
    pssr: Pigment-specific Simple Ratio
    pad: Derivative Pseudo-Absorption
    padvi: Derivative Pseudo-Absorption Vegetation Indices
    pri: Photochemical Reflectance Index
    cci: Chlorophyll/Carotenoid Index
    cri: Carotenoid Reflectance Indices
    sipi: Structure Insensitive Pigment Index
    wi: Water Index
    nwi: Normalized Water Index
    savis: Soil Adjusted Vegetation Indices
    sasi1: Soil Adjusted Salinity Index 1
    sasi2: Soil Adjusted Salinity Index 2
    sasi3: Soil Adjusted Salinity Index 3
    sasi4: Soil Adjusted Salinity Index 4
    si: Salinity Indices
    sr_salinity: Simple Ratio R750/R705
    vog1: Vogelmann Red Edge Index
    ari: Anthocyanin Reflectance Index
    osavi: Optimized Soil Adjusted Salinity Index
    msavi: Modified Soil Adjusted Salinity Index
    swsi1: Salinity and Water Stress Index 1
    swsi2: Salinity and Water Stress Index 2
    swsi3: Salinity and Water Stress Index 3

    Examples
    --------
    Prepare demo spectral data::

        >>> from specpipe.vegeind import create_vegeind_demo_data
        >>> df_spec = create_vegeind_demo_data()

    Summarize vegetation indices of the spectral data::

        >>> vegeind_summary(df_spec, df_spec.columns)
    """
    vegeind_funcs = [
        # Classic
        ndvi,
        sr,
        dvi,
        evi,
        gndvi,
        # Chlorophyll
        cvi,
        mcari,
        mtci,
        s2rep,
        tgi,
        ndre,
        lci,
        sr_nir_r,
        sr_nir_g,
        pssr,
        padvi,
        # Carotenoids
        pri,
        cci,
        cri,
        sipi,
        # Water content
        wi,
        nwi,
        # Salinity stress
        savis,
        sasi1,
        sasi2,
        sasi3,
        sasi4,
        si,
        sr_salinity,
        vog1,
        ari,
        osavi,
        msavi,
        swsi1,
        swsi2,
        swsi3,
    ]

    if axis == 1:
        spec_array = spec_array.T

    result_summary = pd.DataFrame(np.zeros((spec_array.shape[0], 1)), columns=["_preserved_for_init"])
    for vegeind in vegeind_funcs:
        try:
            # Check parameters for functions
            # Soil influence parameters
            is_l_param: bool = "L" in inspect.signature(vegeind).parameters
            if is_l_param:
                for L in [0.25, 0.5, 0.75, 1.0]:  # noqa: N806
                    result_ind = vegeind(spec_array=spec_array, wavelength=wavelength, axis=0, L=L)
                    result_summary = pd.concat(
                        [result_summary.reset_index(drop=True), result_ind.reset_index(drop=True)], axis=1
                    )
            else:
                result_ind = vegeind(spec_array=spec_array, wavelength=wavelength, axis=0)
                result_summary = pd.concat(
                    [result_summary.reset_index(drop=True), result_ind.reset_index(drop=True)], axis=1
                )
        except Exception:
            pass

    # Delete initialization
    result_summary = result_summary.drop("_preserved_for_init", axis=1)

    if axis == 1:
        result_summary = result_summary.T

    return result_summary
