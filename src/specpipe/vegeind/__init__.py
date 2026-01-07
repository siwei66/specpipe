# -*- coding: utf-8 -*-
"""
SpecPipe vegetation spectral index submodule
"""
from .demo_data import create_specind_demo_data
from .vegeind_summary import vegeind_summary
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
from .pssr import pssr, pssr1, pssr2
from .pad import pad, padvi, pad1_r729, pad1_r732, pad2_r664
from .pri import pri
from .cci import cci
from .cri import cri, cri550, cri700, rnir_to_r510
from .sipi import sipi
from .wi import wi
from .nwi import nwi, nwi1, nwi2, nwi3, nwi4
from .savi import savi, savis
from .sasi import sasi1, sasi2, sasi3, sasi4
from .si import si, si1, si2, si3, si4, si5, ndsi, vssi
from .sr_salinity import sr_salinity
from .vog1 import vog1
from .ari import ari
from .osavi import osavi
from .msavi import msavi
from .swsi import swsi1, swsi2, swsi3

__all__ = [
    # Summary
    "vegeind_summary",
    # Classic
    "ndvi",
    "sr",
    "dvi",
    "evi",
    "gndvi",
    # Chlorophyll
    "cvi",
    "mcari",
    "mtci",
    "s2rep",
    "tgi",
    "ndre",
    "lci",
    "sr_nir_r",
    "sr_nir_g",
    "pssr",
    "pssr1",
    "pssr2",
    "pad",
    "padvi",
    "pad1_r729",
    "pad1_r732",
    "pad2_r664",
    # Carotenoids
    "pri",
    "cci",
    "cri",
    "cri550",
    "cri700",
    "rnir_to_r510",
    "sipi",
    # Water content
    "wi",
    "nwi",
    "nwi1",
    "nwi2",
    "nwi3",
    "nwi4",
    # Salinity stress
    "savi",
    "savis",
    "sasi1",
    "sasi2",
    "sasi3",
    "sasi4",
    "si",
    "si1",
    "si2",
    "si3",
    "si4",
    "si5",
    "ndsi",
    "vssi",
    "sr_salinity",
    "vog1",
    "ari",
    "osavi",
    "msavi",
    "swsi1",
    "swsi2",
    "swsi3",
    "create_specind_demo_data",
]
