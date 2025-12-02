# -*- coding: utf-8 -*-
"""
SpecPipe - Vegetation index module
"""
from .demo_data import create_vegeind_demo_data
from specpipe.vegeind.ndvi import ndvi
from specpipe.vegeind.sr import sr
from specpipe.vegeind.dvi import dvi
from specpipe.vegeind.evi import evi
from specpipe.vegeind.gndvi import gndvi
from specpipe.vegeind.cvi import cvi
from specpipe.vegeind.mcari import mcari
from specpipe.vegeind.mtci import mtci
from specpipe.vegeind.s2rep import s2rep
from specpipe.vegeind.tgi import tgi
from specpipe.vegeind.ndre import ndre
from specpipe.vegeind.lci import lci
from specpipe.vegeind.sr_nir_r import sr_nir_r
from specpipe.vegeind.sr_nir_g import sr_nir_g
from specpipe.vegeind.pssr import pssr, pssr1, pssr2
from specpipe.vegeind.pad import pad, padvi, pad1_r729, pad1_r732, pad2_r664
from specpipe.vegeind.pri import pri
from specpipe.vegeind.cci import cci
from specpipe.vegeind.cri import cri, cri550, cri700, rnir_to_r510
from specpipe.vegeind.sipi import sipi
from specpipe.vegeind.wi import wi
from specpipe.vegeind.nwi import nwi, nwi1, nwi2, nwi3, nwi4
from specpipe.vegeind.savi import savi, savis
from specpipe.vegeind.sasi import sasi1, sasi2, sasi3, sasi4
from specpipe.vegeind.si import si, si1, si2, si3, si4, si5, ndsi, vssi
from specpipe.vegeind.sr_salinity import sr_salinity
from specpipe.vegeind.vog1 import vog1
from specpipe.vegeind.ari import ari
from specpipe.vegeind.osavi import osavi
from specpipe.vegeind.msavi import msavi
from specpipe.vegeind.swsi import swsi1, swsi2, swsi3

__all__ = [
    # Classic
    "create_vegeind_demo_data",
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
]
