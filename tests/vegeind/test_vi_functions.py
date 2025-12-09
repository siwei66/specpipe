# -*- coding: utf-8 -*-
"""
Tests for Vegetation Index (VI) functions

Copyright (c) 2025 Siwei Luo. MIT License.
"""
import numpy as np
import pandas as pd
import pytest
import sys

from specpipe.vegeind.demo_data import create_vegeind_demo_data

from specpipe.vegeind.vegeind_summary import vegeind_summary

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


# %% TestArrayOutlier


class TestVegetationIndex:
    """Test functionalities of vegetation index functions."""

    @staticmethod
    def test_ndvi() -> None:
        """Test NDVI function"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = ndvi(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['NDVI']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_evi() -> None:
        """Test EVI function"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = evi(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['EVI']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_gndvi() -> None:
        """Test GNDVI function"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = gndvi(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['GNDVI']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_dvi() -> None:
        """Test DVI function"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = dvi(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['DVI']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_cvi() -> None:
        """Test CVI function"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = cvi(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['CVI']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_sr() -> None:
        """Test SR functions"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = sr(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['SR']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_mcari() -> None:
        """Test MCARI function"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = mcari(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['MCARI']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_mtci() -> None:
        """Test MTCI function"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = mtci(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['MTCI']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_s2rep() -> None:
        """Test S2REP function"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = s2rep(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['S2REP']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_tgi() -> None:
        """Test TGI function"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = tgi(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['TGI']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_ndre() -> None:
        """Test NDRE function"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = ndre(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['NDRE']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_lci() -> None:
        """Test LCI function"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = lci(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['LCI']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_sr_nir_r() -> None:
        """Test SR NIR/Red function"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = sr_nir_r(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['SR_NIR_to_R']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_sr_nir_g() -> None:
        """Test SR NIR/Green function"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = sr_nir_g(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['SR_NIR_to_G']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_pssr() -> None:
        """Test PSSR functions"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = pssr(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 2)
        assert list(vidata.columns) == ['PSSR_R810_to_R676', 'PSSR_R810_to_R682']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        # Single
        vidata = pssr1(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['PSSR_R810_to_R676']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = pssr2(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['PSSR_R810_to_R682']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_pad() -> None:
        """Test PAD functions"""
        specdata = create_vegeind_demo_data(seed=66)
        padv1 = pad(specdata, order=1)
        assert isinstance(padv1, np.ndarray)
        assert padv1.shape == specdata.shape
        assert not np.any(np.isnan(padv1))
        padv2 = pad(specdata, order=2)
        assert isinstance(padv2, np.ndarray)
        assert padv2.shape == specdata.shape
        assert not np.any(np.isnan(padv1))
        vidata = padvi(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 6)
        assert list(vidata.columns) == ['PAD1_R453', 'PAD1_R717', 'PAD1_R729', 'PAD1_R732', 'PAD2_R664', 'PAD2_R726']
        assert not vidata.isna().any().any()
        vidata = pad1_r729(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['PAD1_R729']
        assert not vidata.isna().any().any()
        vidata = pad1_r732(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['PAD1_R732']
        assert not vidata.isna().any().any()
        vidata = pad2_r664(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['PAD2_R664']
        assert not vidata.isna().any().any()

    @staticmethod
    def test_pri() -> None:
        """Test PRI function"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = pri(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['PRI']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_cci() -> None:
        """Test CCI function"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = cci(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['CCI']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_cri() -> None:
        """Test CRI functions"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = cri(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 3)
        assert list(vidata.columns) == ['CRI550', 'CRI700', 'RNIR_to_R510']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = cri550(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['CRI550']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = cri700(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['CRI700']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = rnir_to_r510(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['RNIR_to_R510']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_sipi() -> None:
        """Test SIPI function"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = sipi(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['SIPI']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_wi() -> None:
        """Test WI function"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = wi(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['WI']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_nwi() -> None:
        """Test NWI functions"""
        specdata = create_vegeind_demo_data(seed=66)
        vidata = nwi(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 4)
        assert list(vidata.columns) == ['NWI1', 'NWI2', 'NWI3', 'NWI4']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = nwi1(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['NWI1']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = nwi2(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['NWI2']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = nwi3(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['NWI3']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = nwi4(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['NWI4']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_savi() -> None:
        """Test SAVI functions"""
        specdata = create_vegeind_demo_data(seed=66)
        # Test SAVI with different L
        vidata = savi(specdata, wavelength=specdata.columns, L=0.75)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['SAVI_L0d75']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = savi(specdata, wavelength=specdata.columns, L=1)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['SAVI_L1']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        # Test SAVIs
        vidata = savis(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 6)
        assert list(vidata.columns) == ['SAVI_L0', 'SAVI_L0d25', 'SAVI_L0d5', 'SAVI_L0d75', 'SAVI_L1', 'SAVI_L2']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_sasi() -> None:
        """Test SASI functions"""
        specdata = create_vegeind_demo_data(seed=66)
        # Test SASI with different L
        vidata = sasi1(specdata, wavelength=specdata.columns, L=0.5)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['SASI1_L0d5']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = sasi1(specdata, wavelength=specdata.columns, L=0.75)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['SASI1_L0d75']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        # Test different SASIs
        vidata = sasi2(specdata, wavelength=specdata.columns, L=0.5)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['SASI2_L0d5']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = sasi3(specdata, wavelength=specdata.columns, L=0.5)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['SASI3_L0d5']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = sasi4(specdata, wavelength=specdata.columns, L=0.5)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ['SASI4_L0d5']
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_si() -> None:
        """Test SI functions"""
        specdata = create_vegeind_demo_data(seed=66)
        # Test SI
        vidata = si(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 7)
        assert list(vidata.columns) == ["SI1", "SI2", "SI3", "SI4", "SI5", "NDSI", "VSSI"]
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        # Test single SI
        vidata = si1(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ["SI1"]
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = si2(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ["SI2"]
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = si3(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ["SI3"]
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = si4(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ["SI4"]
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = si5(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ["SI5"]
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = ndsi(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ["NDSI"]
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = vssi(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ["VSSI"]
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_sr_sal() -> None:
        """Test SR_salinity function"""
        specdata = create_vegeind_demo_data(seed=66)
        # Test basic
        vidata = sr_salinity(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ["SR_salinity"]
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_vog1() -> None:
        """Test VOG1 function"""
        specdata = create_vegeind_demo_data(seed=66)
        # Test basic
        vidata = vog1(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ["VOG1"]
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_ari() -> None:
        """Test ARI function"""
        specdata = create_vegeind_demo_data(seed=66)
        # Test basic
        vidata = ari(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ["ARI"]
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_osavi() -> None:
        """Test OSAVI function"""
        specdata = create_vegeind_demo_data(seed=66)
        # Test basic
        vidata = osavi(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ["OSAVI"]
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_msavi() -> None:
        """Test MSAVI function"""
        specdata = create_vegeind_demo_data(seed=66)
        # Test basic
        vidata = msavi(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ["MSAVI"]
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_swsi() -> None:
        """Test SWSI functions"""
        specdata = create_vegeind_demo_data(nband=12000, seed=66)
        # Test basic
        vidata = swsi1(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ["SWSI1"]
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = swsi2(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ["SWSI2"]
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        vidata = swsi3(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape == (20, 1)
        assert list(vidata.columns) == ["SWSI3"]
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()

    @staticmethod
    def test_vegeind_summary() -> None:
        """Test vegeind_summary functionality"""
        # Test full band range coveragy
        specdata = create_vegeind_demo_data(nband=12000, seed=66)
        # Test basic
        vidata = vegeind_summary(specdata, wavelength=specdata.columns)
        assert isinstance(vidata, pd.DataFrame)
        assert vidata.shape[0] == 20
        assert not vidata.isna().any().any()
        assert not (vidata == 0).any().any()
        # Test partial band range coverage making some index not work
        specdata1 = create_vegeind_demo_data(nband=462, seed=66)
        # Test basic
        vidata1 = vegeind_summary(specdata1, wavelength=specdata1.columns)
        assert isinstance(vidata1, pd.DataFrame)
        assert vidata1.shape[0] <= 20
        assert vidata1.shape[1] < vidata.shape[1]
        assert not vidata1.isna().any().any()
        assert not (vidata1 == 0).any().any()


# %% Test main

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
