# -*- coding: utf-8 -*-
"""
Tests for Swectral MinMax (MinMax Normalization) functions

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# ruff: noqa: I001
# OS
import os  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

# Initialize LOKY_MAX_CPU_COUNT if it does not exist before imports to prevent corresponding warning
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')  # noqa: E402

# OS Files
import shutil  # noqa: E402
import tempfile  # noqa: E402

# Test
import pytest  # noqa: E402
import unittest  # noqa: E402


# Basic data
import numpy as np  # noqa: E402
import torch  # noqa: E402

# Raster
import rasterio  # noqa: E402

# Local
from swectral.example_data import create_test_raster  # noqa: E402

# Function to test
from swectral.rasterop import pixel_apply  # noqa: E402
from swectral.functions.minmax import minmax  # noqa: E402
from swectral.functions.minmax_hyper import minmax_hyper  # noqa: E402

# Check if cuda is available
try:
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


# %% Test


class TestMinMax(unittest.TestCase):
    """Test snv and snv_hyper functionalities."""

    test_dir: str = ""
    img_path: str = ""
    dst_path: str = ""
    dst_path_hyper: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a temporary directory and a test image."""
        cls.test_dir = tempfile.mkdtemp()
        cls.img_path = cls.test_dir + "/test_img.tif"
        cls.dst_path = cls.test_dir + "/processed.tif"
        cls.dst_path_hyper = cls.test_dir + "/processed_hyper.tif"
        create_test_raster(raster_path=cls.img_path, width=50, height=50, bands=4)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary directory"""
        shutil.rmtree(cls.test_dir)

    @staticmethod
    def test_minmax() -> None:
        """Test minmax basic functionality."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
            pixel_apply(
                image_path=TestMinMax.img_path,
                output_path=TestMinMax.dst_path,
                spectral_function=minmax,
                tile_size=1,
                progress=False,
                function_type='array',
            )
            assert os.path.exists(TestMinMax.dst_path)

    @staticmethod
    def test_minmax_hyper() -> None:
        """Test minmax_hyper basic functionality."""
        if HAS_CUDA:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
                pixel_apply(
                    image_path=TestMinMax.img_path,
                    output_path=TestMinMax.dst_path_hyper,
                    spectral_function=minmax_hyper,
                    tile_size=1,
                    progress=False,
                    function_type='tensor_hyper',
                )
                assert os.path.exists(TestMinMax.dst_path)
                with rasterio.open(TestMinMax.dst_path) as src1, rasterio.open(TestMinMax.dst_path_hyper) as src2:
                    data1 = src1.read()
                    data2 = src2.read()
                    np.testing.assert_allclose(
                        data1, data2, rtol=1e-5, atol=1e-5, err_msg="Inconsistent resulting rasters."
                    )
        else:
            return


# %% Test main


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
