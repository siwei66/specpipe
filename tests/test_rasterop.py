# -*- coding: utf-8 -*-
"""
Tests for raster operation functions (RasterOp)

The following source code was created with AI assistance and has been human reviewed and edited.

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Test
import pytest
import unittest

# OS Files
import os
import tempfile

# Typing
from typing import Union

# Testing third
import numpy as np
import torch

# Rasters
import rasterio

# Self
# Applied package functions for test
from specpipe.example_data import create_test_raster

# Funcs to test
from specpipe.rasterop import (
    auto_fp,
    croproi,
    dtype_mapper,
    pixel_apply,
    pixel_array_apply,
    pixel_spec_apply,
    pixel_tensor_apply,
    pixel_tensor_hyper_apply,
)

# Check if cuda is available
try:
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

# %% test functions : croproi


class TestCropROI(unittest.TestCase):
    @staticmethod
    def test_croproi_basic_functionality() -> None:
        """Test basic cropping functionality with a simple rectangle ROI."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test raster with 50x100 dimensions and 100 bands
            raster_path = os.path.join(temp_dir, "test_raster.tif")
            output_path = os.path.join(temp_dir, "cropped.tif")

            create_test_raster(raster_path, width=100, height=50, bands=4)

            # Define ROI coordinates (a rectangle in the center)
            roi_coordinates = [
                [
                    (25.0, 10.0),  # Top-left
                    (75.0, 10.0),  # Top-right
                    (75.0, 40.0),  # Bottom-right
                    (25.0, 40.0),  # Bottom-left
                    (25.0, 10.0),  # Close the polygon
                ]
            ]

            # Call the function
            croproi(raster_path, roi_coordinates, output_path)

            # Verify output file exists
            assert os.path.exists(output_path)

            # Verify output properties
            with rasterio.open(output_path) as src:
                # Should be cropped to 50x30 (from 25-75 in width, 10-40 in height)
                assert src.width == 50
                assert src.height == 30
                assert src.count == 4  # All 100 bands preserved

                # Read some sample data to verify it's not all zeros
                data = src.read(1)  # Read first band
                assert not np.all(data == 0)  # Should have non-zero values

    @staticmethod
    def test_croproi_multiple_polygons() -> None:
        """Test cropping with multiple ROI polygons."""
        with tempfile.TemporaryDirectory() as temp_dir:
            raster_path = os.path.join(temp_dir, "test_raster.tif")
            output_path = os.path.join(temp_dir, "cropped_multi.tif")

            # Create test raster with 50x100 dimensions and 120 bands
            create_test_raster(raster_path, width=100, height=50, bands=4)

            # Define two separate ROI polygons
            roi_coordinates = [
                [  # First polygon (top-left corner)
                    (0.0, 0.0),
                    (25.0, 0.0),
                    (25.0, 15.0),
                    (0.0, 15.0),
                    (0.0, 0.0),
                ],
                [  # Second polygon (bottom-right corner)
                    (75.0, 35.0),
                    (100.0, 35.0),
                    (100.0, 50.0),
                    (75.0, 50.0),
                    (75.0, 35.0),
                ],
            ]

            croproi(raster_path, roi_coordinates, output_path)

            assert os.path.exists(output_path)

            with rasterio.open(output_path) as src:
                # Should contain both regions with all bands
                assert src.width > 0
                assert src.height > 0
                assert src.count == 4  # All bands preserved

    @staticmethod
    def test_croproi_high_band_count() -> None:
        """Test cropping with a very high number of bands."""
        with tempfile.TemporaryDirectory() as temp_dir:
            raster_path = os.path.join(temp_dir, "test_raster.tif")
            output_path = os.path.join(temp_dir, "cropped_high_bands.tif")

            # Create test raster with 50x100 dimensions and 224 bands (like AVIRIS)
            create_test_raster(raster_path, width=100, height=50, bands=224)

            roi_coordinates = [[(20.0, 10.0), (80.0, 10.0), (80.0, 40.0), (20.0, 40.0), (20.0, 10.0)]]

            croproi(raster_path, roi_coordinates, output_path)

            assert os.path.exists(output_path)

            with rasterio.open(output_path) as src:
                assert src.count == 224  # All 224 bands preserved
                assert src.width == 60  # 80-20 = 60
                assert src.height == 30  # 40-10 = 30

    @staticmethod
    def test_croproi_complex_polygons() -> None:
        """Test cropping with complex polygon shapes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            raster_path = os.path.join(temp_dir, "test_raster.tif")
            output_path = os.path.join(temp_dir, "cropped_complex.tif")

            # Create test raster with 50x100 dimensions and 150 bands
            create_test_raster(raster_path, width=100, height=50, bands=150)

            # Complex polygon with multiple vertices
            roi_coordinates = [
                [
                    (10.0, 5.0),
                    (40.0, 5.0),
                    (60.0, 15.0),
                    (80.0, 5.0),
                    (90.0, 20.0),
                    (80.0, 35.0),
                    (90.0, 45.0),
                    (70.0, 45.0),
                    (50.0, 35.0),
                    (30.0, 45.0),
                    (15.0, 35.0),
                    (5.0, 20.0),
                    (10.0, 5.0),
                ]
            ]

            croproi(raster_path, roi_coordinates, output_path)

            assert os.path.exists(output_path)

            with rasterio.open(output_path) as src:
                assert src.count == 150  # All bands preserved
                assert src.width > 0
                assert src.height > 0

    @staticmethod
    def test_croproi_metadata_and_band_integrity() -> None:
        """Test that all bands and metadata are properly preserved."""
        with tempfile.TemporaryDirectory() as temp_dir:
            raster_path = os.path.join(temp_dir, "test_raster.tif")
            output_path = os.path.join(temp_dir, "cropped_meta.tif")

            # Create test raster with 50x100 dimensions and 200 bands
            create_test_raster(raster_path, width=100, height=50, bands=200)

            roi_coordinates = [[(20.0, 10.0), (80.0, 10.0), (80.0, 40.0), (20.0, 40.0), (20.0, 10.0)]]

            croproi(raster_path, roi_coordinates, output_path)

            with rasterio.open(raster_path) as src_orig:
                with rasterio.open(output_path) as src_cropped:
                    # Check that all metadata and bands are preserved
                    assert src_orig.count == src_cropped.count
                    assert src_orig.dtypes[0] == src_cropped.dtypes[0]
                    assert src_orig.crs == src_cropped.crs
                    assert src_orig.nodata == src_cropped.nodata

                    # Verify band names or descriptions if they exist
                    if hasattr(src_orig, "descriptions") and src_orig.descriptions[0]:
                        assert src_orig.descriptions == src_cropped.descriptions

    @staticmethod
    def test_croproi_edge_cases() -> None:
        """Test edge cases with different band counts and ROI positions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with exactly 100 bands
            raster_path = os.path.join(temp_dir, "test_raster_100.tif")
            output_path = os.path.join(temp_dir, "cropped_100.tif")

            create_test_raster(raster_path, width=100, height=50, bands=100)

            # ROI covering most of the image
            roi_coordinates = [[(5.0, 5.0), (95.0, 5.0), (95.0, 45.0), (5.0, 45.0), (5.0, 5.0)]]

            croproi(raster_path, roi_coordinates, output_path)

            with rasterio.open(output_path) as src:
                assert src.count == 100
                assert src.width == 90  # 95-5 = 90
                assert src.height == 40  # 45-5 = 40


# %% Test - croproi

# TestCropROI.test_croproi_basic_functionality()
# TestCropROI.test_croproi_multiple_polygons()
# TestCropROI.test_croproi_high_band_count()
# TestCropROI.test_croproi_complex_polygons()
# TestCropROI.test_croproi_metadata_and_band_integrity()
# TestCropROI.test_croproi_edge_cases()


# %% test functions : auto_fp


class TestAutoFP(unittest.TestCase):
    @staticmethod
    def test_numpy_short_types_preserved() -> None:
        """Test that numpy short types are preserved without conversion."""
        # Test int8
        data_int8 = np.array([1, 2, 3], dtype=np.int8)
        result, scale = auto_fp(data_int8)
        assert result.dtype == np.int8
        assert scale is None

        # Test uint8
        data_uint8 = np.array([1, 2, 3], dtype=np.uint8)
        result, scale = auto_fp(data_uint8)
        assert result.dtype == np.uint8
        assert scale is None

        # Test float16
        data_fp16 = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        result, scale = auto_fp(data_fp16)
        assert result.dtype == np.float16
        assert scale is None

    @staticmethod
    def test_numpy_long_types_compatible_fp16() -> None:
        """Test numpy long types that fit natively in FP16."""
        # Float32 data within FP16 range
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result, scale = auto_fp(data)
        assert result.dtype == np.float16
        assert scale == 1.0

        # Float64 data within FP16 range
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result, scale = auto_fp(data)
        assert result.dtype == np.float16
        assert scale == 1.0

    @staticmethod
    def test_numpy_scaling_large_values() -> None:
        """Test scaling of large numpy values to fit FP16."""
        # Create data that exceeds FP16 max
        large_data = np.array([100000, 200000], dtype=np.float32)

        result, scale = auto_fp(large_data, scaling=True)
        assert result is not None
        assert scale is not None
        assert result.dtype == np.float16
        assert scale < 1.0  # Should be a scaling down factor

        # Verify scaling was applied correctly
        expected_scaled = large_data * scale
        assert type(result) is np.ndarray
        assert np.allclose(result.astype(np.float32), expected_scaled, rtol=1e-4, atol=1e-5)

    @staticmethod
    def test_numpy_scaling_small_values() -> None:
        """Test scaling of small numpy values to fit FP16."""
        # Create data with values below FP16 min
        small_data = np.array([2e-7, 1e-7], dtype=np.float64)

        result, scale = auto_fp(small_data, scaling=True)
        assert result is not None
        assert scale is not None
        assert result.dtype == np.float16
        assert scale > 1.0  # Should be a scaling up factor

        # Verify scaling was applied correctly
        expected_scaled = small_data * scale
        assert np.allclose(result.astype(np.float64), expected_scaled, rtol=1e-4, atol=1e-5)

    @staticmethod
    def test_numpy_precision_loss_rejection() -> None:
        """Test that numpy data with precision loss is rejected."""
        # Create data that would lose precision when scaled
        problematic_data = np.array([1.23456789, 2.34567891], dtype=np.float64)

        result, scale = auto_fp(problematic_data, scaling=True, rtol=1e-8, atol=1e-8)
        assert result.dtype == np.float64  # Should return original
        assert scale is None

    @staticmethod
    def test_torch_fp16_preserved() -> None:
        """Test that torch float16 tensors are preserved."""
        data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        result, scale = auto_fp(data)
        assert result.dtype == torch.float16
        assert scale is None

    @staticmethod
    def test_torch_float32_compatible_fp16() -> None:
        """Test torch float32 tensors that fit natively in FP16."""
        data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result, scale = auto_fp(data)
        assert result is not None
        assert scale is not None
        assert result.dtype == torch.float16
        assert scale == 1.0

    @staticmethod
    def test_torch_scaling_large_values() -> None:
        """Test scaling of large torch tensor values to fit FP16."""
        # Create data that exceeds FP16 max
        large_data = torch.tensor([100000, 200000], dtype=torch.float32)

        result, scale = auto_fp(large_data, scaling=True)
        assert result is not None
        assert scale is not None
        assert result.dtype == torch.float16
        assert scale < 1.0

        # Verify scaling was applied correctly
        expected_scaled = large_data * scale
        assert torch.allclose(result.to(torch.float32), expected_scaled, rtol=1e-4, atol=1e-5)

    @staticmethod
    def test_torch_scaling_small_values() -> None:
        """Test scaling of small torch tensor values to fit FP16."""
        # Create data with values below FP16 min
        small_data = torch.tensor([2e-7, 1e-7], dtype=torch.float64)

        result, scale = auto_fp(small_data, scaling=True)
        assert result is not None
        assert scale is not None
        assert result.dtype == torch.float16
        assert scale > 1.0

        # Verify scaling was applied correctly
        expected_scaled = small_data * scale
        assert torch.allclose(result.to(torch.float64), expected_scaled, rtol=1e-4, atol=1e-5)

    @staticmethod
    def test_torch_no_scaling_option() -> None:
        """Test that torch tensors return original when scaling=False and incompatible."""
        fp16_max = 65504 / 2.0
        large_data = torch.tensor([fp16_max * 2], dtype=torch.float32)

        result, scale = auto_fp(large_data, scaling=False)
        assert result.dtype == torch.float32  # Should return original
        assert scale is None

    @staticmethod
    def test_torch_precision_loss_rejection() -> None:
        """Test that torch data with precision loss is rejected."""
        problematic_data = torch.tensor([1.23456789, 2.34567891], dtype=torch.float64)

        result, scale = auto_fp(problematic_data, scaling=True, rtol=1e-8, atol=1e-8)
        assert result.dtype == torch.float64  # Should return original
        assert scale is None

    @staticmethod
    def test_unsupported_data_type() -> None:
        """Test that unsupported data types raise ValueError."""
        # Test with complex numbers (unsupported)
        complex_data = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)

        try:
            auto_fp(complex_data)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "Unsupported data dtype" in str(e)

    @staticmethod
    def test_edge_case_zero_values() -> None:
        """Test handling of zero values which should be excluded from min_val calculation."""
        data_with_zeros = np.array([0.0, 1e-6, 0.001], dtype=np.float32)
        result, scale = auto_fp(data_with_zeros, scaling=True)

        # Should handle zeros gracefully without affecting scaling
        assert result.dtype == np.float16
        assert scale is not None

    @staticmethod
    def test_custom_tolerance_parameters() -> None:
        """Test that custom rtol and atol parameters are respected."""
        # Create data that would pass with default tolerance but fail with stricter
        borderline_data = np.array([1.0001, 2.0001], dtype=np.float64)

        # Should pass with default tolerance
        result1, scale1 = auto_fp(borderline_data, scaling=True, rtol=1e-3, atol=1e-3)
        assert result1.dtype == np.float16

        # Should fail with stricter tolerance
        result2, scale2 = auto_fp(borderline_data, scaling=True, rtol=1e-8, atol=1e-8)
        assert result2.dtype == np.float64
        assert scale2 is None

    @staticmethod
    def test_custom_safety_factor() -> None:
        """Test that custom safety_factor affects FP16 limits."""
        custom_safety = 1.1
        fp16_max_custom = 40000.0

        # Data that fits with custom safety but not with default
        data = np.array([fp16_max_custom], dtype=np.float32)

        # Should convert with custom safety factor
        result, scale = auto_fp(data, safety_factor=custom_safety)
        assert result.dtype == np.float16
        assert scale == 1.0

        # Should not convert with default safety factor
        result_default, scale_default = auto_fp(data, safety_factor=2.0)
        assert result_default.dtype == np.float32
        assert scale_default is None


# %% Test - auto_fp

# TestAutoFP.test_numpy_short_types_preserved()
# TestAutoFP.test_numpy_long_types_compatible_fp16()

# TestAutoFP.test_numpy_scaling_large_values()
# TestAutoFP.test_numpy_scaling_small_values()

# TestAutoFP.test_numpy_precision_loss_rejection()

# TestAutoFP.test_torch_fp16_preserved()
# TestAutoFP.test_torch_float32_compatible_fp16()

# TestAutoFP.test_torch_scaling_large_values()
# TestAutoFP.test_torch_scaling_small_values()

# TestAutoFP.test_torch_no_scaling_option()

# TestAutoFP.test_torch_precision_loss_rejection()

# TestAutoFP.test_unsupported_data_type()
# TestAutoFP.test_edge_case_zero_values()

# TestAutoFP.test_custom_tolerance_parameters()
# TestAutoFP.test_custom_safety_factor()


# %% test functions : dtype_mapper


def test_dtype_mapper() -> None:
    """Test dtype_mapper functionality"""

    # Test basic numpy mappings
    assert dtype_mapper("float32", "numpy") == np.float32
    assert dtype_mapper("int16", "numpy") == np.int16
    assert dtype_mapper("uint8", "numpy") == np.uint8
    assert dtype_mapper("bool", "numpy") == np.bool_

    # Test numpy type inputs
    assert dtype_mapper(np.float32, "numpy") == np.float32
    assert dtype_mapper(np.int64, "numpy") == np.int64

    # Test Python native types
    assert dtype_mapper(float, "numpy") == np.float32
    assert dtype_mapper(int, "numpy") == np.int32
    assert dtype_mapper(bool, "numpy") == np.bool_

    # Test raster mappings
    assert dtype_mapper("float32", "raster") == "float32"
    assert dtype_mapper("int16", "raster") == "int16"
    assert dtype_mapper("uint8", "raster") == "uint8"
    assert dtype_mapper(float, "raster") == "float32"
    assert dtype_mapper(int, "raster") == "int32"

    # Test raster compatibility mappings (min_compatible=True)
    assert dtype_mapper("float16", "raster") == "float32"
    assert dtype_mapper("int8", "raster") == "int16"

    # Test torch mappings
    assert dtype_mapper("float32", "torch") == torch.float32
    assert dtype_mapper("int32", "torch") == torch.int32
    assert dtype_mapper("uint8", "torch") == torch.uint8

    # Test torch compatibility mappings
    assert dtype_mapper("uint16", "torch") == torch.int32
    assert dtype_mapper(np.uint16, "torch") == torch.int32

    # Test various string representations
    assert dtype_mapper("double", "numpy") == np.float64
    assert dtype_mapper("half", "numpy") == np.float16
    assert dtype_mapper("int", "numpy") == np.int32
    assert dtype_mapper("float", "numpy") == np.float32

    # Test raster mappings for string representations
    assert dtype_mapper("double", "raster") == "float32"
    assert dtype_mapper("half", "raster") == "float32"

    # Test error cases
    with pytest.raises(ValueError, match="Unsupported map_type"):
        dtype_mapper("float32", "invalid_type")

    with pytest.raises(ValueError, match="Unsupported NumPy data type"):
        dtype_mapper("invalid_dtype", "numpy")

    # Test that identical types are returned unchanged
    assert dtype_mapper(np.float32, "numpy") is np.float32
    assert dtype_mapper("float32", "raster") == "float32"


# %% Test - dtype_mapper

# test_dtype_mapper()


# %% test functions : pixel_spec_apply


class TestPixelSpecApply:
    """Test class for pixel_spec_apply function"""

    @staticmethod
    def simple_spectral_function(spectrum: np.ndarray) -> np.ndarray:
        """Simple test spectral function that doubles the values"""
        return spectrum * 2

    @staticmethod
    def spectral_function_with_different_output(spectrum: np.ndarray) -> np.ndarray:
        """Spectral function that returns different number of bands"""
        result: np.ndarray = np.array([np.sum(spectrum), np.mean(spectrum), np.max(spectrum)])
        return result

    @staticmethod
    def test_basic_functionality() -> None:
        """Test basic functionality with a simple spectral function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            input_path = os.path.join(temp_dir, "input.tif")
            output_path = os.path.join(temp_dir, "output.tif")

            # Create test raster
            create_test_raster(input_path, 8, 8, 10)

            # Apply function
            pixel_spec_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelSpecApply.simple_spectral_function,
                tile_size=4,
                progress=False,
            )

            # Verify output exists
            assert os.path.exists(output_path)

            # Verify output content
            with rasterio.open(input_path) as src_input, rasterio.open(output_path) as src_output:
                # Check metadata
                assert src_output.count == src_input.count  # Same number of bands
                assert src_output.width == src_input.width
                assert src_output.height == src_input.height

                # Check data content
                input_data = src_input.read()
                output_data = src_output.read()

                # Verify the spectral function was applied correctly
                expected_data = input_data * 2
                np.testing.assert_array_almost_equal(output_data, expected_data)

    @staticmethod
    def test_different_output_bands() -> None:
        """Test with spectral function that returns different number of bands"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.tif")
            output_path = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, 9, 9, 4)

            pixel_spec_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelSpecApply.spectral_function_with_different_output,
                tile_size=3,
                progress=False,
            )

            assert os.path.exists(output_path)

            with rasterio.open(output_path) as src:
                assert src.count == 3  # Should have 3 output bands

    @staticmethod
    def test_different_dtypes() -> None:
        """Test with different output dtypes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.tif")
            output_path = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, 4, 4, 2, dtype="uint8")

            pixel_spec_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelSpecApply.simple_spectral_function,
                dtype="float64",
                tile_size=2,
                progress=False,
            )

            assert os.path.exists(output_path)

            with rasterio.open(output_path) as src:
                assert src.dtypes[0] == "float64"

    @staticmethod
    def test_default_tile_size() -> None:
        """Test with default tile size (-1)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.tif")
            output_path = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, 10, 10, 4)

            pixel_spec_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelSpecApply.simple_spectral_function,
                tile_size=-1,  # Should use default
                progress=False,
            )

            assert os.path.exists(output_path)

    @staticmethod
    def test_invalid_tile_size() -> None:
        """Test with invalid tile size"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.tif")
            output_path = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, 4, 4, 2)

            with pytest.raises(ValueError, match="If provided, tile_size must be positive"):
                pixel_spec_apply(
                    image_path=input_path,
                    output_path=output_path,
                    spectral_function=TestPixelSpecApply.simple_spectral_function,
                    tile_size=0,  # Invalid
                    progress=False,
                )

    @staticmethod
    def test_output_file_overwrite() -> None:
        """Test that existing output file is overwritten"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.tif")
            output_path = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, 4, 4, 2)

            # Create output file first
            with open(output_path, "w") as f:
                f.write("test content")

            assert os.path.exists(output_path)

            # Should overwrite without error
            pixel_spec_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelSpecApply.simple_spectral_function,
                tile_size=2,
                progress=False,
            )

            # Should be a valid raster file now
            assert os.path.exists(output_path)
            with rasterio.open(output_path) as src:
                assert src.count == 2

    @staticmethod
    def test_edge_case_tile_sizes() -> None:
        """Test with tile sizes that don't evenly divide image dimensions"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.tif")
            output_path = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, 7, 5, 3)  # Odd dimensions

            pixel_spec_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelSpecApply.simple_spectral_function,
                tile_size=3,  # Doesn't divide evenly
                progress=False,
            )

            assert os.path.exists(output_path)

            # Verify all pixels were processed
            with rasterio.open(input_path) as src_input, rasterio.open(output_path) as src_output:
                input_data = src_input.read()
                output_data = src_output.read()

                expected_data = input_data * 2
                np.testing.assert_array_almost_equal(output_data, expected_data)

    @staticmethod
    def test_spectral_function_validation() -> None:
        """Test that spectral function is properly validated"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.tif")
            output_path = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, 2, 2, 2)

            # Test with invalid function (wrong signature)
            def invalid_function(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
                return x + y

            with pytest.raises((TypeError, ValueError)):
                pixel_spec_apply(
                    image_path=input_path,
                    output_path=output_path,
                    spectral_function=invalid_function,
                    tile_size=2,
                    progress=False,
                )

    @staticmethod
    def test_single_pixel_image() -> None:
        """Test with single pixel image"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.tif")
            output_path = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, 1, 1, 4)

            pixel_spec_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelSpecApply.simple_spectral_function,
                tile_size=1,
                progress=False,
            )

            assert os.path.exists(output_path)

            with rasterio.open(input_path) as src_input, rasterio.open(output_path) as src_output:
                input_data = src_input.read()
                output_data = src_output.read()

                expected_data = input_data * 2
                np.testing.assert_array_almost_equal(output_data, expected_data)


# %% Test - pixel_spec_apply

# TestPixelSpecApply.test_basic_functionality()
# TestPixelSpecApply.test_different_output_bands()
# TestPixelSpecApply.test_different_dtypes()
# TestPixelSpecApply.test_default_tile_size()
# TestPixelSpecApply.test_invalid_tile_size()
# TestPixelSpecApply.test_output_file_overwrite()
# TestPixelSpecApply.test_edge_case_tile_sizes()
# TestPixelSpecApply.test_spectral_function_validation()
# TestPixelSpecApply.test_single_pixel_image()


# %% test functions : pixel_array_apply


class TestPixelArrayApply:
    """Test class for pixel_array_apply function"""

    @staticmethod
    def simple_spectral_function(spectra_2d: np.ndarray) -> np.ndarray:
        """Simple test function that doubles the values"""
        return spectra_2d * 2

    @staticmethod
    def band_reduction_function(spectra_2d: np.ndarray) -> np.ndarray:
        """Test function that reduces bands (4 -> 2)"""
        return spectra_2d[:, :2]  # Keep only first 2 bands

    @staticmethod
    def band_expansion_function(spectra_2d: np.ndarray) -> np.ndarray:
        """Test function that expands bands (4 -> 6)"""
        result: np.ndarray = np.column_stack([spectra_2d, spectra_2d[:, :2]])
        return result  # Add first two bands as fifth and sixth

    @staticmethod
    def test_basic_functionality() -> None:
        """Test basic functionality with simple spectral function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            input_path: str = os.path.join(temp_dir, "input.tif")
            output_path: str = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, width=8, height=8, bands=4)

            # Apply function
            pixel_array_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelArrayApply.simple_spectral_function,
                dtype="float32",
                tile_size=4,
                progress=False,
            )

            # Verify output exists
            assert os.path.exists(output_path)

            # Verify output content
            with rasterio.open(output_path) as dst:
                assert dst.count == 4  # Same number of bands
                assert dst.meta["dtype"] == "float32"

                # Read and verify data was doubled
                output_data: np.ndarray = dst.read()
                with rasterio.open(input_path) as src:
                    input_data: np.ndarray = src.read()

                np.testing.assert_allclose(output_data, input_data * 2, rtol=1e-6)

    @staticmethod
    def test_band_reduction() -> None:
        """Test function that reduces number of bands"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path: str = os.path.join(temp_dir, "input.tif")
            output_path: str = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, width=6, height=6, bands=4)

            pixel_array_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelArrayApply.band_reduction_function,
                tile_size=3,
                progress=False,
            )

            with rasterio.open(output_path) as dst:
                assert dst.count == 2  # Reduced from 4 to 2 bands

    @staticmethod
    def test_band_expansion() -> None:
        """Test function that increases number of bands"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path: str = os.path.join(temp_dir, "input.tif")
            output_path: str = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, width=5, height=5, bands=4)

            pixel_array_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelArrayApply.band_expansion_function,
                tile_size=2,
                progress=False,
            )

            with rasterio.open(output_path) as dst:
                assert dst.count == 6  # Expanded from 4 to 6 bands

    @staticmethod
    def test_different_dtypes() -> None:
        """Test with different output dtypes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path: str = os.path.join(temp_dir, "input.tif")
            output_path: str = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, width=4, height=4, bands=4)

            pixel_array_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelArrayApply.simple_spectral_function,
                dtype="uint16",
                tile_size=2,
                progress=False,
            )

            with rasterio.open(output_path) as dst:
                assert dst.meta["dtype"] == "uint16"

    @staticmethod
    def test_default_tile_size() -> None:
        """Test with default tile size (-1)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path: str = os.path.join(temp_dir, "input.tif")
            output_path: str = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, width=10, height=10, bands=4)

            pixel_array_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelArrayApply.simple_spectral_function,
                progress=False,
            )

            assert os.path.exists(output_path)

    @staticmethod
    def test_small_tile_size() -> None:
        """Test with small tile size"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path: str = os.path.join(temp_dir, "input.tif")
            output_path: str = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, width=7, height=7, bands=4)

            pixel_array_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelArrayApply.simple_spectral_function,
                tile_size=1,
                progress=False,
            )

            assert os.path.exists(output_path)

    @staticmethod
    def test_large_tile_size() -> None:
        """Test with tile size larger than image"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path: str = os.path.join(temp_dir, "input.tif")
            output_path: str = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, width=5, height=5, bands=4)

            pixel_array_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelArrayApply.simple_spectral_function,
                tile_size=10,
                progress=False,
            )

            assert os.path.exists(output_path)

    @staticmethod
    def test_invalid_tile_size() -> None:
        """Test with invalid tile size"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path: str = os.path.join(temp_dir, "input.tif")
            output_path: str = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, width=5, height=5, bands=4)

            with pytest.raises(ValueError, match="tile_size must be positive"):
                pixel_array_apply(
                    image_path=input_path,
                    output_path=output_path,
                    spectral_function=TestPixelArrayApply.simple_spectral_function,
                    tile_size=0,
                    progress=False,
                )

    @staticmethod
    def test_output_file_overwrite() -> None:
        """Test that existing output file is overwritten"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path: str = os.path.join(temp_dir, "input.tif")
            output_path: str = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, width=4, height=4, bands=4)

            # Create a dummy output file first
            with open(output_path, "w") as f:
                f.write("dummy content")

            pixel_array_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelArrayApply.simple_spectral_function,
                tile_size=2,
                progress=False,
            )

            # Should be a valid raster file, not text
            with rasterio.open(output_path) as dst:
                assert dst.count == 4

    @staticmethod
    def test_complex_spectral_function() -> None:
        """Test with a more complex spectral function"""

        def complex_function(spectra_2d: np.ndarray) -> np.ndarray:
            """Normalize each spectrum and compute mean"""
            normalized: np.ndarray = spectra_2d / np.linalg.norm(spectra_2d, axis=1, keepdims=True)
            mean_band: np.ndarray = np.mean(normalized, axis=1, keepdims=True)
            result: np.ndarray = np.column_stack([normalized, mean_band])
            return result

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path: str = os.path.join(temp_dir, "input.tif")
            output_path: str = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, width=4, height=4, bands=4)

            pixel_array_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=complex_function,
                tile_size=2,
                progress=False,
            )

            with rasterio.open(output_path) as dst:
                assert dst.count == 5  # 4 normalized bands + 1 mean band

    @staticmethod
    def test_single_pixel_image() -> None:
        """Test with a single pixel image"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path: str = os.path.join(temp_dir, "input.tif")
            output_path: str = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, width=1, height=1, bands=4)

            pixel_array_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelArrayApply.simple_spectral_function,
                tile_size=1,
                progress=False,
            )

            with rasterio.open(output_path) as dst:
                assert dst.count == 4
                assert dst.width == 1
                assert dst.height == 1

    @staticmethod
    def test_function_signature_validation() -> None:
        """Test that the function validates spectral_function signature"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path: str = os.path.join(temp_dir, "input.tif")
            output_path: str = os.path.join(temp_dir, "output.tif")

            create_test_raster(input_path, width=3, height=3, bands=4)

            # Function with wrong signature (no parameters)
            def invalid_function() -> np.ndarray:
                result: np.ndarray = np.array([[1, 2]])
                return result

            # This should raise an error when the function is called
            with pytest.raises((TypeError, ValueError)):
                pixel_array_apply(
                    image_path=input_path,
                    output_path=output_path,
                    spectral_function=invalid_function,
                    tile_size=2,
                    progress=False,
                )


# %% Test - pixel_array_apply

# TestPixelArrayApply.test_basic_functionality()

# TestPixelArrayApply.test_band_reduction()
# TestPixelArrayApply.test_band_expansion()
# TestPixelArrayApply.test_complex_spectral_function()

# TestPixelArrayApply.test_different_dtypes()

# TestPixelArrayApply.test_default_tile_size()
# TestPixelArrayApply.test_small_tile_size()
# TestPixelArrayApply.test_large_tile_size()
# TestPixelArrayApply.test_invalid_tile_size()

# TestPixelArrayApply.test_single_pixel_image()

# TestPixelArrayApply.test_output_file_overwrite()
# TestPixelArrayApply.test_function_signature_validation()


# %% test functions : pixel_tensor_apply


@pytest.mark.skipif(not HAS_CUDA, reason="No GPU available")
class TestPixelTensorApply(unittest.TestCase):
    @staticmethod
    def simple_spectral_function(tensor: torch.Tensor) -> torch.Tensor:
        """Simple test function that doubles the values along axis 0."""
        return tensor * 2

    @staticmethod
    def band_reduction_function(tensor: torch.Tensor) -> torch.Tensor:
        """Test function that reduces bands by taking mean of every 2 bands."""
        result = []
        for i in range(0, tensor.shape[0], 2):
            if i + 1 < tensor.shape[0]:
                result.append((tensor[i] + tensor[i + 1]) / 2)
            else:
                result.append(tensor[i])
        return torch.stack(result)

    @staticmethod
    def identity_function(tensor: torch.Tensor) -> torch.Tensor:
        """Identity function for testing."""
        return tensor.clone()

    @staticmethod
    def spectral_normalization_function(tensor: torch.Tensor) -> torch.Tensor:
        """Test function that normalizes each band to [0, 1] range."""
        result = []
        for i in range(tensor.shape[0]):
            band = tensor[i]
            min_val = band.min()
            max_val = band.max()
            if max_val > min_val:
                normalized = (band - min_val) / (max_val - min_val)
            else:
                normalized = band
            result.append(normalized)
        return torch.stack(result)

    @staticmethod
    def test_basic_functionality() -> None:
        """Test basic functionality with a simple spectral function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")

            create_test_raster(input_path, width=10, height=10, bands=3)

            pixel_tensor_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelTensorApply.simple_spectral_function,
                dtype="float32",
                tile_size=5,
                device="cuda",
                progress=False,
            )

            assert os.path.exists(output_path)

            with rasterio.open(output_path) as src:
                output_data = src.read()
                with rasterio.open(input_path) as src_input:
                    input_data = src_input.read().astype(np.float32)

                np.testing.assert_allclose(output_data, input_data * 2, rtol=1e-6)

    @staticmethod
    def test_band_reduction() -> None:
        """Test with a function that changes the number of bands."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")

            create_test_raster(input_path, width=8, height=8, bands=4)

            pixel_tensor_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelTensorApply.band_reduction_function,
                dtype="float32",
                tile_size=4,
                device="cuda",
                progress=False,
            )

            assert os.path.exists(output_path)

            with rasterio.open(output_path) as src:
                output_data = src.read()
                assert output_data.shape[0] == 2

    @staticmethod
    def test_identity() -> None:
        """Test identity function preserves input exactly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")

            create_test_raster(input_path, width=6, height=6, bands=2)

            pixel_tensor_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelTensorApply.identity_function,
                dtype="uint16",
                tile_size=3,
                device="cuda",
                progress=False,
            )

            assert os.path.exists(output_path)

            with rasterio.open(input_path) as src_input:
                with rasterio.open(output_path) as src_output:
                    input_data = src_input.read()
                    output_data = src_output.read()

                    np.testing.assert_array_equal(input_data, output_data)

    @staticmethod
    def test_different_dtypes() -> None:
        """Test with different output data types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path_float32 = os.path.join(temp_dir, "test_output_float32.tif")
            output_path_float64 = os.path.join(temp_dir, "test_output_float64.tif")

            create_test_raster(input_path, width=5, height=5, bands=2)

            pixel_tensor_apply(
                image_path=input_path,
                output_path=output_path_float32,
                spectral_function=TestPixelTensorApply.simple_spectral_function,
                dtype="float32",
                tile_size=3,
                device="cuda",
                progress=False,
            )

            pixel_tensor_apply(
                image_path=input_path,
                output_path=output_path_float64,
                spectral_function=TestPixelTensorApply.simple_spectral_function,
                dtype="float64",
                tile_size=3,
                device="cuda",
                progress=False,
            )

            with rasterio.open(output_path_float32) as src:
                assert src.meta["dtype"] == "float32"

            with rasterio.open(output_path_float64) as src:
                assert src.meta["dtype"] == "float64"

    @staticmethod
    def test_different_tile_sizes() -> None:
        """Test with different tile sizes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path1 = os.path.join(temp_dir, "test_output1.tif")
            output_path2 = os.path.join(temp_dir, "test_output2.tif")

            create_test_raster(input_path, width=12, height=12, bands=3)

            pixel_tensor_apply(
                image_path=input_path,
                output_path=output_path1,
                spectral_function=TestPixelTensorApply.simple_spectral_function,
                dtype="float32",
                tile_size=4,
                device="cuda",
                progress=False,
            )

            pixel_tensor_apply(
                image_path=input_path,
                output_path=output_path2,
                spectral_function=TestPixelTensorApply.simple_spectral_function,
                dtype="float32",
                tile_size=6,
                device="cuda",
                progress=False,
            )

            with rasterio.open(output_path1) as src1:
                with rasterio.open(output_path2) as src2:
                    data1 = src1.read()
                    data2 = src2.read()
                    np.testing.assert_allclose(data1, data2, rtol=1e-6)

    @staticmethod
    def test_invalid_tile_size() -> None:
        """Test that invalid tile size raises ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")

            create_test_raster(input_path, width=10, height=10, bands=2)

            try:
                pixel_tensor_apply(
                    image_path=input_path,
                    output_path=output_path,
                    spectral_function=TestPixelTensorApply.identity_function,
                    dtype="float32",
                    tile_size=0,
                    device="cuda",
                    progress=False,
                )
                raise AssertionError("Expected ValueError")
            except ValueError:
                pass

    @staticmethod
    def test_output_overwrite() -> None:
        """Test that existing output file is overwritten."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")

            create_test_raster(input_path, width=5, height=5, bands=2)

            with open(output_path, "w") as f:
                f.write("dummy content")

            pixel_tensor_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelTensorApply.identity_function,
                dtype="float32",
                tile_size=3,
                device="cuda",
                progress=False,
            )

            with rasterio.open(output_path) as src:
                assert src.count == 2

    @staticmethod
    def test_edge_case_small_image() -> None:
        """Test with very small image size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")

            create_test_raster(input_path, width=2, height=2, bands=1)

            pixel_tensor_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelTensorApply.simple_spectral_function,
                dtype="float32",
                tile_size=64,
                device="cuda",
                progress=False,
            )

            assert os.path.exists(output_path)

            with rasterio.open(input_path) as src_input:
                with rasterio.open(output_path) as src_output:
                    input_data = src_input.read().astype(np.float32)
                    output_data = src_output.read()

                    np.testing.assert_allclose(output_data, input_data * 2, rtol=1e-6)

    @staticmethod
    def test_spectral_normalization() -> None:
        """Test with a spectral normalization function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")

            create_test_raster(input_path, width=8, height=8, bands=3)

            pixel_tensor_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelTensorApply.spectral_normalization_function,
                dtype="float32",
                tile_size=4,
                device="cuda",
                progress=False,
            )

            assert os.path.exists(output_path)

            with rasterio.open(output_path) as src:
                output_data = src.read()
                assert output_data.min() >= 0.0
                assert output_data.max() <= 1.0 + 1e-6

    @staticmethod
    def test_default_tile_size() -> None:
        """Test with default tile size (-1)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")

            create_test_raster(input_path, width=10, height=10, bands=2)

            pixel_tensor_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelTensorApply.identity_function,
                dtype="float32",
                tile_size=-1,
                device="cuda",
                progress=False,
            )

            assert os.path.exists(output_path)

            with rasterio.open(input_path) as src_input:
                with rasterio.open(output_path) as src_output:
                    input_data = src_input.read().astype(np.float32)
                    output_data = src_output.read()

                    np.testing.assert_allclose(input_data, output_data, rtol=1e-6)

    @staticmethod
    def test_cpu_device() -> None:
        """Test with CPU device explicitly specified."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")

            create_test_raster(input_path, width=5, height=5, bands=2)

            pixel_tensor_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelTensorApply.simple_spectral_function,
                dtype="float32",
                tile_size=3,
                device="cpu",
                progress=False,
            )

            assert os.path.exists(output_path)

            with rasterio.open(input_path) as src_input:
                with rasterio.open(output_path) as src_output:
                    input_data = src_input.read().astype(np.float32)
                    output_data = src_output.read()

                    np.testing.assert_allclose(output_data, input_data * 2, rtol=1e-6)

    @staticmethod
    def test_large_tile_size() -> None:
        """Test with tile size larger than image dimensions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")

            create_test_raster(input_path, width=3, height=3, bands=2)

            pixel_tensor_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelTensorApply.simple_spectral_function,
                dtype="float32",
                tile_size=10,
                device="cuda",
                progress=False,
            )

            assert os.path.exists(output_path)

            with rasterio.open(input_path) as src_input:
                with rasterio.open(output_path) as src_output:
                    input_data = src_input.read().astype(np.float32)
                    output_data = src_output.read()

                    np.testing.assert_allclose(output_data, input_data * 2, rtol=1e-6)


# %% Test - pixel_tensor_apply

# TestPixelTensorApply.test_basic_functionality()

# TestPixelTensorApply.test_band_reduction()
# TestPixelTensorApply.test_identity()
# TestPixelTensorApply.test_spectral_normalization()

# TestPixelTensorApply.test_different_dtypes()

# TestPixelTensorApply.test_default_tile_size()
# TestPixelTensorApply.test_different_tile_sizes()
# TestPixelTensorApply.test_invalid_tile_size()
# TestPixelTensorApply.test_large_tile_size()

# TestPixelTensorApply.test_edge_case_small_image()

# TestPixelTensorApply.test_cpu_device()

# TestPixelTensorApply.test_output_overwrite()


# %% test functions : pixel_tensor_hyper_apply


@pytest.mark.skipif(not HAS_CUDA, reason="No GPU available")
class TestPixelTensorHyperApply(unittest.TestCase):
    @staticmethod
    def simple_spectral_function(tensor: torch.Tensor) -> torch.Tensor:
        """Simple spectral function for normalization and scaling"""
        # Simple function: normalize and scale
        tensor = tensor.float()
        mean = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, keepdim=True)
        return ((tensor - mean) / (std + 1e-8)) * 100

    @staticmethod
    def band_reduction_function(tensor: torch.Tensor) -> torch.Tensor:
        """Function that reduces bands"""
        tensor = tensor.float()
        # Simple mean/std normalization
        normalized = (tensor - tensor.mean(dim=1, keepdim=True)) / (tensor.std(dim=1, keepdim=True) + 1e-8)
        # Return first 2 "components"
        return normalized[:, :2, :]

    @staticmethod
    def identity_function(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clone()

    @staticmethod
    def test_basic_functionality_cpu() -> None:
        """Test basic functionality with identity function on CPU."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test raster
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")
            create_test_raster(input_path, width=10, height=5, bands=3, dtype="uint16")
            pixel_tensor_hyper_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelTensorHyperApply.identity_function,
                dtype="float32",
                tile_size=2,
                device="cpu",
                progress=False,
            )

            # Verify output exists and has correct properties
            assert os.path.exists(output_path)

            with rasterio.open(input_path) as src, rasterio.open(output_path) as dst:
                assert dst.count == src.count  # Same number of bands
                assert dst.width == src.width
                assert dst.height == src.height
                assert dst.meta["dtype"] == "float32"

    @staticmethod
    def test_basic_functionality_gpu() -> None:
        """Test basic functionality with identity function on GPU."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test raster
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")
            create_test_raster(input_path, width=10, height=5, bands=3, dtype="uint16")
            pixel_tensor_hyper_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelTensorHyperApply.identity_function,
                dtype="float32",
                tile_size=2,
                device="cuda",
                progress=False,
            )

            # Verify output exists and has correct properties
            assert os.path.exists(output_path)

            with rasterio.open(input_path) as src, rasterio.open(output_path) as dst:
                assert dst.count == src.count  # Same number of bands
                assert dst.width == src.width
                assert dst.height == src.height
                assert dst.meta["dtype"] == "float32"

    @staticmethod
    def test_band_reduction_function_gpu() -> None:
        """Test with a function that reduces number of bands on GPU."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")
            create_test_raster(input_path, width=8, height=4, bands=5, dtype="uint16")
            pixel_tensor_hyper_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelTensorHyperApply.band_reduction_function,
                dtype="float32",
                tile_size=3,
                device="cuda",
                progress=False,
            )

            with rasterio.open(output_path) as dst:
                assert dst.count == 2  # Reduced from 5 to 2 bands

    @staticmethod
    def test_different_dtypes_gpu() -> None:
        """Test with different output data types on GPU."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            # Test float32
            output_path_float32 = os.path.join(temp_dir, "test_output_float32.tif")
            create_test_raster(input_path, width=6, height=3, bands=2, dtype="uint16")
            pixel_tensor_hyper_apply(
                image_path=input_path,
                output_path=output_path_float32,
                spectral_function=TestPixelTensorHyperApply.simple_spectral_function,
                dtype="float32",
                tile_size=1,
                device="cuda",
                progress=False,
            )
            with rasterio.open(output_path_float32) as dst:
                assert dst.meta["dtype"] == "float32"

            # Test float64
            output_path_float64 = os.path.join(temp_dir, "test_output_float64.tif")
            create_test_raster(input_path, width=6, height=3, bands=2, dtype="uint16")
            pixel_tensor_hyper_apply(
                image_path=input_path,
                output_path=output_path_float64,
                spectral_function=TestPixelTensorHyperApply.simple_spectral_function,
                dtype="float64",
                tile_size=1,
                device="cuda",
                progress=False,
            )
            with rasterio.open(output_path_float64) as dst:
                assert dst.meta["dtype"] == "float64"

    @staticmethod
    def test_different_tile_sizes_gpu() -> None:
        """Test with different tile sizes on GPU."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")

            # Test tile_size = 1
            output_path_1 = os.path.join(temp_dir, "test_output_1.tif")
            create_test_raster(input_path, width=10, height=10, bands=3, dtype="uint16")
            pixel_tensor_hyper_apply(
                image_path=input_path,
                output_path=output_path_1,
                spectral_function=TestPixelTensorHyperApply.identity_function,
                dtype="float32",
                tile_size=1,
                device="cuda",
                progress=False,
            )
            assert os.path.exists(output_path_1)

            # Test tile_size = 2
            output_path_2 = os.path.join(temp_dir, "test_output_2.tif")
            create_test_raster(input_path, width=10, height=10, bands=3, dtype="uint16")
            pixel_tensor_hyper_apply(
                image_path=input_path,
                output_path=output_path_2,
                spectral_function=TestPixelTensorHyperApply.identity_function,
                dtype="float32",
                tile_size=2,
                device="cuda",
                progress=False,
            )
            assert os.path.exists(output_path_2)

            # Test tile_size = 5
            output_path_5 = os.path.join(temp_dir, "test_output_5.tif")
            create_test_raster(input_path, width=10, height=10, bands=3, dtype="uint16")
            pixel_tensor_hyper_apply(
                image_path=input_path,
                output_path=output_path_5,
                spectral_function=TestPixelTensorHyperApply.identity_function,
                dtype="float32",
                tile_size=5,
                device="cuda",
                progress=False,
            )
            assert os.path.exists(output_path_5)

            # Test tile_size = 10
            output_path_10 = os.path.join(temp_dir, "test_output_10.tif")
            create_test_raster(input_path, width=10, height=10, bands=3, dtype="uint16")
            pixel_tensor_hyper_apply(
                image_path=input_path,
                output_path=output_path_10,
                spectral_function=TestPixelTensorHyperApply.identity_function,
                dtype="float32",
                tile_size=10,
                device="cuda",
                progress=False,
            )
            assert os.path.exists(output_path_10)

    @staticmethod
    def test_default_tile_size_gpu() -> None:
        """Test with default tile size (-1) on GPU."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")
            create_test_raster(input_path, width=5, height=5, bands=2, dtype="uint16")
            pixel_tensor_hyper_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelTensorHyperApply.identity_function,
                dtype="float32",
                tile_size=-1,  # Default
                device="cuda",
                progress=False,
            )

            assert os.path.exists(output_path)

    @staticmethod
    def test_invalid_tile_size_gpu() -> None:
        """Test that invalid tile size raises ValueError on GPU."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")
            create_test_raster(input_path, width=5, height=5, bands=2, dtype="uint16")
            with pytest.raises(ValueError):
                pixel_tensor_hyper_apply(
                    image_path=input_path,
                    output_path=output_path,
                    spectral_function=TestPixelTensorHyperApply.identity_function,
                    dtype="float32",
                    tile_size=0,  # Invalid
                    device="cuda",
                    progress=False,
                )

    @staticmethod
    def test_output_overwrite_gpu() -> None:
        """Test that existing output file is overwritten on GPU."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")
            create_test_raster(input_path, width=3, height=2, bands=1, dtype="uint16")

            # Create a dummy output file first
            with open(output_path, "w") as f:
                f.write("dummy content")

            pixel_tensor_hyper_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelTensorHyperApply.identity_function,
                dtype="float32",
                tile_size=1,
                device="cuda",
                progress=False,
            )

            # Verify the file was overwritten with valid raster data
            assert os.path.exists(output_path)
            with rasterio.open(output_path) as dst:
                assert dst.count == 1

    @staticmethod
    def test_small_raster_edge_case_gpu() -> None:
        """Test with very small raster dimensions on GPU."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")

            # Create a very small raster
            create_test_raster(input_path, width=1, height=1, bands=1, dtype="uint16")
            pixel_tensor_hyper_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelTensorHyperApply.identity_function,
                dtype="float32",
                tile_size=1,
                device="cuda",
                progress=False,
            )

            # Verify output
            with rasterio.open(output_path) as dst:
                assert dst.width == 1
                assert dst.height == 1
                assert dst.count == 1

    @staticmethod
    def test_large_tile_size_gpu() -> None:
        """Test with tile size larger than image height on GPU."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")

            create_test_raster(input_path, width=3, height=2, bands=2, dtype="uint16")

            pixel_tensor_hyper_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelTensorHyperApply.identity_function,
                dtype="float32",
                tile_size=10,  # Larger than image height
                device="cuda",
                progress=False,
            )

            assert os.path.exists(output_path)

    @staticmethod
    def test_complex_spectral_function_gpu() -> None:
        """Test with a more complex spectral function on GPU."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "test_input.tif")
            output_path = os.path.join(temp_dir, "test_output.tif")

            create_test_raster(input_path, width=8, height=6, bands=4, dtype="uint16")

            pixel_tensor_hyper_apply(
                image_path=input_path,
                output_path=output_path,
                spectral_function=TestPixelTensorHyperApply.simple_spectral_function,
                dtype="float32",
                tile_size=3,
                device="cuda",
                progress=False,
            )

            # Verify output has correct shape and properties
            with rasterio.open(output_path) as dst:
                assert dst.count == 4  # Same number of bands
                assert dst.width == 8
                assert dst.height == 6
                assert dst.meta["dtype"] == "float32"


# %% Test - pixel_tensor_hyper_apply

# TestPixelTensorHyperApply.test_basic_functionality_cpu()
# TestPixelTensorHyperApply.test_basic_functionality_gpu()
# TestPixelTensorHyperApply.test_band_reduction_function_gpu()
# TestPixelTensorHyperApply.test_complex_spectral_function_gpu()

# TestPixelTensorHyperApply.test_different_dtypes_gpu()

# TestPixelTensorHyperApply.test_different_tile_sizes_gpu()
# TestPixelTensorHyperApply.test_default_tile_size_gpu()
# TestPixelTensorHyperApply.test_invalid_tile_size_gpu()
# TestPixelTensorHyperApply.test_large_tile_size_gpu()

# TestPixelTensorHyperApply.test_output_overwrite_gpu()

# TestPixelTensorHyperApply.test_small_raster_edge_case_gpu()


# %% test functions : pixel_apply


class TestPixelApply:
    @staticmethod
    def dummy_spectral_function_spec(spectrum: np.ndarray) -> np.ndarray:
        """Dummy spectral function for 'spec' type."""
        return spectrum * 2

    @staticmethod
    def dummy_spectral_function_array(spectra: np.ndarray) -> np.ndarray:
        """Dummy spectral function for 'array' type."""
        return spectra * 2

    @staticmethod
    def dummy_spectral_function_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Dummy spectral function for 'tensor' type."""
        return tensor * 2

    @staticmethod
    def test_invalid_input_path() -> None:
        """Test handling of non-existent input file."""
        with pytest.raises(ValueError, match="Image file does not exist"):
            pixel_apply(
                image_path="non_existent.tif",
                spectral_function=TestPixelApply.dummy_spectral_function_spec,
                function_type="spec",
            )

    @staticmethod
    def test_invalid_output_dir() -> None:
        """Test handling of non-existent output directory."""
        # Create a temporary input file
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            create_test_raster(tmp.name, 10, 10, 10)
            input_path = tmp.name

        try:
            with pytest.raises(ValueError, match="Directory does not exist"):
                pixel_apply(
                    image_path=input_path,
                    spectral_function=TestPixelApply.dummy_spectral_function_spec,
                    function_type="spec",
                    output_path="/invalid/path/output.tif",
                )
        finally:
            os.remove(input_path)

    @staticmethod
    def test_invalid_tile_size() -> None:
        """Test handling of invalid tile size."""
        # Create temporary input and output paths
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_input:
            create_test_raster(tmp_input.name, 10, 10, 10)
            input_path = tmp_input.name

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.tif")

            try:
                with pytest.raises(ValueError, match="tile_size must be positive integer"):
                    pixel_apply(
                        image_path=input_path,
                        spectral_function=TestPixelApply.dummy_spectral_function_spec,
                        function_type="spec",
                        output_path=output_path,
                        tile_size=0,
                    )
            finally:
                os.remove(input_path)

    @staticmethod
    def test_default_output_path() -> None:
        """Test generation of default output path."""
        # Create a temporary input file
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            create_test_raster(tmp.name, 10, 10, 10)
            input_path = tmp.name

        try:
            result = pixel_apply(
                image_path=input_path,
                spectral_function=TestPixelApply.dummy_spectral_function_spec,
                function_type="spec",
                return_output_path=True,
                progress=False,
            )
            assert result.endswith("_px_app_dummy_spectral_function_spec.tif")
            assert os.path.exists(result)

        finally:
            os.remove(input_path)
            if result is not None:
                os.remove(result)

    @staticmethod
    def test_spec_function_type() -> None:
        """Test processing with 'spec' function type."""
        # Create temporary input and output paths
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_input:
            create_test_raster(tmp_input.name, 10, 10, 10)
            input_path = tmp_input.name

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.tif")

            try:
                result = pixel_apply(
                    image_path=input_path,
                    spectral_function=TestPixelApply.dummy_spectral_function_spec,
                    function_type="spec",
                    output_path=output_path,
                    progress=False,
                )
                assert result == output_path

            finally:
                os.remove(input_path)
                if result is not None:
                    os.remove(result)

    @staticmethod
    def test_array_function_type() -> None:
        """Test processing with 'array' function type."""
        # Create temporary input and output paths
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_input:
            create_test_raster(tmp_input.name, 10, 10, 10)
            input_path = tmp_input.name

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.tif")

            try:
                result = pixel_apply(
                    image_path=input_path,
                    spectral_function=TestPixelApply.dummy_spectral_function_array,
                    function_type="array",
                    output_path=output_path,
                    progress=False,
                )
                assert result == output_path

            finally:
                os.remove(input_path)

    @staticmethod
    @pytest.mark.skipif(not HAS_CUDA, reason="No GPU available")
    def test_tensor_function_type() -> None:
        """Test processing with 'tensor' function type."""
        # Create temporary input and output paths
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_input:
            create_test_raster(tmp_input.name, 10, 10, 10)
            input_path = tmp_input.name

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.tif")

            try:
                result = pixel_apply(
                    image_path=input_path,
                    spectral_function=TestPixelApply.dummy_spectral_function_tensor,
                    function_type="tensor",
                    output_path=output_path,
                    progress=False,
                )
                assert result == output_path

            finally:
                os.remove(input_path)

    @staticmethod
    @pytest.mark.skipif(not HAS_CUDA, reason="No GPU available")
    def test_tensor_hyper_function_type() -> None:
        """Test processing with 'tensor_hyper' function type."""
        # Create temporary input and output paths
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_input:
            create_test_raster(tmp_input.name, 10, 10, 10)
            input_path = tmp_input.name

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.tif")

            try:
                result = pixel_apply(
                    image_path=input_path,
                    spectral_function=TestPixelApply.dummy_spectral_function_tensor,
                    function_type="tensor_hyper",
                    output_path=output_path,
                    progress=False,
                )
                assert result == output_path

            finally:
                os.remove(input_path)

    @staticmethod
    def test_invalid_function_type() -> None:
        """Test handling of invalid function type."""
        # Create temporary input and output paths
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_input:
            create_test_raster(tmp_input.name, 10, 10, 10)
            input_path = tmp_input.name

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.tif")

            try:
                with pytest.raises(ValueError, match="Invalid function_type"):
                    pixel_apply(
                        image_path=input_path,
                        spectral_function=TestPixelApply.dummy_spectral_function_spec,
                        function_type="invalid",
                        output_path=output_path,
                    )
            finally:
                os.remove(input_path)

    @staticmethod
    def test_no_return_output_path() -> None:
        """Test behavior when return_output_path is False."""
        # Create temporary input and output paths
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_input:
            create_test_raster(tmp_input.name, 10, 10, 10)
            input_path = tmp_input.name

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.tif")

            try:
                result = pixel_apply(
                    image_path=input_path,
                    spectral_function=TestPixelApply.dummy_spectral_function_spec,
                    function_type="spec",
                    output_path=output_path,
                    return_output_path=False,
                    progress=False,
                )
                assert result is None

            finally:
                os.remove(input_path)
                if result is not None:
                    os.remove(result)


# %% Test - pixel_apply

# TestPixelApply.test_default_output_path()
# TestPixelApply.test_invalid_input_path()
# TestPixelApply.test_invalid_output_dir()
# TestPixelApply.test_no_return_output_path()

# TestPixelApply.test_invalid_tile_size()

# TestPixelApply.test_spec_function_type()
# TestPixelApply.test_array_function_type()
# TestPixelApply.test_tensor_function_type()
# TestPixelApply.test_tensor_hyper_function_type()
# TestPixelApply.test_invalid_function_type()


# %% Test main

if __name__ == "__main__":
    unittest.main()
