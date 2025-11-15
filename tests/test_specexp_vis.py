# -*- coding: utf-8 -*-
"""
Tests for SpecExp data visualization functions

Copyright (c) 2025 Siwei Luo. MIT License.
"""
# ruff: noqa: I001
# OS
import os  # noqa: E402
import sys  # noqa: E402

# Test
import tempfile  # noqa: E402
import pytest  # noqa: E402

# Local
from specpipe.example_data import create_test_raster, create_test_spec_exp  # noqa: E402
from specpipe.specexp_vis import raster_rgb_preview  # noqa: E402


# %% Test raster visualization


def test_raster_rgb_preview() -> None:
    """Test functionality of 'raster_rgb_preview'."""
    with tempfile.TemporaryDirectory() as test_dir:
        # Create raster
        img_path = test_dir + "/test_img.tif"
        create_test_raster(raster_path=img_path, width=50, height=50, bands=4)
        # Create preview
        output_path = test_dir + "/test_img_rgb.png"
        raster_rgb_preview(
            raster_path=img_path,
            rgb_band_index=(1, 2, 3),
            display_size=(12, 9),
            output_path=output_path,
            show_image=False,
        )
        # Assert output
        assert os.path.exists(test_dir + "/test_img_rgb.png")


def test_exp_show_image() -> None:
    """Test functionality of 'show_image' method of 'SpecExp'."""
    with tempfile.TemporaryDirectory() as test_dir:
        # Create SpecExp
        exp = create_test_spec_exp(test_dir)
        # Show image
        output_path = test_dir + "/specexp_test_img_rgb.png"
        exp.show_image(
            exp.images[0][2],
            exp.images[0][1],
            rgb_band_index=(1, 2, 3),
            output_path=output_path,
            show_image=False,
        )
        # Assert output
        assert os.path.exists(output_path)


# %% Test main

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
