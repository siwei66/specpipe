# -*- coding: utf-8 -*-
"""
Example data generator for SpecPipe

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# OS Files
import os

# Typing
from typing import Optional, Union

# Testing third
import numpy as np
import pandas as pd

# Rasters
import rasterio

# Self
from .specexp import SpecExp
from .specio import roi_to_envi_xml, silent, simple_type_validator

# %% test helper functions


# Test helper functions : create_test_raster
@simple_type_validator
def create_test_raster(
    raster_path: str,
    width: int = 100,
    height: int = 50,
    bands: int = 4,
    incl_nodata: Union[None, int, float] = None,
    nodata_value: Union[int, float] = 0,
    dtype: Union[str, type] = "uint16",
    data: Optional[np.ndarray] = None,
) -> str:
    """Create a test raster file with by default 100x50 dimensions and 4 bands."""
    # Validate raster path
    raster_path = raster_path.replace("\\", "/").replace("//", "/")
    if (raster_path[-4:] != ".tif") and (raster_path[-5:] != ".tiff"):
        raise ValueError(f"raster_path must have .tif or .tiff extension, got: {raster_path}")
    if not os.path.exists(os.path.dirname(raster_path)):
        raise ValueError(f"raster path directory does not exist: {os.path.dirname(raster_path)}")

    # Validate raster data
    if data is None:
        # Mock image data
        np.random.seed(42)
        data = np.array(
            [
                [
                    (
                        np.array([1080 / bands * j + 360 / bands * int(i / 5)] * width)
                        + (np.random.rand(width) * 120 / bands)
                    )
                    .astype(dtype)
                    .tolist()
                    for i in range(height)
                ]
                for j in range(bands)
            ]
        )
    elif type(data) is np.ndarray:
        data = np.array(data).astype(dtype)
        if data.shape != (bands, height, width):
            raise ValueError(
                f"Given raster data shape {data.shape} does not match specified raster dimensions (bands={bands}, \
                    height={height}, width={width})."
            )
    else:
        raise ValueError(f"Given raster data must be numpy.ndarray, got type: {type(data)}.")

    # Set some nodata values
    if incl_nodata is not None:
        data[0, :, :] = incl_nodata
        if bands > 3:
            data[-1, :, :] = incl_nodata

    # Define transform (georeferencing)
    transform = rasterio.transform.from_bounds(0, 0, width, height, width=width, height=height)

    # Create metadata
    meta = {
        "driver": "GTiff",
        "dtype": dtype,
        "nodata": nodata_value,
        "width": width,
        "height": height,
        "count": bands,
        "crs": None,
        "transform": transform,
    }

    # Write to file
    with rasterio.open(raster_path, "w", **meta) as dst:
        dst.write(data)

    return raster_path


# Alias
create_example_raster = create_test_raster


# Test helper functions : create_test_roi_xml
@simple_type_validator
def create_test_roi_xml(
    xml_path: str,
    raster_width: int = 10,
    raster_height: int = 50,
    roi_count: int = 10,
    roi_list: Optional[list[dict]] = None,
    return_path: bool = True,
    return_roi_list: bool = False,
) -> Union[str, list[dict], tuple[str, list[dict]], None]:
    """Create a test ENVI ROI xml file with by default 10 ROIs."""
    # Create line of mock ROIs in the raster
    height_increment = int(raster_height / roi_count)
    if height_increment < 1:
        raise ValueError("raster_height / roi_count must be at least 1.")
    width_range = min(5, raster_width)

    if roi_list is None:
        roi_list = []
        # Test ROI data
        for i in range(roi_count):
            roi_dict = {
                "name": f"ROI_{i + 1}",
                "crs": "none",
                "color": None,
                "type": "polygon",
                "coordinates": [
                    [
                        (0.0, height_increment * i),
                        (width_range, height_increment * i),
                        (width_range, height_increment * (i + 1)),
                        (0.0, height_increment * (i + 1)),
                        (0.0, height_increment * i),
                    ]
                ],
            }
            roi_list.append(roi_dict)

    # Save ROI xml file
    path_out = roi_to_envi_xml(file_path=xml_path, roi_list=roi_list)
    path_out = str(path_out)

    # Output
    if return_path and not return_roi_list:
        return path_out
    elif not return_path and return_roi_list:
        return roi_list
    elif return_path and return_roi_list:
        return path_out, roi_list
    else:
        return None


# Alias
create_example_roi_xml = create_test_roi_xml


# Test helper functions : create_test_spec_exp
@silent
@simple_type_validator
def create_test_spec_exp(dir_path: str, sample_n: int = 10, n_bands: int = 4, is_regression: bool = True) -> SpecExp:
    """Create a standard test SpecExp instance."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    exp1: SpecExp = SpecExp(dir_path, log_loading=False)

    # # Add test
    # group
    exp1.add_groups(["test_group"])

    # Mock image data
    np.random.seed(42)
    mock_img_data = np.array(
        [
            [
                (np.array([1080 / n_bands * j + 360 / n_bands * i] * 10) + (np.random.rand(10) * 120 / n_bands))
                .astype(np.uint16)
                .tolist()
                for i in range(5 * sample_n)
            ]
            for j in range(n_bands)
        ]
    )

    for i in range(5 * sample_n):
        mock_img_data[-int(n_bands / 3) :, i, :] = mock_img_data[-int(n_bands / 3) :, i, :] * (i // 5 % 2 + 0.5)

    # Mock image
    img_path = dir_path + "/test_img.tif"
    create_test_raster(
        raster_path=img_path,
        width=mock_img_data.shape[2],
        height=mock_img_data.shape[1],
        bands=n_bands,
        data=mock_img_data,
    )
    exp1.add_images("test_group", ["test_img.tif"], dir_path)

    # ROIs
    roi_path = dir_path + "/test_roi.xml"
    create_test_roi_xml(roi_path, roi_count=sample_n)
    exp1.add_rois_by_file("test_group", [roi_path], "test_img.tif")

    # Samples
    dflb = exp1.ls_sample_labels()
    assert isinstance(dflb, pd.DataFrame)
    dflb.iloc[:, 1] = [f"sample_{str(i + 1)}" for i in range(len(dflb))]
    exp1.sample_labels = dflb  # type: ignore[assignment]
    # Auto-conversion in setter

    # Target values
    dft = exp1.ls_sample_targets()
    assert isinstance(dft, pd.DataFrame)
    if is_regression:
        dft["Target_value"] = list(range(len(dft)))
    else:
        dft["Target_value"] = [["a", "b"][int(i % 2)] for i in range(len(dft))]
    exp1.sample_targets_from_df(dft)

    return exp1


# Alias
create_example_spec_exp = create_test_spec_exp
