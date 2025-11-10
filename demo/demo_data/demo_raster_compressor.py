# -*- coding: utf-8 -*-
"""
SpecPipe - Demo data subset & compressor

Copyright (c) 2025 Siwei Luo. MIT License.
"""
import os
import copy
import rasterio
from rasterio.transform import from_bounds


#%% Band sampler


def band_sampler(src_rast_path: str, dst_rast_path: str, band_index_subset: int) -> None:
    """Uniform sampling of bands and subset the original image"""
    with rasterio.open(src_rast_path) as src:

        # Read the data
        data = src.read(band_index_subset)
        height, width = data.shape[1], data.shape[2]

        # Reset bbox to keep the pixel coordinates of the demo image
        original_bbox = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        print(f"Original bounding box is: {original_bbox}")

        # New bbox
        left = 0.0
        bottom = src.bounds.bottom - src.bounds.top
        right = src.bounds.right - src.bounds.left
        top = 0.0
        print(f"New bounding box is: {(left, bottom, right, top)}")

        # Update meta
        new_transform = from_bounds(left, bottom, right, top, width, height)
        new_meta = copy.deepcopy(src.meta)
        new_meta.update({
            'count': len(band_index_subset),
            'transform': new_transform,
            'compress': 'lzw',
            'crs': None  # Remove CRS for pure pixel coordinates
        })

        # Write new file with hard-set bounds
        with rasterio.open(dst_rast_path, 'w', **new_meta) as dst:
            dst.write(data)


#%% Validate compressed image


# Source and destination raster paths
src_rast_path = os.getcwd() + "/SpecPipe/demo_data/demo_bounds_reset.tiff"
dst_rast_path = os.getcwd() + "/SpecPipe/demo_data/demo.tiff"

# Band subset index
band_index_subset = [i + 1 for i in range(462) if i % 10 == 5]

# Sample and subset
band_sampler(src_rast_path, dst_rast_path, band_index_subset)


with rasterio.open(dst_rast_path) as rst:
    print(f"\nNew image number of bands: {rst.count}")
    print(f"\nNew image bounding box: {(rst.bounds.left, rst.bounds.bottom, rst.bounds.right, rst.bounds.top)}")
    print(f"\nNew image (400, 400, 600, 600) at band 20: \n\n{rst.read(20)[400: 600, 400: 600]}")
