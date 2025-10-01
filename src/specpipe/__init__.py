# -*- coding: utf-8 -*-
"""
SpecPipe - A file-based pipeline for efficient batch processing and modeling of hyperspectral images
"""

# Package meta
__version__ = "0.1.0"
__author__ = "Siwei Luo"
__license__ = "MIT"

# Imports
__all__ = [
    ## Core pipeline tools
    ### Core class
    "SpecExp",
    "SpecPipe",
    "ModelEva",
    ### SpecIO functions
    "search_file",
    "envi_roi_coords",
    "shp_roi_coords",
    ### Raster operation
    "croproi",
    "pixel_apply",
    ## Spectral statistics tools
    ### ROI statistics
    "round_digit",
    "roispec",
    "pixcount",
    "nderiv",
    "moment2d",
    "bandhist",
    "Stats2d",
    "roi_mean",
    "roi_std",
    "roi_median",
    "spectral_angle",
    "arr_spectral_angles",
    ## Mock example data
    "create_example_raster",
    "create_example_roi_xml",
    "create_example_spec_exp",
]

# Components
## Core pipeline tools
## Example data generator
from .example_data import create_example_raster, create_example_roi_xml, create_example_spec_exp
from .modeleva import ModelEva
from .rasterop import croproi, pixel_apply

## Spectral statistics tools
from .roistats import (
    Stats2d,
    arr_spectral_angles,
    bandhist,
    moment2d,
    nderiv,
    pixcount,
    roi_mean,
    roi_median,
    roi_std,
    roispec,
    round_digit,
    spectral_angle,
)
from .specexp import SpecExp
from .specio import envi_roi_coords, search_file, shp_roi_coords
from .specpipe import SpecPipe

## Spectral processing tools
