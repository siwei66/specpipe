# -*- coding: utf-8 -*-
"""
SpecPipe - A file-based pipeline for efficient batch processing and modeling of hyperspectral images
"""

# Package meta
__version__ = "0.3.1"
__author__ = "Siwei Luo"
__license__ = "MIT"

# Imports
__all__ = [
    ## Core pipeline tools
    ### Core class
    "SpecExp",
    "SpecPipe",
    "ModelEva",
    "raster_rgb_preview",
    ### SpecIO functions
    "search_file",
    "envi_roi_coords",
    "shp_roi_coords",
    ### Raster operation
    "croproi",
    "pixel_apply",
    ### Model combiners
    "combine_transformer_classifier",
    "combine_transformer_regressor",
    "factorial_transformer_chains",
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
    ## Demo
    "download_demo_data",
]

# Components
## Core pipeline tools
## Example data
from .example_data import create_example_raster, create_example_roi_xml, create_example_spec_exp, download_demo_data

## Model Evaluator
from .modeleva import ModelEva

## Model combiners
from .modelconnector import factorial_transformer_chains, combine_transformer_classifier, combine_transformer_regressor

## Raster operation
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

## Spectral experiment data management
from .specexp import SpecExp
from .specexp_vis import raster_rgb_preview

## SpecPipe IO tools
from .specio import envi_roi_coords, search_file, shp_roi_coords

## SpecPipe main pipeline
from .specpipe import SpecPipe
