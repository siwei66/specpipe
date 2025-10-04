# -*- coding: utf-8 -*-
"""
Usage example for SpecPipe

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Simple example code

# 1. Prepare mock spectral experiment data
# Create a directory for mock experiment data (The example uses a temporary directory)
import tempfile

data_dir = tempfile.mkdtemp(prefix="spec_pipe_example_")

# Create mock data
from specpipe import create_example_raster, create_example_roi_xml

create_example_raster(f"{data_dir}/example.tif")
create_example_roi_xml(f"{data_dir}/example_roi.xml")


# 2. Configure your experiment data
# 2.1 Create a spectral experiment
# Here we use the same directory as report directory
report_dir = data_dir

# Create a SpecExp instance for experiment data
from specpipe import SpecExp

exp = SpecExp(report_dir)

# Check report directory
exp.report_directory


# 2.2. Experiment group management
# Add experiment groups
exp.add_groups(["group_1", "group_2"])

# Check groups
exp.ls_groups()

# Remove a group
exp.rm_group("group_2")


# 2.3. Raster image management
# Add raster images
exp.add_images(group_name="group_1", image_name="example", image_directory=data_dir)

# Check added images
exp.ls_images()


# 2.4. Region of interest (ROI) management
# Load image ROIs using suffix to image names
exp.add_rois_by_suffix("group_1", "_roi.xml", data_dir)

# Remove ROIs by name
exp.rm_rois(roi_name="ROI_10")

# Load ROIs to a image using ROI files by paths
exp.add_rois_by_file("group_1", [f"{data_dir}/example_roi.xml"], image_name="example.tif")

# Check added ROIs
exp.ls_rois()

# Check sample_rois
exp.ls_rois_sample()


# 2.5. Sample labels and target values

# 2.5.1 Set sample labels

# List sample label dataframe
labels = exp.ls_labels()

# Update sample labels
labels.iloc[:, 1] = [f"sample_{str(i + 1)}" for i in range(len(labels))]  # type: ignore

# Set sample labels using the updated label dataframe
exp.sample_labels = labels  # type: ignore

# Check sample labels
exp.ls_labels()

# 2.5.2 Set target values

# List target value dataframe
targets = exp.ls_sample_targets()

# Create mock target values for regression and update target dataframe
targets["Target_value"] = [i for i in range(len(targets))]  # type: ignore

# Load target values from updated target dataframe
exp.sample_targets_from_df(targets)

# Check target values
exp.ls_targets()


# 3. Design testing pipeline
"""
The processing functions are wrapped according to different 'data levels'.
A classic data levels in spectral image processing is: 
##    raster image data -> ROI spectra -> ROI statistics -> traits to model

The data levels in SpecPipe is:
    Images: 
    0 - 'image', input image path and output processed image path.
    
    Image pixel spectra: 
    1 - 'pixel_spec', 1D spectrum of image pixel
    2 - 'pixel_specs_array', 2D spectra array of image pixels
    3 - 'pixel_specs_tensor', 2D spectra tensor of image pixels
    4 - 'pixel_hyperspecs_tensor', 2D hyperspectra tensor of image pixels (optimized tiling for hyperspectral images)
    (See 'rasterop.pixel_apply')
    
    Image ROIs
    5 - 'image_roi', get image ROI data, commonly get a unsorted list of ROI spectra
    6 - 'roispecs', ROI spectra in array
    7 - 'spec1d', arbitray 1D data extracted from roispecs or 1D spectra data from spectrascopy
    
    Models
    8 - 'model', model evaluation with standard reports in files, only as output level.

A process method is wrapped according to its data level, 
Parallel processes with same data levels and application sequences are arranged using full-factorial approach by default.
"""

# 3.1 Create processing pipeline
from specpipe import SpecPipe

pipe = SpecPipe(exp)

# 3.2 Image processing

# Create some image processing functions
import numpy as np


# Standard normal variate
def snv(v):  # type: ignore
    vmean = np.mean(v, axis=1, keepdims=True)
    vstd = np.std(v, axis=1, keepdims=True)
    snv = (v - vmean) / vstd
    return snv


# Compared with raw data
def raw(v):  # type: ignore
    return v


# Add these process to the pipeline
pipe.add_process(
    input_data_level="pixel_specs_array",
    output_data_level="pixel_specs_array",
    application_sequence=0,
    method=snv,
)

# Or specify data level using data level number
pipe.add_process(2, 2, 0, raw)


# 3.3 ROI statistics
# Import some ROI spectral statistic metrics
from specpipe import roi_mean, roi_median

# Add these process to the pipeline
pipe.add_process(
    input_data_level="image_roi",
    output_data_level="spec1d",
    application_sequence=0,
    method=roi_mean,
)

# Or specify data level using data level number
pipe.add_process(5, 7, 0, roi_median)


# 3.4 Sample data wrangling
# Create a function to remove nan and inf values
def replace_nan(v: np.ndarray) -> np.ndarray:  # type: ignore
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore


# Add process
pipe.add_process("spec1d", "spec1d", 0, replace_nan)

# Check added process
pipe.ls_process()

# Remove added process
pipe.rm_process(method="raw")

# 3.5 Add models to the pipeline
# Create some models
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.neighbors import KNeighborsRegressor  # type: ignore
from sklearn.svm import SVR  # type: ignore

linear_regressor = LinearRegression()
rf_regressor = RandomForestRegressor(n_estimators=10)
knn_regressor = KNeighborsRegressor(n_neighbors=3)
svr = SVR()

# Add model using add_process
pipe.add_process("spec1d", "model", 1, linear_regressor, validation_method="10-fold")
pipe.add_process(7, 8, 1, rf_regressor, validation_method="10-fold")

# Add model using add_model
pipe.add_model(knn_regressor, validation_method="10-fold")
pipe.add_model(svr, validation_method="10-fold")

# Check added models
pipe.ls_model()

# Check all processes including models
pipe.ls_process()


# 4 Run pipeline

# Check processing chains with method id
pipe.ls_chains()

# Manually test all added methods (also run automatically before formal run)
pipe.test_run()

# Run pipeline
pipe.run()

# Set resume True to enable resuming after break
# pipe.run(resume=True)

# Input data file structure
"""
Input data directory

data_directory/
├── SpecExp_configuration/
│    ├── Loading_history/
│    │   ├── Loaded_images.csv
│    │   └── Loaded_ROIs.csv
│    └── SpecExp_data_configuration.dill
├── Your_rasters.tif
└── Your_ROIs.xml
"""

# Output report file structure for regression task
"""
Output report directory

report_directory/
├── Modeling/
│    └── Model_evaluation_reports/
│        ├── Data_chain_Preprocessing_#0_Model_(model label 0)/
│        │   ├── Model_for_application/
│        │   ├── Model_in_validation/
│        │   ├── Regression_performance.csv
│        │   ├── Validation_results.csv
│        │   ├── Residual_analysis.csv
│        │   ├── Influence_analysis.csv
│        │   ├── Scatter_plot.png
│        │   └── Residual_plot.png
│        ├── Data_chain_Preprocessing_#0_Model_(model label 1)/
│        ├── Data_chain_Preprocessing_#1_Model_(model label 0)/
│        ├── Data_chain_Preprocessing_#1_Model_(model label 1)/
│        ├──  Preprocessing_#0.txt
│        └──  Preprocessing_#1.txt
├── Pre_execution_test_data/
├── Preprocessing/
└── test_run/
"""

# Output report file structure for classification task
"""
Output report directory

report_directory/
├── Modeling/
│    └── Model_evaluation_reports/
│        ├── Data_chain_Preprocessing_#0_Model_(model label 0)/
│        │   ├── Model_for_application/
│        │   ├── Model_in_validation/
│        │   ├── Classification_performance.csv
│        │   ├── Validation_results.csv
│        │   ├── Residual_analysis.csv
│        │   ├── Influence_analysis.csv
│        │   └── ROC_curve.png
│        ├── Data_chain_Preprocessing_#0_Model_(model label 1)/
│        ├── Data_chain_Preprocessing_#1_Model_(model label 0)/
│        ├── Data_chain_Preprocessing_#1_Model_(model label 1)/
│        ├──  Preprocessing_#0.txt
│        └──  Preprocessing_#1.txt
├── Pre_execution_test_data/
├── Preprocessing/
│    ├── Step_results/
│    ├── PreprocessingChainResult_chain_0.csv
│    └── PreprocessingChainResult_chain_1.csv
└── test_run/
"""

# Clear the temporary directory after use
import shutil

shutil.rmtree(data_dir)
