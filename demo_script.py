# -*- coding: utf-8 -*-
"""
Usage demo for SpecPipe

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Real-world data demo

# 1. Data preparation
# Set data directory path
import os
import shutil

# Setup a directory for demo
demo_dir = os.getcwd() + "/SpecPipeDemo/"

if os.path.exists(demo_dir):
    shutil.rmtree(demo_dir)

# Setup data directory and demo data
data_dir = demo_dir + "demo_data/"

os.makedirs(data_dir)

# Download real-world demo raster image and ROI files
# Demo data URL: https://github.com/siwei66/specpipe/tree/master/demo/demo_data/
from specpipe import download_demo_data

download_demo_data(data_dir)

# Create a directory for pipeline results
report_dir = demo_dir + "/demo_results_classification/"

os.makedirs(report_dir)


# 2. Configure your experiment data
# 2.1 Create a spectral experiment
# Create a SpecExp instance for experiment data
from specpipe import SpecExp

exp = SpecExp(report_dir)

# Check report directory
exp.report_directory

# 2.2. Experiment group management
# Add experiment groups
exp.add_groups(["group_1", "group_2"])

# 2.3. Raster image management
# Add raster images
exp.add_images_by_name(image_name="demo.", image_directory=data_dir, group="group_1")
exp.add_images_by_name("demo.", data_dir, "group_2")

# Check added images
exp.ls_images()

# 2.4. Region of interest (ROI) management
# Load image ROIs using suffix to image names
exp.add_rois_by_suffix(roi_filename_suffix="_[12].xml", search_directory=data_dir, group="group_1")
exp.add_rois_by_suffix("_[345].xml", data_dir, "group_2")

# Check added ROIs
exp.ls_rois()

# Show RGB preview with ROIs
exp.show_image("demo.tiff", "group_1", rgb_band_index=(19, 12, 6), output_path=report_dir + "demo_rast_rgb1.png")
exp.show_image("demo.tiff", "group_2", rgb_band_index=(19, 12, 6), output_path=report_dir + "demo_rast_rgb2.png")

# 2.5. Sample labels and target values

# 2.5.1 Set sample labels

# Retrieve original sample label dataframe
labels = exp.ls_labels()

# Update sample labels using sample ROI names
labels.iloc[:, 1] = exp.ls_rois_sample(return_dataframe=True)["ROI_name"]  # type: ignore

# Set sample labels using the updated label dataframe
exp.sample_labels = labels  # type: ignore

# Check new sample labels
exp.ls_labels()

# 2.5.2 Set target values

# List target value dataframe
targets = exp.ls_sample_targets()

# Set the leaf order as target values
targets["Target_value"] = [f"leaf_{labl[0]}" for labl in targets['Label']]  # type: ignore

# Load target values from updated target dataframe
exp.sample_targets_from_df(targets)

# Check target values
exp.ls_targets()[["Sample_ID", "Target_value"]]


# 3. Design testing pipeline

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
    vstd[vstd == 0] = 1e-10
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
pipe.add_process(5, 7, 0, roi_median)

# 3.4 Add models to the pipeline
# Create some models
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore

rf_classifier = RandomForestClassifier(n_estimators=10)
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Add models
pipe.add_model(knn_classifier, validation_method="2-fold")
pipe.add_model(rf_classifier, validation_method="2-fold")

# Check added models
pipe.ls_model()


# 4 Run pipeline

# Check processing chains with method id
pipe.ls_chains()

# Run pipeline
pipe.run()

pipe.run(n_processor=10)


# 5 Regression Case

# Create a directory for regression results
report_dir_reg = demo_dir + "/demo_results_regression/"
if not os.path.exists(report_dir_reg):
    os.makedirs(report_dir_reg)

# Copy SpecExp and SpecPipe
import copy

exp_reg = copy.deepcopy(exp)
pipe_reg = copy.deepcopy(pipe)
targets_reg = copy.deepcopy(targets)

# Update report directory of SpecExp
exp_reg.report_directory = report_dir_reg

# Check report directory
exp_reg.report_directory

# Modify targets to numeric, here the numbers approaximate the age of the leaves
targets_reg["Target_value"] = [(5 - int(labl[0])) for labl in targets['Label']]  # type: ignore
exp_reg.sample_targets_from_df(targets_reg)

# Check target values
exp_reg.ls_targets()[["Sample_ID", "Target_value"]]

# Check and remove classification models
pipe_reg.ls_model()
pipe_reg.rm_model()

# Update the pipeline
pipe_reg.spec_exp = exp_reg

# Add regressors to the pipeline
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.neighbors import KNeighborsRegressor  # type: ignore

rf_regressor = RandomForestRegressor(n_estimators=10)
knn_regressor = KNeighborsRegressor(n_neighbors=3)

# Add models
pipe_reg.add_model(knn_regressor, validation_method="2-fold")
pipe_reg.add_model(rf_regressor, validation_method="2-fold")

# Check models
pipe_reg.ls_model()

# Check processing chains and run the pipeline
pipe_reg.ls_chains()

# Run regression pipeline
pipe_reg.run()

pipe_reg.run(n_processor=10)
