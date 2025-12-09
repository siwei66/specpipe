# -*- coding: utf-8 -*-
"""
SpecPipe - Pipeline management and implemention module for spectral image processing and modeling

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# OS
import os
import sys
import shutil

# Interface
from tqdm import tqdm

# Warning
import warnings

# Typing
from typing import Annotated, Any, Callable, Literal, Optional, Union, overload
from types import ModuleType

# Time
import time
from datetime import datetime

# Functions
from functools import partial

# Basic data
import copy
import json
import numpy as np
import pandas as pd
import torch

# Raster
import rasterio
from rasterio.windows import Window
from rasterio.errors import NotGeoreferencedWarning

# Visualization
import matplotlib.pyplot as plt

# Multiprocessing
import dill
from pathos.helpers import mp
from pathos.multiprocessing import ProcessingPool, cpu_count

# Local
from .groupstats import sample_group_stats, performance_marginal_stats
from .modeleva import ModelEva
from .rasterop import croproi, pixel_apply
from .resultcli import group_stats_report, core_chain_report
from .roistats import ROISpec, minbbox
from .specexp import SpecExp
from .specio import (
    arraylike_validator,
    dataframe_validator,
    df_to_csv,
    dump_vars,
    load_vars,
    RealNumber,
    simple_type_validator,
    unc_path,
)
from .specpipe_validator import (
    _target_type_validation_for_serialization,
    _dl_val,
    _data_level_seq_validator,
    _spec_exp_validator,
    _process_validator,
    _classifier_validator,
    _regressor_validator,
)
from .specpipe_processor import (
    _preprocessing_sample,
    _ModelMethod,
    _model_evaluator,
    _model_evaluator_mp,
)
from .modelconnector import (
    combined_model_marginal_stats,
)

# For multiprocessing
global ModelEva


# %% Spectral Modeling Pipeline Class - SpecPipe


class SpecPipe:
    """
    Design and implement a processing and modeling pipeline for spectral experiment datasets.


    Attributes:
    -----------
    spec_exp : SpecExp
        Instance of SpecExp configuring spectral experiment datasets. See SpecExp for details.
    process : list[tuple[str, str, str, str, int, Callable, int, int]]
        Added process items.
    process_chains : list[tuple[str, ...]]
        Generated full-factorial process chains.

    Methods:
    --------
    - add_process :
        Add a processing method with defined input/output data levels and application sequence to the pipeline.
        A processing method can be a preprocessing function or a model for evaluation.

    - ls_process :
        List process items filtered by process properties.

    - rm_process :
        Remove added processes by ID, Process_label, Data_level, Application_sequence and Method.

    - add_model :
        Add a model to the pipeline.

    - ls_model :
        List model process items filtered by given properties.

    - rm_model :
        Remove added models by model_id, model_label and model method (object).

    - process_chains_to_df (Alias: ls_process_chains) :
        List process chains in a dataframe. Returns the default full factorial process chains.
        Output is a dataframe, where each row represents a processing chain with process IDs.

    - custom_chains_from_df :
        Customize processing chains and update chains using chain dataframe.
        Once custom chains are created, SpecPipe will prioritize their execution, bypassing the original full-factorial chains.

    - custom_chains_to_df (Alias: ls_custom_chains) :
        List custom chains in a dataframe.

    - ls_chains :
        List the process chains in the pipeline execution.
        Returns custom chains if configured, otherwise returns default full factorial chains.

    - save_pipe_config :
        Save current pipeline configurations to files in the root of report directory.

    - load_pipe_config :
        Load SpecPipe configurations from dill file.

    - test_run :
        Run the pipeline of all processing chains using simplified test data. This method is executed automatically prior to each formal run.

    - preprocessing :
        Implement preprocessing steps of all processing chains on the entire dataset and output modeling-ready sample_list data to files.

    - model_evaluation :
        Evaluating added models on sample data from all preprocessing chains.

    - run :
        Run pipeline of given processes on SpecExp instance (corresponding manager of spectral experiment data).
        Full-factorial test is applied to multiple processes of identical sequence.

    - report_summary :
        Retrieve summary of reports in the console, including performance summary and marginal performances among processes.

    - report_chain :
        Retrieve major model evaluation reports of every processing chain in the console.
    """  # noqa: E501

    @simple_type_validator
    def __init__(self, spec_exp: SpecExp) -> None:  # noqa: C901
        ## Experiment data manager - SpecExp
        # Validate SpecExp integrity
        _spec_exp_validator(spec_exp)

        # SpecExp
        # SpecExp._groups: [0 group]
        # SpecExp._images: [0 id, 1 group, 2 image_name, 3 image_use_type, 4 image_path]
        # SpecExp._rois: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
        self._spec_exp: SpecExp = spec_exp

        # Sample target values
        # [0 fixed sample id, 1 user assinged labels, 2 Target values]
        self._sample_targets: list[tuple[str, str, Union[str, bool, int, float], str]] = spec_exp.sample_targets

        # Is_target_numeric according to sample target values
        self._is_target_numeric: bool = self._check_target_numeric(spec_exp)

        # Report directory
        self._report_directory: str = self.spec_exp._report_directory

        # Processes
        # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
        self._process: list[tuple[str, str, str, str, int, Callable, int, int]] = []

        # Generated process chain for full factors
        # [(process 1 ID of step 1, process 2 ID of step 1,...), (process 1 ID of step 2, process 2 ID of step 2,...), ...]  # noqa: E501
        self._process_steps: list = []
        # [(process 1 ID of step 1, process 1 ID of step 2,...), (process 2 ID of step 1, process 1 ID of step 2,...), ...]  # noqa: E501
        self._process_chains: list = []
        # Custom chains for custom partly test
        self._custom_chains: list = []

        # Band wavelength
        self._band_wavelength: Optional[tuple] = None

        # Pre-execution test data
        # Sample data format - ROI: {ID, label, target, img_path, roi_coords}
        # Sample data format - standalone spec: {ID, label, target, spec1d: tuple}
        # Sample data format - test: {img_path, test_img_path, roi_coords, test_roi_coords, roitable, spec1d}
        self._pretest_data: Optional[dict[str, Any]] = None
        self._pretest_data_init()

        # Pipeline creating time
        self._create_time: str = datetime.now().strftime("created_at_%Y-%m-%d_%H-%M-%S")

        # Input sample data
        self._sample_data: list[dict[str, Any]] = []

        # test_run status
        # Set False for any process modification, set True after successful calling of 'test_run'
        self._tested: bool = False

        # Sample preprocessing result paths
        self._preprocess_result_path: list[str] = []

    ## Mutable properties
    @property
    def report_directory(self) -> str:
        return self._report_directory

    @report_directory.setter
    def report_directory(self, value: str) -> None:
        if os.path.exists(unc_path(value)):
            warning_msg = (
                "The current report_directory is shared with the SpecExp's report_directory. "
                "It is recommended to set the report_directory directly in SpecExp instead. "
                "Note that modifying it here will not update SpecExp's report_directory, "
                "which may result in test reports being saved in two different locations."
            )
            warnings.warn(warning_msg, UserWarning, stacklevel=2)
            value = (str(value).replace("\\", "/") + "/").replace("//", "/")
            self._report_directory = value
        else:
            raise ValueError(f"Given report_directory is invalid: {value}")

    @property
    def spec_exp(self) -> SpecExp:
        return self._spec_exp

    @spec_exp.setter
    def spec_exp(self, spec_exp: SpecExp) -> None:
        if isinstance(spec_exp, SpecExp):
            self._tested = False
            self._spec_exp_updater(spec_exp)
        else:
            raise ValueError(f"{self.__class__.__name__}.spec_exp must be a SpecExp instance")

    # Alias method
    def update_spec_exp(self, spec_exp: SpecExp) -> None:
        self.spec_exp = spec_exp

    @property
    def band_wavelength(self) -> Optional[tuple]:
        return self._band_wavelength

    @band_wavelength.setter
    def band_wavelength(self, value: Optional[tuple]) -> None:
        if value is not None:
            value = tuple(value)
            if self._pretest_data is None:
                raise ValueError(
                    "Internal Error: 'SpecPipe.pretest_data' is None. \
                        Pre-execution test data initialization fails. Please report."
                )
            if len(value) != len(self._pretest_data["spec1d"]):
                raise ValueError(
                    f"The number of band wavelengths ({len(value)}) does not match \
                        the number of bands ({len(self._pretest_data['spec1d'])})."
                )
            v0 = 0
            for v in value:
                if (type(v) is not float) | (type(v) is not int):
                    raise TypeError(f"Band wavelengths must be numeric, got type: {type(v)}.")
                if v <= 0:
                    raise ValueError(f"Band wavelengths must be positive, got: {v}")
                if v > v0:
                    v0 = v
                else:
                    raise ValueError(f"Band wavelengths must be in an ascending order without ties, got: {value}")
        self._band_wavelength = value

    ## Read only or immuatable properties
    @property
    def sample_targets(self) -> list[tuple[str, str, Union[str, bool, int, float], str]]:
        return self._sample_targets

    @sample_targets.setter
    def sample_targets(self, value: list[tuple[str, str, Union[str, bool, int, float], str]]) -> None:
        raise ValueError("sample_targets cannot be modified in SpecPipe, please update using 'SpecExp' instead")

    @property
    def is_target_numeric(self) -> bool:
        return self._is_target_numeric

    @is_target_numeric.setter
    def is_target_numeric(self, value: bool) -> None:
        raise ValueError("is_target_numeric cannot be modified in SpecPipe, please update using 'SpecExp' instead")

    @property
    def process(self) -> list[tuple[str, str, str, str, int, Callable, int, int]]:
        return self._process

    @process.setter
    def process(self, value: list[tuple[str, str, str, str, int, Callable, int, int]]) -> None:
        raise ValueError("process cannot be modified directly, use 'add_process' and 'rm_process' instead")

    @property
    def process_steps(self) -> list:
        return self._process_steps

    @process_steps.setter
    def process_steps(self, value: list) -> None:
        raise ValueError("process_steps cannot be modified")

    @property
    def process_chains(self) -> list:
        return self._process_chains

    @process_chains.setter
    def process_chains(self, value: list) -> None:
        raise ValueError("process_chains cannot be modified")

    @property
    def custom_chains(self) -> list:
        return self._custom_chains

    @custom_chains.setter
    def custom_chains(self, value: list) -> None:
        raise ValueError("custom_chains cannot be modified directly, use 'custom_chains_from_df' to set custom_chains")

    @property
    def sample_data(self) -> list[dict[str, Any]]:
        return self._sample_data

    @sample_data.setter
    def sample_data(self, value: list[dict[str, Any]]) -> None:
        raise ValueError("sample_data cannot be modified")

    @property
    def pretest_data(self) -> Optional[dict]:
        return self._pretest_data

    @pretest_data.setter
    def pretest_data(self, value: Optional[dict]) -> None:
        raise ValueError("pretest_data cannot be modified")

    @property
    def preprocess_result_path(self) -> list[str]:
        return self._preprocess_result_path

    @preprocess_result_path.setter
    def preprocess_result_path(self, value: list[str]) -> None:
        raise ValueError("preprocess_result_path cannot be modified")

    @property
    def tested(self) -> bool:
        return self._tested

    @tested.setter
    def tested(self, value: bool) -> None:
        raise ValueError("tested is immutable and cannot be modified")

    @property
    def create_time(self) -> str:
        return self._create_time

    @create_time.setter
    def create_time(self, value: str) -> None:
        raise ValueError("create_time is immutable and cannot be modified")

    # SpecExp target value numeric validator
    @staticmethod
    @simple_type_validator
    def _check_target_numeric(spec_exp: SpecExp) -> bool:
        # Read target values
        sample_target_values = [spt[2] for spt in spec_exp.sample_targets]
        # Validate whether numeric
        is_numeric: bool = True
        for yi in sample_target_values:
            if not isinstance(yi, RealNumber):
                is_numeric = False
        return is_numeric

    # SpecExp-related initializer / updater
    @simple_type_validator
    def _spec_exp_updater(self, spec_exp: SpecExp) -> None:
        # Backup current spec_exp
        spec_exp_old = copy.deepcopy(self.spec_exp)
        # Update data and test
        try:
            _spec_exp_validator(spec_exp)
        except Exception as e:
            raise ValueError("Given SpecExp instance is invalid") from e
        try:
            self._spec_exp = spec_exp
            self._sample_targets = spec_exp.sample_targets
            self._is_regression = self._check_target_numeric(spec_exp)
            self._report_directory = spec_exp._report_directory
            self._pretest_data_init()
            if len(self.process_chains) > 0:
                self.test_run(dump_result=False)
        except Exception as e:
            # Roll back when fail in test
            self._spec_exp = spec_exp_old
            self._sample_targets = spec_exp_old.sample_targets
            self._is_regression = self._check_target_numeric(spec_exp_old)
            self._report_directory = spec_exp_old._report_directory
            self._pretest_data_init()
            raise ValueError("Given SpecExp failed in test_run, spec_exp configuration rolls back.") from e

    # Process configuration ===================================

    # Generate test data of each level for process validation
    # From data levels:
    # 0 - image (path), \
    # 1 - pixel_spec (1D), 2 - pixel_specs_array (2D), 3 - pixel_specs_tensor (3D), 4 - pixel_hyperspecs_tensor (3D), \
    # 5 - image_ROI (img_path + ROI coords), 6 - ROI_specs (2D), 7 - spec1d (1D spec stats)
    # Pretest_data: [img_path, test_img_path, roi_coords, test_roi_coords, roitable, spec1d]
    @simple_type_validator
    def _pretest_data_init(self) -> None:
        """
        Initialize test data of each level for process validation before method execution.
        """
        ### For image samples
        if len(self.spec_exp.standalone_specs_sample) == 0:
            # Get first sample ROI item
            sroi_test = self._spec_exp._rois_sample[0]
            roicoords = sroi_test[5]

            # Get associated image item - raster image path

            img_test0 = self._spec_exp._get_images([sroi_test[2]], sroi_test[1], None)[0]
            if len(img_test0) == 1:
                img_test = img_test0[0]
                img_path = img_test[4]
            else:
                raise ValueError(f"\nSpecExp data error: duplicated image items: \n{img_test0}")

            # Get ROI spec table
            try:
                bdmin = minbbox(img_path, roicoords, 100)
                roitable = ROISpec(as_type=float).roispec(img_path, bdmin)
            except Exception as e:
                raise ValueError(
                    f"\nUnable to retrieve ROI spectra from image:\n\nROI:\n{sroi_test}\n\nImage:\n{img_test}\n\n{e}"
                ) from e

            # Get test image ROI
            test_roi_coords = [
                [
                    (0, 0),
                    (max(int(bdmin[0][1][0] - bdmin[0][0][0]), 1), 0),
                    (
                        max(int(bdmin[0][1][0] - bdmin[0][0][0]), 1),
                        max(int(bdmin[0][2][1] - bdmin[0][1][1]), 1),
                    ),
                    (0, max(int(bdmin[0][2][1] - bdmin[0][1][1]), 1)),
                    (0, 0),
                ]
            ]

            # Single spec
            spec1d = tuple(roitable[0, :])

            ## Save to file
            # Output path
            sdir = self._spec_exp.report_directory + "Pre_execution_test_data/"
            if not os.path.exists(unc_path(sdir)):
                os.makedirs(unc_path(sdir))

            # Save test image
            test_img_path = sdir + "test_images." + img_path.split(".")[-1]
            with open(unc_path(sdir + "pre_execution_data.json"), "w") as f:
                # Save test image
                croproi(img_path, bdmin, test_img_path)
                # Convert np.float32 to native float for json dump
                # Save test spectra
                td1 = {
                    "img_path": img_path,
                    "test_img_path": test_img_path,
                    "roi_coords": bdmin,
                    "test_roi_coords": test_roi_coords,
                    "spec1d": spec1d,
                }
                # Dump
                json.dump(td1, f)

            # Testing spectrum table
            td2 = pd.DataFrame(roitable, columns=[("Band_" + str(i + 1)) for i in range(roitable.shape[1])])
            td2.to_csv(unc_path(sdir + "Pre_execution_data_roi_specs.csv"), index=False)

            # Output for pre-execution testing data
            test_data_pre = {
                "ID": "test_run",
                "label": "test_run",
                "target": None,
                "img_path": img_path,
                "test_img_path": test_img_path,
                "roi_coords": bdmin,
                "test_roi_coords": test_roi_coords,
                "roi_specs": roitable,
                "spec1d": spec1d,
            }

            self._pretest_data = test_data_pre

        ### For standalone spectral samples
        # Pretest_data: [img_path, test_img_path, roi_coords, test_roi_coords, roitable, spec1d]
        else:
            # Config test item data
            img_path = None
            test_img_path = None
            bdmin = None
            test_roi_coords = None
            roitable = None
            spec1d = tuple(self.spec_exp.standalone_specs_sample[0][-1])

            ## Save to file
            # Output path
            sdir = self._spec_exp.report_directory + "Pre_execution_test_data/"
            if not os.path.exists(unc_path(sdir)):
                os.makedirs(unc_path(sdir))

            # Testing spectrum table
            td2 = pd.DataFrame([list(spec1d)], columns=[("Band_" + str(i + 1)) for i in range(len(spec1d))])
            td2.to_csv(unc_path(sdir + "Pre_execution_data_standalone_specs.csv"), index=False)

            # Output for pre-execution testing data
            test_data_pre = {
                "ID": "test_run",
                "label": "test_run",
                "target": None,
                "img_path": img_path,
                "test_img_path": test_img_path,
                "roi_coords": bdmin,
                "test_roi_coords": test_roi_coords,
                "roi_specs": roitable,
                "spec1d": spec1d,
            }
            self._pretest_data = test_data_pre

    # Process function validator - pretest validator
    # Data levels:
    # 0 - image (path), \
    # 1 - pixel_spec (1D), 2 - pixel_specs_array (2D), 3 - pixel_specs_tensor (3D), 4 - pixel_hyperspecs_tensor (3D), \
    # 5 - image_ROI (img_path + ROI coords), 6 - ROI_specs (2D), 7 - spec1d (1D spec stats)
    # Pretest_data: [img_path, test_img_path, roi_coords, test_roi_coords, roitable, spec1d]
    @simple_type_validator
    def _process_validator(  # noqa: C901
        self,
        method: Callable,
        input_data_level: Union[str, int],
        output_data_level: Union[str, int],
    ) -> Callable:
        """
        Validate process method of specified input data level before execution of entire processing chain.
        """
        # Pretest_data validation for static typing
        if self._pretest_data is None:
            raise ValueError(
                "Internal Error: 'SpecPipe.pretest_data' is None. \
                    Pre-execution test data initialization fails. Please report."
            )

        # Applied only for image samples
        if len(self.spec_exp.standalone_specs_sample) == 0:
            # Validate data_level
            dl_in = _dl_val(input_data_level)[0]
            dl_out = _dl_val(output_data_level)[0]

            # Test image path
            test_img_path = self._pretest_data["test_img_path"]

            # Test data
            if dl_out < 8:
                if dl_in == 0:
                    # Output dst image path
                    img_path = os.path.splitext(str(test_img_path).replace("\\", "/").replace("//", "/"))
                    img_name = img_path[0].split("/")[-1]
                    output_path = (
                        self.report_directory
                        + "/Pre_execution_test_data/"
                        + img_name
                        + "_processed_by_"
                        + method.__name__
                        + img_path[1]
                    )
                    # Process image and return path of processed image
                    result = method(test_img_path, output_path)
                elif dl_in == 1:
                    # Output dst image path
                    img_path = os.path.splitext(str(test_img_path).replace("\\", "/").replace("//", "/"))
                    img_name = img_path[0].split("/")[-1]
                    output_path = (
                        self.report_directory
                        + "/Pre_execution_test_data/"
                        + img_name
                        + "_px_app_"
                        + method.__name__
                        + img_path[1]
                    )
                    # Process image and return path of processed image
                    result = pixel_apply(test_img_path, method, "spec", output_path, progress=False)
                elif dl_in == 2:
                    # Output dst image path
                    img_path = os.path.splitext(str(test_img_path).replace("\\", "/").replace("//", "/"))
                    img_name = img_path[0].split("/")[-1]
                    output_path = (
                        self.report_directory
                        + "/Pre_execution_test_data/"
                        + img_name
                        + "_px_app_"
                        + method.__name__
                        + img_path[1]
                    )
                    # Process image and return path of processed image
                    result = pixel_apply(test_img_path, method, "array", output_path, progress=False)
                elif dl_in == 3:
                    # Output dst image path
                    img_path = os.path.splitext(str(test_img_path).replace("\\", "/").replace("//", "/"))
                    img_name = img_path[0].split("/")[-1]
                    output_path = (
                        self.report_directory
                        + "/Pre_execution_test_data/"
                        + img_name
                        + "_px_app_"
                        + method.__name__
                        + img_path[1]
                    )
                    # Process image and return path of processed image
                    result = pixel_apply(test_img_path, method, "tensor", output_path, progress=False)
                elif dl_in == 4:
                    # Output dst image path
                    img_path = os.path.splitext(str(test_img_path).replace("\\", "/").replace("//", "/"))
                    img_name = img_path[0].split("/")[-1]
                    output_path = (
                        self.report_directory
                        + "/Pre_execution_test_data/"
                        + img_name
                        + "_px_app_"
                        + method.__name__
                        + img_path[1]
                    )
                    # Process image and return path of processed image
                    result = pixel_apply(test_img_path, method, "tensor_hyper", output_path, progress=False)
                elif dl_in == 5:
                    result = method(test_img_path, self._pretest_data["roi_coords"])
                elif dl_in == 6:
                    testing_data = self._pretest_data["roi_specs"]
                    result = method(testing_data)
                elif dl_in == 7:
                    testing_data = self._pretest_data["spec1d"]
                    result = method(testing_data)
                else:
                    raise ValueError("Input data level cannot be 'model' or 8 (corresponding index).")
            else:
                # Model method is not validated here
                return method

            # Output validation
            if result is None:
                raise ValueError(
                    f"Method '{method.__name__}' returns no data. \
                        The added method must have a return. \
                            For image processing methods, absolute path of resulting image must be returned."
                )

            # For raster image path and image file output
            if dl_out <= 4:
                # Raster file validation
                if os.path.exists(unc_path(result)):
                    # Open raster validation
                    try:
                        with rasterio.open(result) as src:
                            # Raster validation
                            if src is None:
                                raise ValueError("Invalid raster: raster is None.")
                            elif (src.width == 0) or (src.height == 0) or (src.count == 0):
                                raise ValueError(
                                    f"Invalid raster, \
                                        got dimensions: {src.width} x {src.height}, got number of bands: {src.count}."
                                )
                            else:
                                # Raster value validation
                                all_no_data = True
                                sample = src.read(min(int(src.count / 2), 1))
                                if np.all(sample == src.nodata) or np.all(np.isnan(sample)):
                                    sample = src.read(
                                        window=(
                                            (
                                                max(int(src.height / 2) - 1, 0),
                                                min(int(src.height / 2) + 1, src.height),
                                            ),
                                            (
                                                max(int(src.width / 2) - 1, 0),
                                                min(int(src.width / 2) + 1, src.width),
                                            ),
                                        )
                                    )
                                    if np.all(sample == src.nodata) or np.all(np.isnan(sample)):
                                        for i in range(0, src.height, 32):
                                            for j in range(0, src.width, 32):
                                                # Define window for current tile
                                                win = Window(
                                                    col_off=j,
                                                    row_off=i,
                                                    width=min(32, src.width - j),
                                                    height=min(32, src.height - i),
                                                )
                                                # Read all bands for current tile (shape: [bands, rows, cols])
                                                sample = src.read(window=win)
                                                if not (np.all(sample == src.nodata) or np.all(np.isnan(sample))):
                                                    all_no_data = False
                                                    break
                                    else:
                                        all_no_data = False
                                else:
                                    all_no_data = False
                                if all_no_data:
                                    raise ValueError("All raster values are NoData")
                    except Exception as e:
                        raise ValueError(
                            f"Failed to open resulting raster image of {method.__name__}.\
                                \nGot path:\n{result}, \nError msg:{e}"
                        ) from e
                else:
                    raise ValueError(f"Resulting file path is invalid: {result}")

            # For array-like output
            if (dl_out >= 6) & (dl_out <= 7):
                result = arraylike_validator()(result)
                if type(result) is np.ndarray:
                    if np.issubdtype(result.dtype, np.number):
                        if (dl_out == 6) and (result.ndim != 2):
                            raise ValueError(
                                f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' \
                                    must return an 2D array, got array dimension: {result.ndim}"
                            )
                        else:
                            result = np.array(result)
                            if (dl_out == 7) and (result.ndim != 1):
                                raise ValueError(
                                    f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' \
                                        must return an 1D array-like, got array dimension: {result.ndim}"
                                )
                    else:
                        raise ValueError(
                            f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' \
                                must return an array of numbers, got array dtype: {result.dtype}"
                        )
                else:
                    raise TypeError(
                        f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' \
                            must return an NumPy array-like, got: {type(result)}"
                    )

            return method

        else:
            # Validate data_level
            dl_in = _dl_val(input_data_level)[0]
            dl_out = _dl_val(output_data_level)[0]
            if dl_in != 7:
                raise ValueError(
                    f"Method for one-dimensional standalone spectra must have input data level of 7 ('spec1d'), \
                        but got: {input_data_level}"
                )
            if dl_out < 7:
                raise ValueError(
                    f"Method for one-dimensional standalone spectra cannot have output data level below 7 ('spec1d'), \
                        but got level number: {dl_out}"
                )
            if dl_out == 8:
                # Model method is not validated here
                return method

            testing_data = self._pretest_data["spec1d"]
            result = method(testing_data)

            # Output validation
            if result is None:
                raise ValueError(
                    f"Method '{method.__name__}' returns no data. The added method must have a return. \
                        For image processing methods, absolute path of resulting image must be returned."
                )

            # For array-like output
            result = arraylike_validator()(result)
            if type(result) is np.ndarray:
                if np.issubdtype(result.dtype, np.number):
                    result = np.array(result)
                    if (dl_out == 7) and (result.ndim != 1):
                        raise ValueError(
                            f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' \
                                must return an 1D array-like, got array dimension: {result.ndim}"
                        )
                else:
                    raise ValueError(
                        f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' \
                            must return an array of numbers, got array dtype: {result.dtype}"
                    )
            else:
                raise TypeError(
                    f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' \
                        must return an NumPy array-like, got: {type(result)}"
                )

            return method

    # Add model - from add_process
    @simple_type_validator
    def add_model(
        self,
        model_method: object,
        model_label: Optional[str] = None,
        # Process parameters
        test_error_raise: bool = True,
        # Modeling parameters
        is_regression: Optional[bool] = None,
        validation_method: str = "2-fold",
        unseen_threshold: Optional[float] = 0.0,
        x_shape: Optional[tuple[int]] = None,
        result_backup: bool = False,
        # Model validation and evaluation configurations
        data_split_config: Union[str, dict[str, Any]] = "default",
        validation_config: Union[str, dict[str, Any]] = "default",
        metrics_config: Union[str, dict[str, Any], None] = "default",
        roc_plot_config: Union[str, dict, None] = "default",
        scatter_plot_config: Union[str, dict[str, Any], None] = "default",
        residual_config: Union[str, dict[str, Any], None] = "default",
        residual_plot_config: Union[str, dict[str, Any], None] = "default",
        influence_analysis_config: Union[str, dict[str, Any], None] = "default",
    ) -> None:
        """
        Add a model to the pipeline.
        The model process must accept 1D data (7 / 'spec1d') and has a output data level of 8 / 'model'.
        All models have same application sequence.


        Parameters
        ----------
        ** Model adding parameters **
        - model_method : sklearn-style model object
            Sklearn-style model object.
            The regressor class must have methods of 'fit' and 'predict'.
            The classifier class must have methods of 'predict_proba'.

        - model_label : str, optional
            Custom label for the added model. The default is ''.

        - test_error_raise : bool, optional
            The model is validated on highly simplified mock data before being added to the pipeline.
            If True, an error is raised if the validation fails. If False, the error is suppressed and only a warning is issued.
            The default is True. Set to False if you are confident the model will work with your actual data despite a validation failure.


        ** Modeling parameters **
        - is_regression : Optional[bool]
            Whether regression model is applied.
            If None, it is automatically determined using the type of target values in the sample list. The default is None.

        - validation_method : str
            Validation method, Choose between:
            - "loo" :
                Leave-one-out cross-validation.
            - "k-fold" (e.g. "5-fold") :
                K-fold cross validation, k is the number of folds.
            - "m-n-split" (e.g. "70-30-split") :
                Train-test split. m% training, n% testing (only test set used for evaluation).
            The default is '2-fold'.

        - unseen_threshold : float = 0.0
            For classification models trained on data with missing classes, a sample is assigned to a unknown class if its highest predicted probability among the known classes is below the unseen_threshold.
            No use for regression.

        - x_shape: Optional[tuple[int]]
            Shape of independent variables if the method use shaped sample data.
            In the current version, the parameter has no effect.

        - result_backup : bool
            Whether copies of result files are saved, if True, copy files with modeling time are saved in addition.


        ** Model validation and evaluation configurations **
        - data_split_config : Union[str,dict], optional
            Configuration of data_split options in dictionary. The type of data spliting is specified by the 'ModelEva' object.
            The default is 'default'.
            The parameters of data split include:
            - random_state : Optional[int], optional
                Random state for data splitting and shuffling. If None, random_state is not fixed.

        - validation_config : Union[str,dict], optional
            Configuration of validation options. The default is 'default', using default settings.

            The parameters of validation include:
            - unseen_threshold : Optional[float]
                For classification models trained on data with missing classes, a sample is assigned to a unknown class if its highest predicted probability among the known classes is below the unseen_threshold.
                Regression validation_config does not have this argumente.
            - use_original_shape : bool
                Whether the sample data is reshaped to its original shape.
                If False, the flattened data is used. The default is False.
            - save_fold_model: bool
                Whether to save model of each fold. The default is True.
            - save_fold_data: bool
                Whether to save training and validation data of each fold. The default is True.
                Saving fold data and fold models could consuming significant storage when applied to large data with large sample size with large fold numbers.

        - metrics_config : Union[str,dict,None], optional
            Configuration of metrics options. The default is 'default', using default settings.
            If None, the performance metrics computation is skipped.

        - roc_plot_config : Union[str,dict,None], optional
            For classification only. Configuration of plotting Response Operating Characteristics curves in dictionary.
            The default is 'default'.
            The parameters of scatter_plot and its default values are listed as follows:
            - plot_title : str = 'ROC Curve',
            - title_size : Union[int,float] = 26,
            - title_pad : Union[int,float,None] = None,
            - figure_size : tuple[Union[int,float],Union[int,float]] = (8, 8),
            - plot_margin : tuple[float,float,float,float] = (0.15, 0.95, 0.9, 0.13), # (left,right,top,bottom) Margin
            - plot_line_with : Union[int,float] = 3,
            - plot_line_alpha : float = 0.8,
            - diagnoline_width : Union[int,float] = 3,
            - x_axis_limit : Optional[tuple[Union[int,float],Union[int,float]]] = None,
            - x_axis_label : str = 'False Positive Rate',
            - x_axis_label_size : Union[int,float] = 26,
            - x_tick_size : Union[int,float] = 24,
            - x_tick_number : int = 6,
            - y_axis_limit : Optional[tuple[Union[int,float],Union[int,float]]] = None,
            - y_axis_label : str = 'True Positive Rate',
            - y_axis_label_size : Union[int,float] = 26,
            - y_tick_size : Union[int,float] = 24,
            - y_tick_number : int = 6,
            - axis_line_size_left : Union[int,float,None] = 1.5,
            - axis_line_size_right : Union[int,float,None] = 1.5,
            - axis_line_size_top : Union[int,float,None] = 1.5,
            - axis_line_size_bottom : Union[int,float,None] = 1.5,
            - legend : bool = True,
            - legend_location : str = 'lower right',
            - legend_fontsize : Union[int,float] = 20,
            - legend_title : str = '',
            - legend_title_fontsize : Union[int,float] = 24,
            - background_grid : bool = False,
            - show_plot : bool = False
            None refers to default values.

        - scatter_plot_config : Union[str,dict,None], optional
            For regression only. Configuration of scatter_plot options in dictionary.
            The default is 'default'.
            The parameters of scatter_plot and its default values are listed as follows:
            - plot_title : str = '',
            - title_size : Union[int,float] = 26,
            - title_pad : Union[int,float,None] = None,
            - figure_size : tuple[Union[int,float],Union[int,float]] = (8, 8),
            - plot_margin : tuple[float,float,float,float] = (0.2, 0.95, 0.95, 0.15), # (left,right,top,bottom) Margin
            - plot_line_with : Union[int,float] = 3,
            - point_size : Union[int,float] = 120,
            - point_color : str = 'firebrick',
            - point_alpha : float = 0.7,
            - x_axis_limit : Optional[tuple[Union[int,float],Union[int,float]]] = None,
            - x_axis_label : str = 'Predicted target values',
            - x_axis_label_size : Union[int,float] = 26,
            - x_tick_values : Optional[list[Union[int,float]]] = None,
            - x_tick_size : Union[int,float] = 24,
            - x_tick_number : int = 5,
            - y_axis_limit : Optional[tuple[Union[int,float],Union[int,float]]] = None,
            - y_axis_label : str = 'Residuals',
            - y_axis_label_size : Union[int,float] = 26,
            - y_tick_values : Optional[list[Union[int,float]]] = None,
            - y_tick_size : Union[int,float] = 24,
            - y_tick_number : int = 5,
            - axis_line_size_left : Union[int,float,None] = 1.5,
            - axis_line_size_right : Union[int,float,None] = 1.5,
            - axis_line_size_top : Union[int,float,None] = 1.5,
            - axis_line_size_bottom : Union[int,float,None] = 1.5,
            - background_grid : bool = False,
            - show_plot : bool = False

        - residual_config : Union[str,dict[str,Any],None]
            Configuration of residual analysis options. The default is 'default', using default settings.
            If None, the residual analysis is skipped.

        - residual_plot_config: Union[str,dict,None], optional
            For regression only. Configuration of residual plot parameters is same as scatter_plot_config.
            See 'scatter_plot_config' for details.
            The parameters is same as 'scatter_plot'.

        - influence_analysis_config : Union[str,dict,None], optional
            Configuration of influence analysis. The default is 'default', using default settings.
            If None, Influence analysis is skipped.

            When enabled, calculates the Cook's distance-like influence of each sample on the model's predictions using a Leave-One-Out (LOO) approach.
            Please note this computation is highly time-consuming for large sample size. To save time, use a simple validation method or set this to None.

            The parameters of validation include:
            - validation_method : bool, optional
                Independent validation_method for leave-one-out analysis of data point influence.
                Default is using model validation method if it is train-test split, and "2-fold" if the model validation method is "k-fold" or "loo".
            - random_state : None, optional
                random state for data splitting. If None, the random state is not fixed. The default is None.
        """  # noqa: E501

        # Process general parameters
        input_data_level = 7
        output_data_level = 8
        # Application sequence of non-model processes with input data level spec1d / 7
        preprocess_dl7 = self.ls_process(input_data_level=7, return_result=True, print_result=False)
        app_seqs = [proci[4] for proci in np.array(preprocess_dl7[preprocess_dl7["Output_data_level"] != "model"])]
        if len(app_seqs) > 0:
            application_sequence = max(app_seqs) + 1
        else:
            application_sequence = 0
        method = model_method
        process_label = ""
        # No use for model data level, model_label is applied for model data level in add_process istead

        # Add model process
        self.add_process(
            input_data_level=input_data_level,
            output_data_level=output_data_level,
            application_sequence=application_sequence,
            method=method,
            process_label=process_label,
            test_error_raise=test_error_raise,
            # Modeling parameters
            is_regression=is_regression,
            validation_method=validation_method,
            model_label=model_label,
            unseen_threshold=unseen_threshold,
            x_shape=x_shape,
            result_backup=result_backup,
            # Model evaluation parameters
            data_split_config=data_split_config,
            validation_config=validation_config,
            metrics_config=metrics_config,
            roc_plot_config=roc_plot_config,
            scatter_plot_config=scatter_plot_config,
            residual_config=residual_config,
            residual_plot_config=residual_plot_config,
            influence_analysis_config=influence_analysis_config,
        )

    # List model - from ls_process
    @overload
    def ls_model(
        self,
        model_id: Optional[str] = None,
        model_label: Optional[str] = None,
        model_method: Union[str, object, None] = None,
        *,
        exact_match: bool = True,
        print_result: bool = True,
        return_result: Literal[False] = False,
    ) -> None: ...

    @overload
    def ls_model(
        self,
        model_id: Optional[str] = None,
        model_label: Optional[str] = None,
        model_method: Union[str, object, None] = None,
        *,
        exact_match: bool = True,
        print_result: bool = True,
        return_result: Literal[True] = True,
    ) -> pd.DataFrame: ...

    # List model - from ls_process
    @simple_type_validator
    def ls_model(
        self,
        model_id: Optional[str] = None,
        model_label: Optional[str] = None,
        model_method: Union[str, object, None] = None,
        *,
        exact_match: bool = True,
        print_result: bool = True,
        return_result: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        List model process items filtered by given properties.
        If a property is not specified, the property will be ignored in filtering.
        If return_df is set True, a complete dataframe of processes is returned, if False, a simplified dataframe is printed.

        Parameters
        ----------
        model_id : str, optional
            Model (process) ID. The default is None.

        model_label : str, optional
            Custom model_label. The default is None.

        model_method : Union[str,object], optional
            Model object. The default is None.

        exact_match : bool, optional
            If False, any process with property value containing the specified value is removed.
            The default is True.

        print_result : bool, optional
            Whether simplified results are printed. The default is True.

        return_result : bool, optional
            Whether a complete resulting dataframe is returned. The default is False.
        """  # noqa: E501
        # List models
        if return_result:
            return self.ls_process(
                process_id=model_id,
                process_label=model_label,
                input_data_level="spec1d",
                output_data_level="model",
                application_sequence=None,
                method=model_method,
                full_application_sequence=None,
                exact_match=exact_match,
                print_result=print_result,
                return_result=True,
            )
        else:
            self.ls_process(
                process_id=model_id,
                process_label=model_label,
                input_data_level="spec1d",
                output_data_level="model",
                application_sequence=None,
                method=model_method,
                full_application_sequence=None,
                exact_match=exact_match,
                print_result=print_result,
                return_result=False,
            )
            return None

    # Remove model - from rm_process
    @simple_type_validator
    def rm_model(
        self,
        model_id: Optional[str] = None,
        model_label: Optional[str] = None,
        model_method: Union[str, object, None] = None,
        exact_match: bool = True,
    ) -> None:
        """
        Remove added models by model_id, model_label and model method (object).
        If a property is not specified, the property criterion will be ignored.

        Parameters
        ----------
        model_id : str, optional
            Model (process) ID. The default is None.
        model_label : str, optional
            Custom model label. The default is None.
        model_method : Union[str,Callable,object], optional
            Method object. The default is None.
        exact_match : bool, optional
            If False, any process with property value containing the specified value is removed.
            The default is True.
        """
        self.rm_process(
            process_id=model_id,
            process_label=model_label,
            input_data_level=7,
            output_data_level=8,
            application_sequence=None,
            method=model_method,
            exact_match=exact_match,
        )

    # Add process
    # Format of associated attribute:
    # [0 Process_ID, 1 Process_label, 2 input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    @simple_type_validator
    def add_process(  # noqa: C901
        self,
        # Process general parameters
        input_data_level: Union[str, int],
        output_data_level: Union[str, int],
        application_sequence: int,
        method: Union[Callable, object],
        process_label: str = "",
        *,
        test_error_raise: bool = True,
        # Modeling parameters
        is_regression: Optional[bool] = None,
        validation_method: str = "2-fold",
        model_label: Optional[str] = None,
        unseen_threshold: Optional[float] = 0.0,
        x_shape: Optional[tuple[int]] = None,
        result_backup: bool = False,
        # Model evaluation parameters
        data_split_config: Union[str, dict[str, Any]] = "default",
        validation_config: Union[str, dict[str, Any]] = "default",
        metrics_config: Union[str, dict[str, Any], None] = "default",
        roc_plot_config: Union[str, dict, None] = "default",
        scatter_plot_config: Union[str, dict[str, Any], None] = "default",
        residual_config: Union[str, dict[str, Any], None] = "default",
        residual_plot_config: Union[str, dict[str, Any], None] = "default",
        influence_analysis_config: Union[str, dict[str, Any], None] = "default",
    ) -> None:
        """
        Add a processing method with defined input/output data levels and application sequence to the pipeline.
        A processing method can be a preprocessing function or a model for evaluation.


        Parameters
        ----------
        ** Process general parameters **
        - input_data_level : Union[str, int]
            Input data level for the process method:
                0 or 'image' :
                    Method directly applied to raster images.
                    The function must accepts input raster image path as the first parameter and output raster image path as the second parameter.

                1 or 'pixel_spec' :
                    Method applied to the 1D spectra of every pixel of the raster image, accepting pixel 1D spectra as input.

                2 or 'pixel_specs_array' :
                    Method applied to the 1D spectra of every pixel of the raster image, accepting 2D spectral NumPy array as input, where each data row represents a pixel spectrum.

                3 or 'pixel_specs_tensor' :
                    Method applied to the 1D spectra of every pixel of the raster image, accepting 3D spectral PyTorch Tensor as input, whose shape is [C, H, W].
                    C - Channel (Bands), H - Height (Raster rows), W - Width (Raster columns). The calculation must be performed along axis 0 (Channel axis).

                4 or 'pixel_hyperspecs_tensor' :
                    Method applied to the 1D spectra of every pixel of the raster image, accepting pixel 3D hyperspectral PyTorch Tensor as input, optimized for hyperspectral image data.
                    The input shape must be [C, H, W] (see 'pixel_specs_tensor'), but the calculation must be performed along axis 1.

                5 or 'image_roi' :
                    Method applied to Region of Interest (ROI) of raster image, accepting raster image path and ROI coordinates from 'SpecExp' as input.

                6 or 'roi_specs' :
                    Method applied to 2D array of spectra or spectral measures in a sample ROI, where each row represents the data of a pixel.

                7 or 'spec1d' :
                    Method applied to 1D spectral data or flattened 1D spectral data and accepts 1D array-like.

            Methods are executed in ascending order of their input_data_level.
            If the input_data_level > 5, the input_data_level must equal to the output_data_level of the preceding process.

        - output_data_level : Union[str, int]
            Output data level:

                0 or 'image' :
                    If method outputs a raster image file and returns its path.

                1 or 'pixel_spec' :
                    Output same as image, while the input data level is 'pixel_spec'.

                2 or 'pixel_specs_array' :
                    Output same as image, while the input data level is 'pixel_specs_array'.

                3 or 'pixel_specs_tensor' :
                    Output same as image, while the input data level is 'pixel_specs_tensor'.

                4 or 'pixel_hyperspecs_tensor' :
                    Output same as image, while the input data level is 'pixel_hyperspecs_tensor'.

                5 or 'image_roi' :
                    Currently not available.

                6 or 'roi_specs' :
                    If the method output a 2D array of spectra or spectral measures, where each row represents a spectrum.

                7 or 'spec1d' :
                    If the method output a 1D array-like spectral data or flattened 1D spectral data.

                8 or 'model' :
                    If the method is used for modeling and outputs model evaluation reports. The model method only accepts 'spec1d' level input data. Image data can be flattened and provided with shape.

            Methods with identical input_data_level and application_sequence must have identical output_data_level.
            If the output_data_level > 5, the output_data_level must equal to the input_data_level of the following process.

        - application_sequence : int
            Application sequence number of a method in an input data level.
            If two methods have identical input data levels, method with lower application sequence number is applied first.

        - method : Callable or sklearn-style model object
            Method function or model object. The input and output should be defined according to the input and output data levels, see 'input_data_level' and 'output_data_level'.
            For modeling, this parameter accepts a sklearn-style model object with method 'fit', 'predict' and 'predict_proba' for classifier.

        - process_label : str, optional
            Custom label for the added process. The default is ''.

        - test_error_raise : bool, optional
            The process method is validated on highly simplified mock data before being added to the pipeline.
            If True, an error is raised if the validation fails. If False, the error is suppressed and only a warning is issued.
            The default is True. Set to False if you are confident the method will work with your actual data despite a validation failure.


        ** Modeling parameters **
        See SpecPipe.add_model.


        ** Model evaluation parameters **
        See SpecPipe.add_model.

        """  # noqa: E501

        # Validate Data_level
        dl_in = _dl_val(input_data_level)
        dl_in_name = dl_in[1]
        dl_in_ind = dl_in[0]

        dl_out = _dl_val(output_data_level)
        dl_out_name = dl_out[1]
        dl_out_ind = dl_out[0]

        # Full application sequence
        fapp_seq = 1000000 * dl_in_ind + application_sequence

        # Validate data levels and data level sequence
        _data_level_seq_validator(
            input_data_level=input_data_level,
            output_data_level=output_data_level,
            application_sequence=application_sequence,
            full_application_sequence=fapp_seq,
            existed_process=self.process,
        )

        # Generate process number
        existed_proc_num = [0]
        if len(self.process) > 0:
            for pr in self.process:
                # Get existed process number (repeat number)
                if (pr[2] == dl_in_name) & (pr[4] == application_sequence):
                    existed_proc_num.append(pr[7])
        proc_num_new = max(existed_proc_num) + 1

        # Build process ID
        proc_id = str(dl_in_ind) + "_" + str(application_sequence) + "_%#" + str(proc_num_new)

        # Add preprocess - validate preprocess method
        if dl_out_ind < 8:
            try:
                proc_method = _process_validator(
                    method=method,
                    input_data_level=input_data_level,
                    output_data_level=output_data_level,
                    pretest_data=self.pretest_data,
                    standalone_specs_sample=self.spec_exp.standalone_specs_sample,
                    report_directory=self.report_directory,
                )
            except Exception as e:
                if test_error_raise:
                    raise ValueError(f"Method testing fails: \n{e}") from e
                else:
                    warnings.warn(
                        f"Method fails on '{input_data_level}' testing data: \n{e}", UserWarning, stacklevel=2
                    )

        # Add model - model method constructor
        else:
            # Get report dir
            report_dir = self.report_directory

            # Validate model labels
            existed_model_labels = [proc[1] for proc in self.process if proc[3] == 'model' and proc[1] != ''] + [
                proc[5].model_label for proc in self.process if proc[3] == 'model' and hasattr(proc[5], 'model_label')
            ]
            if model_label is None:
                # Validate for duplication
                model_name = method.__class__.__name__
                name_k = 0
                while model_name in existed_model_labels:
                    name_k = name_k + 1
                    model_name = method.__class__.__name__ + f"_{name_k}"
                model_label = model_name
            else:
                # Validate for duplication
                name_k = 0
                while model_label in existed_model_labels:
                    name_k = name_k + 1
                    model_label = method.__class__.__name__ + f"_{name_k}"
                process_label = model_label

            # Validate is_regression
            if is_regression is None:
                try:
                    self.sample_targets[0][2] + 1  # type: ignore[operator]
                    # Behavior-based type check after serialization, safe approach for serialization
                    is_regression = True
                except Exception:
                    is_regression = False

            # Validate model instance
            if is_regression:
                _regressor_validator(method)
            else:
                _classifier_validator(method)

            # Generate process method for model evaluation
            proc_method = _ModelMethod(
                model_label=model_label,
                is_regression=is_regression,
                x_shape=x_shape,
                report_dir=report_dir,
                method=method,
                result_backup=result_backup,
                validation_method=validation_method,
                unseen_threshold=unseen_threshold,
                data_split_config=data_split_config,
                validation_config=validation_config,
                metrics_config=metrics_config,
                roc_plot_config=roc_plot_config,
                scatter_plot_config=roc_plot_config,
                residual_config=residual_config,
                residual_plot_config=residual_plot_config,
                influence_analysis_config=influence_analysis_config,
            )

        # Change process test status
        self._tested = False

        # Add process item
        proc_item = (
            proc_id,
            process_label,
            dl_in_name,
            dl_out_name,
            application_sequence,
            proc_method,
            fapp_seq,
            proc_num_new,
        )
        self._process.append(proc_item)
        # Sort processes and update process_steps and process_chains
        self._generate_process_steps()
        self._generate_process_chains()

    # Sort processes
    # Format of associated attribute:
    # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    def _sort_proc(self) -> None:
        """
        Sort added processes
        """
        if len(self._process) > 1:
            # Sort processes
            proc_df = pd.DataFrame(
                self._process,
                columns=[
                    "Process_ID",
                    "Process_label",
                    "Input_data_level",
                    "Output_data_level",
                    "Application_sequence",
                    "Method_callable",
                    "Full_app_seq",
                    "Alternative_number",
                ],
            )
            proc_sorted = proc_df.sort_values(by=["Full_app_seq", "Alternative_number"])
            proc_sorted = np.array(proc_sorted)
            proc = [tuple(pit) for pit in proc_sorted]
            # Update processes
            self._process = proc

    # Generate process steps
    # Format of associated attribute:
    # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    def _generate_process_steps(self) -> None:
        """
        Generates process steps of Process_IDs, representing a sequential workflow of added process items.
        Length of the chain list represents the total number of proccessing steps.
        """
        if len(self._process) > 0:
            # Sort process
            self._sort_proc()
            # Grow a process tree
            pchain = []
            fseq = -1
            pti = -1
            p_item: tuple[str, str, str, str, int, Callable, int, int]
            for p_item in self._process:
                if p_item[6] != fseq:
                    pchain.append([p_item[0]])
                    fseq = p_item[6]
                    pti = pti + 1
                else:
                    pchain[pti].append(p_item[0])
            # Update tree
            self._process_steps = pchain
        else:
            raise ValueError("No process added")

    # Number to mixed radix vector
    @simple_type_validator
    def _num_to_mixed_radix(self, number: int, shape: Union[list[int], tuple[int]]) -> list[int]:
        """
        Converts a given integer number into the index of a mixed-radix numeral system with given shape
        """
        digits = []
        for radix in reversed(shape):
            digits.append(number % radix)
            number = number // radix
        return list(reversed(digits))

    # Generate process chains
    def _generate_process_chains(self) -> None:
        """
        Generates a list of all actual chains of added process items in execution.
        Each chain corresponds to an unique final results.
        """
        process_steps = self._process_steps
        # Get number of chains and chain shape
        nchain = 1
        chain_shape = []
        for proc_ids in process_steps:
            nids = len(proc_ids)
            chain_shape.append(nids)
            nchain = nchain * nids
        # Fill the chain table
        pcs = np.full((nchain, len(chain_shape)), np.nan, dtype="O")
        for i in range(pcs.shape[0]):
            sidi = self._num_to_mixed_radix(i, chain_shape)
            for j in range(pcs.shape[1]):
                pcs[i, j] = process_steps[j][sidi[j]]
        # Update process chains
        self._process_chains = [tuple(pc) for pc in pcs]

    # Generate process id - label reference table
    def _process_id_label_ref(self) -> np.ndarray:
        """Generate process id - label reference table, return the table [proc_id, label]"""
        ref_table = []
        for proc in self.process:
            if proc[1] != "":
                label_name = proc[1]
            else:
                label_name = proc[5].__name__
            ref_table.append((proc[0], label_name))
        result: np.ndarray = np.array(ref_table)
        return result

    # Convert process ID to process label
    def _process_id_to_label(self, process_id: str, ref_table: np.ndarray) -> str:
        """Convert process ID to process label"""
        proc_label_item = ref_table[ref_table[:, 0] == process_id]
        if len(proc_label_item) == 1:
            return str(proc_label_item[0, 1])
        elif len(proc_label_item) == 0:
            raise ValueError(f"Got invalid process ID '{process_id}', no corresponding process label found.")
        else:
            raise ValueError(
                f"Process ID '{process_id}' has multiple label references: {proc_label_item[:, 1]}. \
                    This indicates corrupted process data."
            )

    # List process chains
    @simple_type_validator
    def ls_process_chains(
        self, print_label: bool = True, return_label: bool = False
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame], None]:
        """
        List process chains.

        Returns the default full factorial process chains.
        For custom chains, use 'ls_custom_chains'.

        Output is a dataframe, where each row represents a processing chain with process IDs.

        Parameters
        ----------
        print_label : bool, optional
            If True, prints chains using chain label. The default is True.
        return_label : bool, optional
            If True, return additional dataframe of process labels. The default is False.
        """
        if len(self._process_chains) > 0:
            df_chains = pd.DataFrame(
                self._process_chains,
                columns=["Step_" + str(i) for i in range(len(self._process_chains[0]))],
            )
            if return_label or print_label:
                df_chains_label = copy.deepcopy(df_chains)
                ref_table = self._process_id_label_ref()
                for i in range(df_chains.shape[0]):
                    for j in range(df_chains.shape[1]):
                        df_chains_label.iloc[i, j] = self._process_id_to_label(df_chains.iloc[i, j], ref_table)
                if print_label:
                    with pd.option_context("display.max_rows", None, "display.max_columns", None):
                        print(df_chains_label)
                if return_label:
                    return df_chains, df_chains_label
                else:
                    return df_chains
            else:
                return df_chains
        else:
            print("No process chain found")
            return None

    # Alias
    process_chains_to_df = ls_process_chains

    # Save pipeline configurations
    @simple_type_validator
    def save_pipe_config(self, copy: bool = False, save_spec_exp_config: bool = True) -> None:
        """
        Save current pipeline configurations to files in the root of report directory.

        Parameters
        ----------
        copy : bool, optional
            Whether to create a backup copy of configuration files. The default is True.
        save_spec_exp_config : bool, optional
            Whether to save data configurations of SpecExp. The default is True.
        """
        # Create save dir
        report_dir = self.report_directory + "SpecPipe_configuration/"
        if not os.path.exists(unc_path(report_dir)):
            os.makedirs(unc_path(report_dir))

        # Get configs
        df_process = self.ls_process(print_result=False, return_result=True)
        df_full_chains, df_full_chains_label = self.ls_process_chains(print_label=False, return_label=True)
        df_exec_chains, df_exec_chains_label = self.ls_chains(print_label=False, return_label=True)

        # Save configs
        df_process.to_csv(unc_path(report_dir + "SpecPipe_added_process.csv"), index=False)
        df_full_chains.to_csv(unc_path(report_dir + "SpecPipe_full_factorial_chains_in_ID.csv"), index=False)
        df_full_chains_label.to_csv(unc_path(report_dir + "SpecPipe_full_factorial_chains_in_label.csv"), index=False)
        df_exec_chains.to_csv(unc_path(report_dir + "SpecPipe_exec_chains_in_ID.csv"), index=False)
        df_exec_chains_label.to_csv(unc_path(report_dir + "SpecPipe_exec_chains_in_label.csv"), index=False)

        # Save SpecPipe
        with open(unc_path(f"{report_dir}SpecPipe_pipeline_configuration_{self.create_time}.dill"), 'wb') as f:
            dill.dump(self, f)

        # Save copies
        if copy:
            # Prevent duplication
            time.sleep(1.0)
            # Dump copy
            cts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            df_process.to_csv(unc_path(report_dir + f"SpecPipe_added_process_{cts}.csv"), index=False)
            df_full_chains.to_csv(unc_path(report_dir + f"SpecPipe_full_factorial_chains_in_ID_{cts}.csv"), index=False)
            df_full_chains_label.to_csv(
                unc_path(report_dir + f"SpecPipe_full_factorial_chains_in_label_{cts}.csv"), index=False
            )
            df_exec_chains.to_csv(unc_path(report_dir + f"SpecPipe_exec_chains_in_ID_{cts}.csv"), index=False)
            df_exec_chains_label.to_csv(unc_path(report_dir + f"SpecPipe_exec_chains_in_label_{cts}.csv"), index=False)
            with open(
                unc_path(report_dir + f"SpecPipe_pipeline_configuration_{self.create_time}_copy_at_{cts}.dill"), 'wb'
            ) as f:
                dill.dump(self, f)

        # Save SpecExp
        if save_spec_exp_config:
            self.spec_exp.save_data_config(copy=copy)

        # Print output path
        print(
            f"\nSpecPipe configurations saved to: \
              {report_dir}SpecPipe_pipeline_configuration_{self.create_time}.dill\n"
        )

    # Alias
    save_config = save_pipe_config

    # Load pipeline configurations
    @simple_type_validator
    def load_pipe_config(self, config_file_path: str = "") -> None:
        """
        Load SpecPipe configurations from dill file.

        Parameters
        ----------
        config_file_path : str, optional
            Configuration file path of 'SpecExp_data_configuration_(creating-time).dill'.
            The path can be absolute path of the dill file or its relative path to report directory.
            If not given, the path will be '(SpecExp.report_directory)/SpecExp_configuration/SpecExp_data_configuration_(SpecExp.create_time).dill'
        """  # noqa: E501
        # Load path
        if config_file_path == "":
            dump_path0 = (
                self.report_directory
                + "SpecPipe_configuration/"
                + f"SpecPipe_pipeline_configuration_{self.create_time}.dill"
            )
        elif ("/" not in config_file_path) & ("\\" not in config_file_path):
            dump_path0 = self.report_directory + "SpecPipe_configuration/" + config_file_path
        else:
            dump_path0 = config_file_path

        # Load to instance
        with open(unc_path(dump_path0), 'rb') as f:
            loaded_instance = dill.load(f)
        self.__dict__.update(loaded_instance.__dict__)

    # Alias
    load_config = load_pipe_config

    # Read process chains from dataframe
    @simple_type_validator
    def custom_chains_from_df(
        self,
        # process_chain_dataframe: Annotated[Any, AfterValidator(dataframe_validator())]
        process_chain_dataframe: Annotated[Any, dataframe_validator()],
    ) -> None:
        """
        Customize processing chains and update chains using chain dataframe.

        Once custom chains are created, SpecPipe will prioritize their execution, bypassing the original full chains.

        Read custom process chains from process chain dataframe:
            - Columns = ['Step_1', 'Step_2', ...] and the length should be identical as the column length of process_chains.
            - All values should be valid process IDs

        It is recommended to modify the dataframe from method 'process_chains_to_df' and set the modified dataframe as custom chains.
        """  # noqa: E501
        # Validate chain df
        process_chain_dataframe = dataframe_validator(dtype="str", ncol=len(self._process_chains[0]))(
            process_chain_dataframe
        )

        # Convert chain df to list
        cchain = [tuple(row) for row in process_chain_dataframe.to_numpy()]
        full_chain = self._process_chains

        # Validate given custom chains
        for ind, ccr in enumerate(cchain):
            if ccr not in full_chain:
                raise ValueError(
                    f"\nInvalid process chain in given chains: \n{ccr}, \nRow index of invalid chain: {ind}"
                )

        # Change process test status
        self._tested = False

        # Update
        self._custom_chains = cchain

    # List custom process chains
    @simple_type_validator
    def ls_custom_chains(
        self, print_label: bool = True, return_label: bool = False
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame], None]:
        """
        List custom process chains.

        Output is a dataframe, where each row represents a processing chain with process IDs.

        Parameters
        ----------
        print_label : bool, optional
            If True, prints chains using chain label. The default is True.
        return_label : bool, optional
            If True, return additional dataframe of process labels. The default is False.
        """
        if len(self._custom_chains) > 0:
            df_chains = pd.DataFrame(
                self._custom_chains,
                columns=["Step_" + str(i) for i in range(len(self._custom_chains[0]))],
            )
            if return_label or print_label:
                df_chains_label = copy.deepcopy(df_chains)
                ref_table = self._process_id_label_ref()
                for i in range(df_chains.shape[0]):
                    for j in range(df_chains.shape[1]):
                        df_chains_label.iloc[i, j] = self._process_id_to_label(df_chains.iloc[i, j], ref_table)
                if print_label:
                    with pd.option_context("display.max_rows", None, "display.max_columns", None):
                        print(df_chains_label)
                if return_label:
                    return df_chains, df_chains_label
                else:
                    return df_chains
            else:
                return df_chains
        else:
            print("No custom chain configured")
            return None

    # Alias
    custom_chains_to_df = ls_custom_chains

    # List default chains
    @simple_type_validator
    def ls_chains(
        self, print_label: bool = True, return_label: bool = False
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame], None]:
        """
        List the process chains in the pipeline execution.
        Returns custom chains if configured, otherwise returns default full factorial chains.

        Output is a dataframe, where each row represents a processing chain with process IDs.

        Parameters
        ----------
        print_label : bool, optional
            If True, prints chains using chain label. The default is True.
        return_label : bool, optional
            If True, return additional dataframe of process labels. The default is False.
        """
        if len(self._custom_chains) > 0:
            result = self.ls_custom_chains(print_label, return_label)
        else:
            result = self.ls_process_chains(print_label, return_label)
        return result

    # Get matched and unmatched process items
    # Format of associated attribute:
    # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    # Data levels:
    # 0 - image (path), \
    # 1 - pixel_spec (1D), 2 - pixel_specs_array (2D), 3 - pixel_specs_tensor (3D), 4 - pixel_hyperspecs_tensor (3D), \
    # 5 - image_ROI (img_path + ROI coords), 6 - ROI_specs (2D), 7 - spec1d (1D spec stats)
    @simple_type_validator
    def _get_process(  # noqa: C901
        self,
        process_id: Optional[str] = None,
        process_label: Optional[str] = None,
        input_data_level: Union[str, int, None] = None,
        output_data_level: Union[str, int, None] = None,
        application_sequence: Union[int, tuple[int, int], None] = None,
        method: Union[str, Callable, object, None] = None,
        full_application_sequence: Union[int, tuple[int, int], None] = None,
        exact_match: bool = True,
    ) -> tuple[
        list[tuple[str, str, str, str, int, Callable, int, int]],
        list[tuple[str, str, str, str, int, Callable, int, int]],
    ]:
        """
        Get matched and unmatched process items by process properties.
        """
        # Validate processes
        if len(self._process) == 0:
            print("No process added!")
            return ([], [])

        # Properties
        if input_data_level is not None:
            input_data_level = _dl_val(input_data_level)[1]
        if output_data_level is not None:
            output_data_level = _dl_val(output_data_level)[1]
        prps: list = [
            process_id,
            process_label,
            input_data_level,
            output_data_level,
            application_sequence,
            method,
            full_application_sequence,
        ]

        # Retrieve removed and remained image items
        get_items = []
        rest_items = []
        for pit in self._process:
            # Validate properties - retrieve and validate
            rcond = True
            for i, prop in enumerate(pit[0:-1]):
                # Condition - not applied
                if prps[i] is None:
                    cond = True
                # Condition - Application_sequence and Full_application_sequence
                # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
                elif ((i == 4) or (i == 6)) and (type(prps[i]) is tuple):
                    if prps[i][0] >= prps[i][1]:
                        raise ValueError(f"Invalid value range [{prps[i][0]}, {prps[i][1]})")
                    else:
                        cond = (prop >= prps[i][0]) & (prop < prps[i][1])
                elif i == 5:
                    if type(prps[i]) is Callable:
                        cond = prop is prps[i]
                    else:
                        assert hasattr(prop, '__name__')
                        cond = ((prps[i] == prop.__name__) & exact_match) | (
                            (prps[i] in prop.__name__) & (not exact_match)
                        )
                else:
                    cond = ((prps[i] == prop) & exact_match) | ((str(prps[i]) in str(prop)) & (not exact_match))
                rcond = rcond & cond

            # Retrieval
            if rcond:
                get_items.append(pit)
            else:
                rest_items.append(pit)

        return get_items, rest_items

    # List / view added processes
    @overload
    def ls_process(
        self,
        process_id: Optional[str] = None,
        process_label: Optional[str] = None,
        input_data_level: Union[str, int, None] = None,
        output_data_level: Union[str, int, None] = None,
        application_sequence: Optional[int] = None,
        method: Union[str, Callable, object, None] = None,
        full_application_sequence: Optional[int] = None,
        *,
        exact_match: bool = True,
        print_result: bool = True,
        return_result: Literal[True] = True,
    ) -> pd.DataFrame: ...

    @overload
    def ls_process(
        self,
        process_id: Optional[str] = None,
        process_label: Optional[str] = None,
        input_data_level: Union[str, int, None] = None,
        output_data_level: Union[str, int, None] = None,
        application_sequence: Optional[int] = None,
        method: Union[str, Callable, object, None] = None,
        full_application_sequence: Optional[int] = None,
        *,
        exact_match: bool = True,
        print_result: bool = True,
        return_result: Literal[False] = False,
    ) -> None: ...

    # List / view added processes
    # Format of associated attribute:
    # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    # Data levels:
    # 0 - image (path), \
    # 1 - pixel_spec (1D), 2 - pixel_specs_array (2D), 3 - pixel_specs_tensor (3D), 4 - pixel_hyperspecs_tensor (3D), \
    # 5 - image_ROI (img_path + ROI coords), 6 - ROI_specs (2D), 7 - spec1d (1D spec stats)
    @simple_type_validator
    def ls_process(
        self,
        process_id: Optional[str] = None,
        process_label: Optional[str] = None,
        input_data_level: Union[str, int, None] = None,
        output_data_level: Union[str, int, None] = None,
        application_sequence: Optional[int] = None,
        method: Union[str, Callable, object, None] = None,
        full_application_sequence: Optional[int] = None,
        *,
        exact_match: bool = True,
        print_result: bool = True,
        return_result: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        List process items filtered by process properties.
        If a property is not specified, the property will be ignored in filtering.
        If return_df is set True, a complete dataframe of processes is returned, if False, a simplified dataframe is printed.

        Parameters
        ----------
        process_id : str, optional
            Process ID. The default is None.

        process_label : str, optional
            Custom process_label. The default is None.

        input_data_level : Union[str,int], optional
            See SpecPipe.add_process.
            The default is None.

        output_data_level : Union[str,int], optional
            See SpecPipe.add_process.
            The default is None.

        application_sequence : int, optional
            Exact sequence number or sequence number range of the method in a data level.
            Range must be specified in tuple.
            The default is None.

        method : Union[str,object,Callable], optional
            Method function or function name. The default is None.

        full_application_sequence : int, optional
            Exact sequence number or sequence number range of the method in the entire pipeline.
            Range must be specified in tuple.
            The default is None.

        exact_match : bool, optional
            If False, any process with property value containing the specified value is removed.
            The default is True.

        print_df : bool, optional
            If simplified results are printed. The default is True.

        return_df : bool, optional
            If a complete resulting dataframe is returned. The default is False.
        """  # noqa: E501
        # Get matched processes
        matched = self._get_process(
            process_id,
            process_label,
            input_data_level,
            output_data_level,
            application_sequence,
            method,
            full_application_sequence,
            exact_match,
        )[0]

        # Full matched in dataframe
        df_proc = pd.DataFrame(
            matched,
            columns=[
                "ID",
                "Process_label",
                "Input_data_level",
                "Output_data_level",
                "Application_sequence",
                "Method",
                "Sequence_in_complete_process",
                "Alternative_number",
            ],
        )
        for i in range(df_proc.shape[0]):
            if callable(df_proc.iloc[i, -3]):
                df_proc.iloc[i, -3] = df_proc.iloc[i, -3].__name__
            else:
                try:
                    df_proc.iloc[i, -3] = df_proc.iloc[i, -3].model_label
                except Exception as e:
                    raise ValueError(
                        f"\nInvalid processing method with type '{type(df_proc.iloc[i, -3])}' detected in process:\
                            \n{df_proc.drop('Method', axis=1).iloc[i, :]}\n\nError message:\n{e}"
                    ) from e
        # Print simple df
        if print_result:
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                df_proc_simple = df_proc.iloc[:, [0, 1, 2, 3, 4, 5]]
                print(df_proc_simple)

        # Return df
        if return_result:
            return df_proc
        elif print_result:
            return None
        else:
            raise ValueError("At least one of return_result or print_result must be True.")

    # Update custom process chains
    def _update_custom_chains(self) -> None:
        if len(self._custom_chains) > 0:
            # Remove non-existed chains
            new_chains = []
            for chain in self._custom_chains:
                if chain in self._process_chains:
                    new_chains.append(chain)
            # Update
            self._custom_chains = new_chains

    # Remove process
    # Format of associated attribute:
    # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    # Data levels:
    # 0 - image (path), \
    # 1 - pixel_spec (1D), 2 - pixel_specs_array (2D), 3 - pixel_specs_tensor (3D), 4 - pixel_hyperspecs_tensor (3D), \
    # 5 - image_ROI (img_path + ROI coords), 6 - ROI_specs (2D), 7 - spec1d (1D spec stats)
    @simple_type_validator
    def rm_process(
        self,
        process_id: Optional[str] = None,
        process_label: Optional[str] = None,
        input_data_level: Union[str, int, None] = None,
        output_data_level: Union[str, int, None] = None,
        application_sequence: Optional[int] = None,
        method: Union[str, Callable, object, None] = None,
        # full_application_sequence : Optional[int] = None,
        exact_match: bool = True,
    ) -> None:
        """
        Remove added processes by ID, Process_label, Data_level, Application_sequence and Method.
        If a property is not specified, the property criterion will be ignored.

        Parameters
        ----------
        process_id : str, optional
            Process ID. The default is None.

        process_label : str, optional
            Custom process_label. The default is None.

        input_data_level : Union[str,int], optional
            See SpecPipe.add_process.
            The default is None.

        output_data_level : Union[str,int], optional
            See SpecPipe.add_process.
            The default is None.

        application_sequence : int, optional
            Exact sequence number or sequence number range of the method in a data level.
            Range must be specified in tuple.
            The default is None.

        method : Union[str,Callable,object], optional
            Method function or function name. The default is None.

        exact_match : bool, optional
            If False, any process with property value containing the specified value is removed.
            The default is True.
        """
        # Not applied parameters
        full_application_sequence = None
        # Change process test status
        self._tested = False

        # Filter processes
        matched, unmatched = self._get_process(
            process_id,
            process_label,
            input_data_level,
            output_data_level,
            application_sequence,
            method,
            full_application_sequence,
            exact_match,
        )

        if len(unmatched) == 0:
            print("\nAll processes are removed")
            self._process = []
            self._process_steps = []
            self._process_chains = []
            self._custom_chains = []

        else:
            if len(matched) > 0:
                # Update process
                self._process = unmatched
                # Sort processes and update process_steps and process_chains
                self._generate_process_steps()
                self._generate_process_chains()
                self._update_custom_chains()

                # Print report
                df_proc = pd.DataFrame(
                    matched,
                    columns=[
                        "ID",
                        "Process_label",
                        "Input_data_level",
                        "Output_data_level",
                        "Application_sequence",
                        "Method",
                        "Sequence_in_complete_process",
                        "Alternative_number",
                    ],
                )
                for i in range(df_proc.shape[0]):
                    if callable(df_proc.iloc[i, -3]):
                        df_proc.iloc[i, -3] = df_proc.iloc[i, -3].__name__
                    else:
                        try:
                            df_proc.iloc[i, -3] = df_proc.iloc[i, -3].model_label
                        except Exception as e:
                            raise ValueError(
                                f"\nInvalid processing method with type '{type(df_proc.iloc[i, -3])}' \
                                    detected in process: \n{df_proc.drop('Method', axis=1).iloc[i, :]}\
                                    \n\nError message:\n{e}"
                            ) from e
                with pd.option_context("display.max_rows", None, "display.max_columns", None):
                    df_proc_simple = df_proc.iloc[:, [1, 2, 3, 4, 5]]
                    print(f"\nFollowing processes are removed from the pipeline: \n{df_proc_simple}")
            else:
                print("\nNo matched process found")

    # Construct initial sample data from SpecExp
    # Sample data format - ROI: {ID, label, target, img_path, roi_coords}
    # Sample data format - standalone spec: {ID, label, target, spec1d: tuple}
    def _sample_data_constructor(self) -> None:
        """
        Constructing the initial sample data from SpecExp.
        If the sample is defined as ROIs of spectral images, the data constructing information is initialized, but the data itself is not loaded (for lazy-loading).
        """  # noqa: E501
        # ROI level (level ind 5)
        # ROI item format: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
        # Img item format: [0 id, 1 group, 2 image_name, 3 mask_of, 4 image_path]
        # Target value item format: [0 fixed sample id, 1 user assinged labels, 2 Target values]
        if (len(self.spec_exp.rois_sample) > 0) and (len(self.spec_exp.standalone_specs_sample) == 0):
            sample_data: list[dict] = []
            for roit in self.spec_exp.rois_sample:
                sdata: dict[str, Any] = {}
                sdata["ID"] = roit[0]
                sdata["label"] = [lbt[1] for lbt in self.spec_exp.sample_labels if lbt[0] == roit[0]][0]
                sdata["target"] = [tg[2] for tg in self.spec_exp.sample_targets if tg[0] == roit[0]][0]
                sdata["img_path"] = [
                    imgt[4] for imgt in self.spec_exp.images if ((imgt[1] == roit[1]) & (imgt[2] == roit[2]))
                ][0]
                sdata["roi_coords"] = roit[5]
                sample_data.append(sdata)
            self._sample_data = sample_data

        # Standalone spectra (level ind 7)
        # Standalone spectrum item format: [0 id, 1 group, 2 use_type, 3 sample_name, 4 spectral_data_list]
        elif (len(self.spec_exp.standalone_specs_sample) > 0) and (len(self.spec_exp.rois_sample) == 0):
            sample_data = []
            for st in self.spec_exp.standalone_specs_sample:
                sdata = {}
                sdata["ID"] = st[0]
                sdata["label"] = [lbt[1] for lbt in self.spec_exp.sample_labels if lbt[0] == st[0]][0]
                sdata["target"] = [tg[2] for tg in self.spec_exp.sample_targets if tg[0] == st[0]][0]
                sdata["spec1d"] = tuple(st[4])
                sample_data.append(sdata)
            self._sample_data = sample_data

        else:
            raise ValueError(
                "Hybrid samples from both standalone spectra and spectral images \
                    is not allowed by SpecPipe pipeline.\
                    \nPlease provide pure image ROI samples or standalone spectrum samples"
            )

    # Test run of all chains
    # Format of associated attribute:
    # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    # Data levels:
    # 0 - image (path), \
    # 1 - pixel_spec (1D), 2 - pixel_specs_array (2D), 3 - pixel_specs_tensor (3D), 4 - pixel_hyperspecs_tensor (3D), \
    # 5 - image_ROI (img_path + ROI coords), 6 - ROI_specs (2D), 7 - spec1d (1D spec stats)
    # Pretest_data: [img_path, test_img_path, roi_coords, test_roi_coords, roitable, spec1d]
    @simple_type_validator
    def test_run(
        self,
        test_modeling: bool = True,
        return_result: bool = False,
        dump_result: bool = True,
        dump_backup: bool = False,
        save_preprocessed_images: bool = False,
    ) -> Optional[dict]:
        """
        Run the pipeline of all processing chains using simplified test data. This method is executed automatically prior to each formal run.

        Parameters
        ----------
        test_modeling : bool, optional
            Whether added models are tested.
            If False, only first chain is tested. The default is True.

        return_result : bool, optional
            Whether returned results of the processes are returned.
            If True, results of all tested steps are returned in a list. The default is False.

        dump_result : bool. optional
            Whether test results are dumped in chains. The default is True.

        dump_backup : bool, optional
            Whether backup for the step result is dumped. The backup file is named with the datetime of dumping.

        Returns
        -------
        status_results : Optional[list[dict[list,list]]]
            List of all tested steps in dictionary:
                - Dictionary keys : list of applied processes of a processing chain.
                - Dictionary values : list of processing results of the applied processes at all the steps of the processing chain.
        """  # noqa: E501

        # Test data
        if self._pretest_data is not None:
            test_data: dict[str, Any] = copy.deepcopy(self._pretest_data)
        else:
            raise ValueError(
                "Internal Error: 'SpecPipe.pretest_data' is None. \
                    Pre-execution test data initialization fails. Please report."
            )

        # Preprocessed test image dir
        preprocessed_img_dir = self._spec_exp.report_directory + "test_run/Preprocessed_images/"
        if not os.path.exists(unc_path(preprocessed_img_dir)):
            os.makedirs(unc_path(preprocessed_img_dir))

        # Preprocessing test
        status_results: dict = _preprocessing_sample(  # type: ignore[call-overload]
            sample_data=test_data,
            process=self.process,
            custom_chains=self.custom_chains,
            process_chains=self.process_chains,
            specpipe_report_directory=self.spec_exp.report_directory,
            preprocess_status=self._init_preprocessing_status(),
            dump_result=dump_result,
            return_result_path=False,
            dump_backup=dump_backup,
            return_step_result=True,
            final_result_only=False,
            is_test_run=True,
        )
        # Multiple dynamic bool switches passed to '_preprocessing_sample' that fails for type 'Literal' in overloads

        # If not save preprocessed images, remove after test
        if not save_preprocessed_images:
            shutil.rmtree(preprocessed_img_dir)

        # Modeling test
        if test_modeling:
            if len(self.ls_model(print_result=False, return_result=True)) > 0:
                self._test_model(status_results, self.spec_exp.report_directory)
            # Change process test status
            self._tested = True

        if return_result:
            return status_results
        else:
            return None

    # Test modeling
    def _test_model(self, status_results: dict, specpipe_report_directory: str) -> None:  # noqa: C901
        """
        Test models added on a simulated testing targets based on the given preprocessing results.
        """
        # Get model processes
        model_procs = [procit for procit in self._process if procit[3] == "model"]

        # Get model processes with input data level index dl_in_ind
        # Process: [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
        for modelit in model_procs:
            # Get test model
            model_methodi = modelit[5]
            assert hasattr(model_methodi, 'validation_method')

            # Get minimum mock sample size according to validation fold
            val_method = model_methodi.validation_method
            val_method = str(val_method).lower()
            if "fold" in val_method:
                sample_min_size = max(10, int(val_method.split("-")[0]))
            elif "split" in val_method:
                a = float(val_method.split("-")[0])
                b = float(val_method.split("-")[1])
                sample_min_size = max(10, int(max(a, b) / min(a, b)) + 1)
            elif "loo" in val_method:
                sample_min_size = 10

            # Get model input data level
            dl_in_name = _dl_val(modelit[2])[1]
            dl_in_ind = _dl_val(modelit[2])[0]

            # Get model input data shape
            assert hasattr(model_methodi, "input_shape")
            ts_shape = model_methodi.input_shape

            ## Construct test samples list and implement model testing
            # Status result: (0 - step_id, 1 step_procs, 2 dl_in, 3 dl_out, 4 sample_result)
            for pci, status_result in enumerate(status_results["status_results"][-1]):
                # Get sample data of the model input data level
                if status_result[3] == dl_in_ind:
                    # Get test sample data
                    tsample = np.array(status_result[4])

                    # Validate data shape
                    if ts_shape is None:
                        ts_shape = tsample.shape
                    elif np.prod(tsample.shape) != np.prod(ts_shape):
                        raise ValueError(
                            f"Cannot reshape sample data with shape {tsample.shape} into specified input data shape \
                                {ts_shape} of the model.\
                                \nInput step data ID: {status_result[0]}\nModel label: {modelit[1]}"
                        )

                    # Save preprocess chain of sample_list
                    with open(unc_path(specpipe_report_directory + f"test_run/Preprocess_chain_#{pci}.txt"), "w") as f:
                        for ppci, pproc in enumerate(status_result[1]):
                            if ppci < (len(status_result[1]) - 1):
                                f.write(f"{pproc}\n")
                            else:
                                f.write(f"{pproc}")

                    # For data level of numeric values
                    # Sample_list: (0 - Sample id, 1 - Original shape, 2 - Target value, 3 - Sample predictor value)
                    # if (dl_in_ind != 0) & (dl_in_ind != 5):
                    if dl_in_ind == 7:
                        test_data_range: float = float(np.nanmax(tsample) - np.nanmin(tsample))
                        # Model report dir
                        model_report_dir = specpipe_report_directory + "test_run/"
                        if not os.path.exists(unc_path(model_report_dir)):
                            os.makedirs(unc_path(model_report_dir))
                        # Build testing sample list
                        assert hasattr(model_methodi, 'is_regression')
                        if model_methodi.is_regression:
                            # Regression mock data
                            test_samples: list[tuple[str, Any, Union[float, int, bool, str], np.ndarray]] = [
                                (
                                    str(i),
                                    ts_shape,
                                    float(i),
                                    tsample
                                    * (
                                        1
                                        + test_data_range
                                        * (0.5 / sample_min_size)
                                        * (i + 0.1 * i * np.random.rand(1)[0])
                                    ),
                                )
                                for i in range(sample_min_size)
                            ]
                        else:
                            # Classification mock data
                            test_samples = [
                                (
                                    str(i),
                                    ts_shape,
                                    str(["a", "b"][int(i % 2)]),
                                    tsample * (1 + test_data_range * 0.25 * (i % 2 + 0.1 * np.random.rand(1)[0])),
                                )
                                for i in range(sample_min_size)
                            ]
                        # Test modeling
                        assert hasattr(model_methodi, 'evaluation')
                        model_methodi.evaluation(
                            sample_list=test_samples,
                            data_label="test_chain_" + str(pci),
                            report_directory=model_report_dir,
                        )
                    else:
                        raise ValueError(f"Model only accepts data level 'spec1d' as input, but got: {dl_in_name}")

    # Run entire pipeline
    # Sample data format - ROI: {ID, label, target, img_path, roi_coords}
    # Sample data format - standalone spec: {ID, label, target, spec1d: tuple}
    # Process: [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    # Status_result: (0 - step_id, 1 step_procs, 2 dl_in, 3 dl_out, 4 sample_result)
    # Sample_list: (0 - Sample id, 1 - Original shape, 2 - Target value, 3 - Sample predictor value)
    @simple_type_validator
    def preprocessing(  # noqa: C901
        self,
        result_directory: str = "",
        n_processor: int = 1,
        dump_backup: bool = False,
        final_result_only: bool = True,
        keep_chain_results: bool = True,
        resume: bool = False,
        to_csv: bool = True,
        show_progress: bool = True,
        save_config: bool = True,
        summary: bool = True,
        geo_reference_warning: bool = False,
        skip_test: bool = False,
    ) -> None:
        """
        Implement preprocessing steps of all processing chains on the entire dataset and output modeling-ready sample_list data to files.

        Parameters
        ----------
        result_directory : str, optional
            Directory to save the final preprocess results. The default is using the report_directory of given SpecExp.

        n_processor : int
            The number of processor to use in the preprocessing. The default is 1 (Parallel processing is not applied).

            Windows Note: When using n_processor > 1 on Windows, all main codes in the working script must be placed within block "if __name__ == '__main__':".
            This requirement comes from 'pathos', which uses dill for object serialization and is essential for parallel execution of the package functions.

        dump_backup : bool, optional
            Create backup files of result with time stamp. The default is False.

        keep_chain_results : bool, optional
            Whether step results of the chains are kept, if False, the step results will be removed after preprocessing. The default is True.

        final_result_only : bool, optional
            If True, only results of final steps are saved for the chain results. The default is True.

        resume : bool
            If True, the computation will resume from preprocessing progress logs.
            Apply 'resume' after breaks to avoid repeated preprocessing of the processed samples. The default is False.

        to_csv : bool
            If True, the final results of preprocessing is also saved to CSV files in addition to dill files. The default is True.

        show_progress : bool, optional
            Show processing progress. The default is True.

        save_config : bool, optional
            Save SpecPipe configurations. The default is True.

        summary : bool, optional
            Whether to summarize preprocessed data and target value. The default is True.

        geo_reference_warning: bool, optional
            Whether to suppress GeoReferenceWarning, if False, the warning is suppressed. The default is False.

        skip_test: bool, optional
            Whether skip test execution completely. Test execution valiates every processing chain and serves as a safeguard against runtime errors in a long formal execution.
        """  # noqa: E501
        # Prompt "if __name__ == '__main__':" protection for windows multiprocessing
        if n_processor > 1:
            if os.name == "nt":
                print(
                    "\n\n\n\
                    >>> WARNING: Windows users must run multiprocessing within block \n\nif __name__ == '__main__':\
                    \n\n Please make sure all of your main codes in the script are placed within this block. \n\n",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(2)
                warn_msg = (
                    "\nWindows users must run multiprocessing within block \n\nif __name__ == '__main__':"
                    + "Please make sure all of your main codes in the script are placed within this block."
                )
                warnings.warn(warn_msg, UserWarning, stacklevel=2)

        # Suppress NotGeoreferencedWarning
        if not geo_reference_warning:
            warnings.simplefilter("ignore", NotGeoreferencedWarning)

        # Save configurations
        if save_config:
            self.save_pipe_config(copy=True)

        # Added chain testing
        if not self.tested and not skip_test:
            self.test_run(test_modeling=False, return_result=False, dump_result=False, dump_backup=False)

        # Default result_directory
        if result_directory == "":
            result_directory = self._spec_exp.report_directory
        # Preprocessed image dir for data level 0~4
        preprocessed_img_dir = result_directory + "Preprocessing/Preprocessed_images/"
        if not os.path.exists(unc_path(preprocessed_img_dir)):
            os.makedirs(unc_path(preprocessed_img_dir))

        # Preprocessing
        self._preprocessor(
            result_directory=result_directory,
            n_processor=n_processor,
            dump_backup=dump_backup,
            final_result_only=final_result_only,
            resume=resume,
            show_progress=show_progress,
        )

        # Construct sample_list
        self._sample_list_constructor(result_directory=result_directory, to_csv=to_csv, show_progress=show_progress)

        # Clear step result data after sample list construction after finishing sample_list construction
        if not keep_chain_results:
            self._cl_step_result(result_directory=result_directory)

        # Print result dir
        if len(result_directory) > 0:
            result_dir = str(result_directory)
        else:
            result_dir = self.report_directory
        self._save_sample_targets(result_dir)
        print(f"\n\nPipeline preprocessing complete, result stored in:\n{result_dir}")

        # Group stats for preprocessing
        if summary:
            _ = sample_group_stats(self.report_directory, is_regression=self.is_target_numeric)

        # Recover NotGeoreferencedWarning
        warnings.simplefilter("default", NotGeoreferencedWarning)

    @simple_type_validator
    def _preprocessor(  # noqa: C901
        self,
        result_directory: str = "",
        n_processor: int = 1,
        dump_backup: bool = False,
        final_result_only: bool = True,
        resume: bool = False,
        show_progress: bool = True,
        *,
        # Dependencies for multiprocessing
        copy: ModuleType = copy,
        os: ModuleType = os,
        time: ModuleType = time,
        datetime: type = datetime,
        np: ModuleType = np,
        torch: ModuleType = torch,
    ) -> None:
        """
        Preprocessing of all processing chains on the entire dataset.
        Output sample results from all processing chains to files and update the resulting file paths.
        """
        # Store SpecExp configs
        self.spec_exp.save_data_config(copy=dump_backup)

        # Validate report directory
        if result_directory == "":
            result_directory = self._spec_exp.report_directory

        # Construct sample data
        self._sample_data_constructor()

        ## Preprocessing
        # Validate result_directory / report_directory
        # Step dir for chain results
        step_dir = result_directory + "Preprocessing/Step_results/"
        if not os.path.exists(unc_path(step_dir)):
            os.makedirs(unc_path(step_dir))
        # log dir for resume
        log_dir_path = step_dir + "Preprocess_progress_logs/"
        if not os.path.exists(unc_path(log_dir_path)):
            os.makedirs(unc_path(log_dir_path))

        # Check running log and subset sample data
        if not resume:
            rest_sample_data = self.sample_data
        else:
            if not os.path.exists(unc_path(log_dir_path)):
                os.makedirs(unc_path(log_dir_path))
                finished_samples = []
            else:
                finished_samples = [
                    f.split(".")[0]
                    for f in os.listdir(unc_path(log_dir_path))
                    if os.path.isfile(unc_path(log_dir_path + f))
                ]
                existed_samples = [sd["ID"] for sd in self.sample_data]
                finished_samples = [sdid for sdid in finished_samples if sdid in existed_samples]
            if len(finished_samples) > 0:
                rest_sample_data = [sd for sd in self.sample_data if sd["ID"] not in finished_samples]
            else:
                rest_sample_data = self.sample_data

        # Initialize preprocessing status for image operation
        preprocess_status = self._init_preprocessing_status()

        # Preprocessing of all data and generate sample_list data of all chains
        if show_progress:
            print("\nPreprocess samples ...")
        self._preprocess_result_path = []

        # Sequential compute
        if n_processor <= 1:
            for sd_ind in tqdm(
                range(len(rest_sample_data)),
                total=len(rest_sample_data),
                disable=(not show_progress),
            ):
                sdata = rest_sample_data[sd_ind]
                pti = _preprocessing_sample(
                    sdata,
                    process=self.process,
                    custom_chains=self.custom_chains,
                    process_chains=self.process_chains,
                    specpipe_report_directory=result_directory,
                    dump_result=True,
                    return_result_path=True,
                    dump_backup=dump_backup,
                    return_step_result=False,
                    final_result_only=final_result_only,
                    is_test_run=False,
                    dump_directory=step_dir,
                    update_progress_log=True,
                    preprocess_status=preprocess_status,
                )
                self._preprocess_result_path.append(pti)

        # Parallel compute
        else:
            # Initialize errorlogs dir for err handling in parallel computing
            errorlog_path = result_directory + "Preprocessing/Step_results/Error_logs/"
            if os.path.exists(unc_path(errorlog_path)):
                shutil.rmtree(unc_path(errorlog_path))
            # Validate number of processors to use
            ncpu_max = max(cpu_count() - 1, 1)
            ncpu = min(n_processor, ncpu_max)
            # Bind constant arguments for _preprocessing_sample - Result dumped in _preprocessing_sample
            preprocessing_sample_it = partial(
                _preprocessing_sample,
                process=self.process,
                custom_chains=self.custom_chains,
                process_chains=self.process_chains,
                specpipe_report_directory=result_directory,
                dump_result=True,
                return_result_path=True,
                dump_backup=dump_backup,
                return_step_result=False,
                final_result_only=final_result_only,
                is_test_run=False,
                dump_directory=step_dir,
                update_progress_log=True,
                preprocess_status=preprocess_status,
                # Explicitly assign dependencies for multiprocessing
                copy=copy,
                os=os,
                datetime=datetime,
                np=np,
                torch=torch,
            )
            # Processing - multiprocessing for loop
            with ProcessingPool(nodes=ncpu) as pool:
                preprocess_result_paths = list(
                    tqdm(
                        pool.imap(preprocessing_sample_it, rest_sample_data),
                        total=len(rest_sample_data),
                        disable=(not show_progress),
                    )
                )
            # Collect and print errors if exist
            if os.path.exists(unc_path(errorlog_path)):
                raise ValueError(
                    f"\nPreprocessing errors, please check error logs in the following path:\n{errorlog_path}"
                )
            # Validate returning path list
            if len(preprocess_result_paths) != len(rest_sample_data):
                raise ValueError(
                    f"\nIncomplete preprocessing results, expected number of results: {len(self.sample_data)}, \
                        got: {len(preprocess_result_paths)}"
                )
            for pti in preprocess_result_paths:
                if type(pti) is not str:
                    raise TypeError(f"\nResult file path must be str, got path: {pti}, with type: {type(pti)}")
                elif not os.path.exists(unc_path(pti)):
                    raise ValueError(f"\nGot invalid path: {pti}")
            # Update preprocess result file paths
            self._preprocess_result_path = preprocess_result_paths

        # Clear log after finishing whole preprocessing
        if os.path.exists(unc_path(log_dir_path)):
            shutil.rmtree(unc_path(log_dir_path))

    # Step results to modeling-ready sample_list data
    @simple_type_validator
    def _sample_list_constructor(  # noqa: C901
        self, result_directory: str = "", to_csv: bool = True, show_progress: bool = False
    ) -> None:
        """
        Convert Step_result data from file to modeling-ready sample_list data.
        """
        # Validate report directory
        if result_directory == "":
            result_directory = self._spec_exp.report_directory
        preprocess_result_dir = result_directory + "Preprocessing/"
        if not os.path.exists(unc_path(preprocess_result_dir)):
            os.makedirs(unc_path(preprocess_result_dir))

        # Validate preprocessing result file paths
        sd_paths = [
            f"{preprocess_result_dir}Step_results/PreprocessingResult_sample_{sd['label']}.dill"
            for sd in self.sample_data
        ]
        for sdp in sd_paths:
            if not os.path.exists(unc_path(sdp)):
                raise ValueError(f"\nPreprocessing step result file path not found : \n{sdp}\n")

        ## Chain results to sample_list data
        # Get preprocess chains of all preprocessing steps
        pchains = []
        for pchain in self.process_chains:
            if pchain[:-1] not in pchains:
                pchains.append(pchain[:-1])

        ## Loop preprocess chains across all samples and transform to modeling data
        if show_progress:
            print("\nConstruct chain sample list ...\nChain :")
        for pci in tqdm(range(len(pchains)), total=len(pchains), disable=(not show_progress)):
            pchain = pchains[pci]
            pre_results = []
            for spath in sd_paths:
                sdata = load_vars(unc_path(spath))
                status_results = sdata["status_results"][-1]
                # Sample ID and sample target value
                sample_id = sdata["ID"]
                sample_y = sdata["target"]
                # Sample data
                for status_result in status_results:
                    if tuple(status_result[1]) == tuple(pchain):
                        # Construct sample_list item
                        step_data = status_result[4]
                        step_data_shape = np.array(step_data).shape
                        # Validate step output data level
                        step_dl_out = status_result[3]
                        if step_dl_out != 7:
                            raise ValueError("Input data level of modeling step")
                        pre_results.append((sample_id, step_data_shape, sample_y, step_data))

            ## Save resutls to files
            # Create file name
            chain_name = ""
            for proc_name in pchain:
                chain_name = chain_name + proc_name + "-"
            chain_name1 = f"PreprocessingChainResult_chain_ind_{str(pci)}"

            # Dump results to dill
            # Chain_res / sample_list
            # Sample_list item: (0 - Sample id, 1 - Original shape, 2 - Target value, 3 - Sample predictor value)
            # Typing: list[tuple[str, tuple[int], Union[str,int,bool,float], Annotated[Any,arraylike_validator(ndim=1)]]]  # noqa: E501
            res_path_dill = preprocess_result_dir + chain_name1 + ".dill"
            dump_vars(unc_path(res_path_dill), {"chain_ind": pci, "chain_procs": pchain, "chain_res": pre_results})

            # Save results to CSV
            if to_csv:
                # Results to table (df)
                chain_res_table = [(pres[0], str(pres[1]), pres[2]) + tuple(pres[3]) for pres in pre_results]
                arr_chain_res = np.array(chain_res_table)
                coln_chain_res = ["Sample_ID", "X_shape", "y"] + [f"x{i}" for i in range(arr_chain_res.shape[1] - 3)]
                df_chain_res = pd.DataFrame(arr_chain_res, columns=coln_chain_res)
                # Add chain name to table content (as first col)
                df_chain_res = pd.concat(
                    [
                        pd.DataFrame(
                            [[chain_name]] + [[""]] * (df_chain_res.shape[0] - 1),
                            columns=["Preprocessing_chain"],
                        ),
                        df_chain_res,
                    ],
                    ignore_index=True,
                    axis=1,
                )
                # Recover colnames
                df_chain_res.columns = ["Preprocessing_chain"] + coln_chain_res
                # Save table to CSV
                res_path_csv = preprocess_result_dir + chain_name1 + ".csv"
                df_to_csv(df_chain_res, res_path_csv, index=False, return_path=False)
        # Add line after progress bar
        print("")

    ## Clear generated step results
    @simple_type_validator
    def _cl_step_result(self, result_directory: str) -> None:
        # Step dir for chain results
        step_dir = result_directory + "Preprocessing/Step_results/"
        if os.path.exists(unc_path(step_dir)):
            for item in os.listdir(step_dir):
                if "PreprocessingChainResult_" in str(item):
                    item_path = os.path.join(step_dir, item)
                    try:
                        if os.path.isfile(unc_path(item_path)):
                            os.remove(item_path)
                    except Exception as e:
                        print(f"Error in removing '{item_path}': \n{e}")
        else:
            warnings.warn(
                f"Step_results path is invalid:\n{result_directory}\nNo 'PreprocessingChainResult' is cleared.",
                UserWarning,
                stacklevel=3,
            )

    # Preprocessing image processing status
    @staticmethod
    def _init_preprocessing_status() -> dict:
        """Initialize shared image processing status for preprocessing."""
        manager = mp.Manager()
        preprocess_status: dict = {
            'start_status': manager.list(),
            'completion_status': manager.list(),
            'processed_image_init': manager.list(),
            'lock': manager.Lock(),
        }
        return preprocess_status

    # Run modeling on single dataset
    # Sample_list item: (0 - Sample id, 1 - Original shape, 2 - Target value, 3 - Sample predictor value)
    @simple_type_validator
    def model_evaluation(  # noqa: C901
        self,
        n_processor: int = 1,
        resume: bool = False,
        report_directory: str = "",
        show_progress: bool = True,
        save_config: bool = True,
        summary: bool = True,
    ) -> None:
        """
        Evaluating added models on sample data from all preprocessing chains.

        Specify modeling and evaluation parameters when adding models.

        Parameters
        ----------
        n_processor : int, optional
            The number of processor to use in the preprocessing. The default is 1 (Parallel processing is not applied).

            Windows Note: When using n_processor > 1 on Windows, all main codes in the working script must be placed within block "if __name__ == '__main__':".
            This requirement comes from 'pathos', which uses dill for object serialization and is essential for parallel execution of the package functions.

        resume : bool, optional
            If True, the computation will resume from preprocessing progress logs.
            Apply 'resume' after breaks to avoid repeated preprocessing of the processed samples. The default is False.

        report_directory : str, optional
            Directory for saving modeling reports. The default is using the report_directory of given SpecExp.

        show_progress : bool, optional
            Show processing progress. The default is True.

        save_config : bool, optional
            Save SpecPipe configurations. The default is True.

        summary : bool, optional
            Whether to summarize performance metrics and marginal performance metrics.
            The marginal performances of different processes at each step are compared using Mann-Whitney U test.
            The default is True.
        """  # noqa: E501

        # Prompt "if __name__ == '__main__':" protection for windows multiprocessing
        if n_processor > 1:
            if os.name == "nt":
                print(  # noqa: E501
                    "\n\n\n\
                    >>> WARNING: Windows users must run multiprocessing within block \n\nif __name__ == '__main__':\
                    \n\n Please make sure all of your main codes in the script are placed within this block. \n\n",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(2)
                warnings.warn(
                    "Windows users must run multiprocessing within block \n\nif __name__ == '__main__': \n\n\
                    Please make sure all of your main codes in the script are placed within this block.",
                    UserWarning,
                    stacklevel=2,
                )

        # Save configurations
        if save_config:
            self.save_pipe_config(copy=True)

        # Close existing pyplot to save memory
        plt.close("all")

        # Validate report directory
        result_directory = report_directory
        if result_directory == "":
            result_directory = self._spec_exp.report_directory
        # Preprocessing dir
        preprocess_result_dir = result_directory + "Preprocessing/"
        if not os.path.exists(unc_path(preprocess_result_dir)):
            raise ValueError(f"\nPreprocessing result directory not found, got path:\n{preprocess_result_dir}")
        # Modeling dir
        model_result_dir = result_directory + "Modeling/"
        if not os.path.exists(unc_path(model_result_dir)):
            os.makedirs(unc_path(model_result_dir))

        # Validate models
        model_procs = [proc for proc in self.process if proc[3] == "model"]
        if len(model_procs) < 1:
            raise ValueError("No model added.")

        # Validate preprocessing result file paths
        # Get preprocess chains of all preprocessing steps
        pchains = []
        for chain in self.process_chains:
            pchain = chain[:-1]
            if pchain not in pchains:
                pchains.append(pchain)
        # Validate result file existence
        cd_paths = [
            f"{preprocess_result_dir}PreprocessingChainResult_chain_ind_{pci}.dill" for pci in range(len(pchains))
        ]
        pchains_f = []
        for pci, cdp in enumerate(cd_paths):
            if not os.path.exists(unc_path(cdp)):
                raise ValueError(f"\nPreprocessing result file of chain {pchains[pci]} not found, path : \n{cdp}\n")
            cprocs = load_vars(unc_path(cdp))["chain_procs"]
            pchains_f.append(cprocs)
        spcs = set(pchains)
        spcf = set(pchains_f)
        if spcs != spcf:
            raise ValueError(
                f"\nPreprocessing chains and preprocessing results do not match, \
                    unmatched chains:\n\n{list((spcs | spcf) - (spcs & spcf))}\n"
            )

        # Check running log and subset sample data
        log_path = model_result_dir + "Model_evaluation_reports/" + "modeling_progress_log.dill"
        if not resume:
            rest_cd_paths = cd_paths
            # Clear log
            self._clear_model_log(log_path)
        else:
            if not os.path.exists(unc_path(log_path)):
                rest_cd_paths = cd_paths
            else:
                modeling_progress_log = load_vars(unc_path(log_path))["modeling_progress_log"]
                rest_cd_paths = []
                for cdp in cd_paths:
                    cprocs = load_vars(unc_path(cdp))["chain_procs"]
                    if cprocs not in modeling_progress_log:
                        rest_cd_paths.append(cdp)
            if len(rest_cd_paths) == 0:
                rest_cd_paths = cd_paths

        # Modeling by chain sample_list
        # Sequential processing modeling
        if n_processor <= 1:
            for pci, cdp in enumerate(rest_cd_paths):
                try:
                    # Progress
                    if show_progress:
                        print(f"\nModeling preprocessing result {pci + 1}/{len(rest_cd_paths)} :")
                    pc_it = load_vars(unc_path(cdp))
                    pc_sample_list = pc_it["chain_res"]
                    pc_sample_list = _target_type_validation_for_serialization(pc_sample_list)
                    pchain = pc_it["chain_procs"]
                    # Use preprocess chain ID as chain label
                    pproc_chain_label = [f"Preprocessing_#{pci}" for pci, pc in enumerate(pchains) if pc == pchain][0]
                    _model_evaluator(
                        preprocess_result=pc_sample_list,
                        preprocess_chain=pchain,
                        preprocess_chain_label=pproc_chain_label,
                        model_processes=model_procs,
                        specpipe_report_directory=self.report_directory,
                        result_directory=result_directory,
                        update_progress_log=True,
                    )

                # Error handling
                except Exception as e:
                    # Validate report directory
                    model_result_dir = result_directory + "Modeling/"
                    if not os.path.exists(unc_path(model_result_dir)):
                        os.makedirs(unc_path(model_result_dir))
                    errdir = model_result_dir + "Error_logs/"
                    if not os.path.exists(unc_path(errdir)):
                        os.makedirs(unc_path(errdir))
                    cts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    pid = os.getpid()
                    error_log_path = errdir + f"error_{cts}_pid_{pid}.log"
                    err_msg = f"\nFailed in the modeling of preprocessing chain from path '{cdp}', \
                                error message: \n\n{str(e)}\n"
                    with open(unc_path(error_log_path), "w") as f:
                        f.write(err_msg)
                    raise ValueError(e) from e

        # Multiprocessing modeling
        else:
            # Initialize errorlogs dir for err handling in parallel computing
            errorlog_path = model_result_dir + "Error_logs/"
            if os.path.exists(unc_path(errorlog_path)):
                shutil.rmtree(unc_path(errorlog_path))
            # Validate number of processors to use
            ncpu_max = max(cpu_count() - 1, 1)
            ncpu = min(n_processor, ncpu_max)
            # Bind constant arguments for _model_evaluator_it
            _model_evaluator_it = partial(
                _model_evaluator_mp,
                pchains=pchains,
                model_processes=model_procs,
                specpipe_report_directory=self.report_directory,
                result_directory=result_directory,
                update_progress_log=True,
                # Assign applied functions
                _model_evaluator=_model_evaluator,
                _dl_val=_dl_val,
                unc_path=unc_path,
                load_vars=load_vars,
                dump_vars=dump_vars,
                _target_type_validation_for_serialization=_target_type_validation_for_serialization,
                modeleva=ModelEva,
                silent_all=True,
            )
            # Processing - multiprocessing for loop
            with ProcessingPool(nodes=ncpu) as pool:
                for _ in tqdm(
                    pool.imap(_model_evaluator_it, rest_cd_paths),
                    total=len(rest_cd_paths),
                    disable=(not show_progress),
                ):
                    pass
            # Collect and print errors if exist
            if os.path.exists(unc_path(errorlog_path)):
                raise ValueError(
                    f"\nPreprocessing errors, please check error logs in the following path:\n{errorlog_path}"
                )

        # Clear log
        self._clear_model_log(log_path)

        # Print result dir and save corresponding targets
        if len(result_directory) > 0:
            result_dir = str(result_directory)
        else:
            result_dir = self.report_directory
        self._save_sample_targets(result_dir)
        print(f"\n\nPipeline model evaluation complete, result stored in:\n{result_dir}")

        # Performance summary and marginal performance stats
        if summary:
            _ = performance_marginal_stats(self.report_directory)
            _ = combined_model_marginal_stats(self.report_directory)

    @staticmethod
    def _clear_model_log(log_path: str) -> None:
        # Clear log when finished
        if os.path.exists(unc_path(log_path)):
            try:
                os.remove(unc_path(log_path))
            except PermissionError as e:
                raise PermissionError(f"\nNo permission to clear existed running log : \n'{log_path}'.\n") from e
            except Exception as e:
                raise ValueError(f"\nError in clearing existed running log : \n{e}\n") from e

    def _save_sample_targets(self, result_dir: str) -> None:
        # Save dir
        save_dir = f"{result_dir}/Modeling/".replace("//", "/")
        # Validate save dir
        if os.path.exists(unc_path(save_dir)):
            self.spec_exp.sample_targets_to_csv(unc_path(save_dir + "sample_targets.csv"))
        else:
            os.makedirs(unc_path(save_dir))
            self.spec_exp.sample_targets_to_csv(unc_path(save_dir + "sample_targets.csv"))
            # raise ValueError(f"Invalid modeling result directory: {save_dir}")

    @simple_type_validator
    def run(  # noqa: C901
        self,
        result_directory: str = "",
        n_processor: int = -1,
        test_model: bool = True,
        pipe_parallel_for_modeling: bool = False,
        dump_backup: bool = False,
        step_result: bool = True,
        keep_chain_results: bool = True,
        resume: bool = False,
        sample_data_to_csv: bool = True,
        show_progress: bool = True,
        save_config: bool = True,
        summary: bool = True,
        geo_reference_warning: bool = False,
        skip_test: bool = False,
    ) -> None:
        """
        Run pipeline of given processes on SpecExp instance (corresponding manager of spectral experiment data).
        Full-factorial test is applied to multiple processes of identical sequence.
        Processes are configured using method 'add_process'.

        Parameters
        ----------
        result_directory : str, optional
            Directory to save the preprocess results and modeling reports.
            The default is using the report_directory of given SpecExp.

        n_processor : int
            The number of processor to use in the preprocessing.
            The default is -1, which automatically uses sequential processing for Windows and (maximum available CPUs - 1) processors for other OS.
            Set to -2 to use (maximum available CPUs - 1) processors on Windows.

            Windows Note: When using n_processor > 1 on Windows, all main codes in the working script must be placed within block "if __name__ == '__main__':".
            This requirement comes from 'pathos', which uses dill for object serialization and is essential for parallel execution of the package functions.

        test_model : bool, optional
            Whether tests added models, if False, the model testing will be skipped. The default is True.
            The tests use minimal sample sizes, this can cause error for some models.

        pipe_parallel_for_modeling : bool
            If True, the pipeline-level parallel computing is not applied to modeling.
            If modeling method using multiprocessing internally or using GPU acceleration, set True to prevent duplicated parallelism.

        dump_backup : bool, optional
            Create backup files of result with time stamp. The default is False.

        step_result : bool, optional
            Whether step results of the chains are kept, if False, the step results will be removed after preprocessing. The default is True.

        resume : bool
            If True, the computation resumes from last saved checkpoint in the preprocessing progress logs.
            Useful for continuing interrupted runs to avoid redundant processing. The default is False.

        sample_data_to_csv : bool
            If True, the sample data after preprocessing is written to CSV files in addition. The default is True.

        show_progress : bool, optional
            Show processing progress. The default is True.

        save_config : bool, optional
            Save SpecPipe configurations. The default is True.

        summary : bool, optional
            Whether to summarize the preprocessed data, performance metrics and marginal performance metrics.
            The marginal performances of different processes at each step are compared using Mann-Whitney U test.
            The default is True.

        geo_reference_warning: bool, optional
            Whether to suppress GeoReferenceWarning, if False, the warning is suppressed. The default is False.

        skip_test: bool, optional
            Whether skip test execution completely. Test execution valiates every processing chain and serves as a safeguard against runtime errors in a long formal execution.
        """  # noqa: E501

        # Validate processor
        if n_processor < 0:
            if os.name == "nt" and n_processor != -2:
                n_processor = 1
            else:
                n_processor = max(1, cpu_count() - 1)

        # Prompt "if __name__ == '__main__':" protection for windows multiprocessing
        if n_processor > 1:
            if os.name == "nt":
                print(
                    "\n\n\n\
                    >>> WARNING: Windows users must run multiprocessing within block \n\nif __name__ == '__main__':\
                    \n\n Please make sure all of your main codes in the script are placed within this block. \n\n",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(2)
                warnings.warn(
                    "Windows users must run multiprocessing within block \n\nif __name__ == '__main__': \n\n\
                    Please make sure all of your main codes in the script are placed within this block.",
                    UserWarning,
                    stacklevel=2,
                )

        # Suppress NotGeoreferencedWarning
        if not geo_reference_warning:
            warnings.simplefilter("ignore", NotGeoreferencedWarning)

        # Save configurations
        if save_config:
            self.save_pipe_config(copy=True)

        # Test process
        if not skip_test:
            print("\n========= Test added chains =========\n")
            self.test_run(test_modeling=test_model)

        # Preprocessing
        print("\n========= Preprocessing samples =========\n")
        final_result_only = not step_result
        self.preprocessing(
            result_directory=result_directory,
            n_processor=n_processor,
            dump_backup=dump_backup,
            final_result_only=final_result_only,
            keep_chain_results=keep_chain_results,
            resume=resume,
            to_csv=sample_data_to_csv,
            show_progress=show_progress,
            save_config=False,
            summary=summary,
            geo_reference_warning=geo_reference_warning,
            skip_test=skip_test,
        )

        # Model evaluation
        model_procs = [proc[0] for proc in self.process if proc[3] == "model"]
        if len(model_procs) > 0:
            if pipe_parallel_for_modeling:
                ncpu_model = n_processor
            else:
                ncpu_model = 1
            print("\n\n========= Model evaluation =========\n")
            self.model_evaluation(
                n_processor=ncpu_model,
                resume=resume,
                report_directory=result_directory,
                show_progress=show_progress,
                save_config=False,
                summary=summary,
            )
        else:
            print("\nNo model added, pipeline complete with preprocessing results.\n")

        # Print result dir and save corresponding targets
        if len(result_directory) > 0:
            result_dir = str(result_directory)
        else:
            result_dir = self.report_directory
        self._save_sample_targets(result_dir)
        print(f"\n\nPipeline running complete, result stored in:\n{result_dir}")

        warnings.simplefilter("default", NotGeoreferencedWarning)

    # For get results in console
    def report_summary(self) -> dict:
        """
        Retrieve summary of reports in the console, including performance summary and marginal performances among processes.
        The output is a dictionary of report dataframes, includes:
            For regression:
                - Performance summary
                - Marginal R2 of the steps with multiple processes.
            For classification:
                - Macro- and micro-average performance summary
                - Marginal macro- and micro-average AUC of the steps with multiple processes.
        """  # noqa: E501
        # Validate pipeline running completion
        if os.path.exists(unc_path(self.report_directory + "Modeling/sample_targets_stats.csv")):
            result = group_stats_report(self.report_directory)
            assert isinstance(result, dict)
            return result
        else:
            print("Unfinished or incomplete pipeline running results. Cannot retrieve summary reports.")
            return {}

    # For get results in console
    def report_chains(self) -> list[dict]:
        """
        Retrieve major model evaluation reports of every processing chain in the console.
        The output is a list of dictionaries. Each dictionary of processing chain includes:
            For regression:
                - Processes of the chain
                - Validation results
                - Performance metrics
                - Residual analysis
                - Influence analysis (if available)
                - Scatter plot
                - Residual plot
            For classification:
                - Processes of the chain
                - Validation results
                - Residual analysis
                - Influence analysis
                - ROC curves
        """
        # Validate pipeline running completion
        if os.path.exists(unc_path(self.report_directory + "Modeling/sample_targets_stats.csv")):
            result = core_chain_report(self.report_directory)
            assert isinstance(result, list)
            return result
        else:
            print("Unfinished or incomplete pipeline running results. Cannot retrieve chain reports.")
            return []
