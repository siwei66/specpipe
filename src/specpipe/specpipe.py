# -*- coding: utf-8 -*-
"""
SpecPipe - Pipeline management and implemention module for spectral image processing and modeling

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Basics
import copy
import json
import os
import shutil

# For local test - delete after use
import warnings
from datetime import datetime

# import dill
from functools import partial

# Typing
from typing import Annotated, Any, Callable, Literal, Optional, Union, overload

# Plots
import matplotlib.pyplot as plt
import numpy as np

# Calculation
import pandas as pd

# Raster
import rasterio

# Multiprocessing
from pathos.multiprocessing import ProcessingPool, cpu_count
from rasterio.windows import Window
from tqdm import tqdm

# Self
from .modeleva import ModelEva
from .rasterop import croproi, pixel_apply
from .roistats import ROISpec, minbbox
from .specexp import SpecExp
from .specio import (
    arraylike_validator,
    dataframe_validator,
    df_to_csv,
    dump_vars,
    load_vars,
    simple_type_validator,
)

# For multiprocessing
global ModelEva

# %% Static functions for SpecPipe


@simple_type_validator
def _target_type_validation_for_serialization(
    pc_sample_list: list[tuple[str, tuple[int], Any, Annotated[Any, arraylike_validator(ndim=1)]]],
) -> list[tuple[str, tuple[int], Union[str, int, bool, float], Annotated[Any, arraylike_validator(ndim=1)]]]:
    """Fix typing for integer after dill serialization."""
    for loaded_i, loaded_sample in enumerate(pc_sample_list):
        loaded_y = loaded_sample[2]
        # Behavior check to fix type
        test_value_y = loaded_y
        try:
            _ = test_value_y + 1
            if "." in str(loaded_y):
                loaded_y = float(loaded_y)
            elif str(loaded_y) in ["True", "False"]:
                loaded_y = bool(loaded_y)
            else:
                loaded_y = int(loaded_y)
        except Exception:
            loaded_y = str(loaded_y)
        # Update target value
        pc_sample_list[loaded_i] = (loaded_sample[0], loaded_sample[1], loaded_y, loaded_sample[3])
    return pc_sample_list


# Data_level validator
# Data_level: 0 - image (path), \
# 1 - pixel_spec (1D), 2 - pixel_specs_array (2D), 3 - pixel_specs_tensor (3D), 4 - pixel_hyperspecs_tensor (3D), \
# 5 - image_ROI (img_path + ROI coords), 6 - ROI_specs (2D), 7 - spec1d (1D spec stats)
# @validate_call
@simple_type_validator
def _dl_val(data_level: Union[str, int]) -> tuple[int, str]:
    """
    Data level validator, input data level name or index number, return (data level index, data level name).
    """
    # Validate data_level
    data_levels = [
        "image",
        "pixel_spec",
        "pixel_specs_array",
        "pixel_specs_tensor",
        "pixel_hyperspecs_tensor",
        "image_roi",
        "roi_specs",
        "spec1d",
        "model",
    ]
    data_level_n = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    if type(data_level) is str:
        if data_level not in data_levels:
            raise ValueError(f"data_level must be one of {data_levels}, but got: {data_level}")
        else:
            dlind = data_levels.index(data_level)
    elif type(data_level) is int:
        if data_level not in data_level_n:
            raise ValueError(f"data_level number must be one of {data_level_n}, but got: {data_level}")
        else:
            dlind = data_level
    return (dlind, data_levels[dlind])


# Preprocessing of single sample using all chains
@overload
def _preprocessing_sample(
    sample_data: dict,
    process: list[tuple[str, str, str, str, int, Any, int, int]],
    custom_chains: list,
    process_chains: list,
    specpipe_report_directory: str,
    *,
    dump_result: Literal[True] = True,
    return_result_path: Literal[True] = True,
    dump_backup: bool = False,
    return_step_result: Literal[False] = False,
    final_result_only: bool = True,
    is_test_run: bool = False,
    dump_directory: str = "",
    # Update progress status, use in a processing loop for resume
    update_progress_log: bool = False,
    # Explicitly load function for multiprocessing
    _dl_val: Callable = _dl_val,
    pixel_apply: Callable = pixel_apply,
    dump_vars: Callable = dump_vars,
) -> str: ...


@overload
def _preprocessing_sample(
    sample_data: dict,
    process: list[tuple[str, str, str, str, int, Any, int, int]],
    custom_chains: list,
    process_chains: list,
    specpipe_report_directory: str,
    *,
    dump_result: Literal[True] = True,
    return_result_path: Literal[True] = True,
    dump_backup: bool = False,
    return_step_result: Literal[True] = True,
    final_result_only: bool = True,
    is_test_run: bool = False,
    dump_directory: str = "",
    # Update progress status, use in a processing loop for resume
    update_progress_log: bool = False,
    # Explicitly load function for multiprocessing
    _dl_val: Callable = _dl_val,
    pixel_apply: Callable = pixel_apply,
    dump_vars: Callable = dump_vars,
) -> tuple[str, dict]: ...


@overload
def _preprocessing_sample(
    sample_data: dict,
    process: list[tuple[str, str, str, str, int, Any, int, int]],
    custom_chains: list,
    process_chains: list,
    specpipe_report_directory: str,
    *,
    dump_result: Literal[True] = True,
    return_result_path: Literal[False] = False,
    dump_backup: bool = False,
    return_step_result: Literal[True] = True,
    final_result_only: bool = True,
    is_test_run: bool = False,
    dump_directory: str = "",
    # Update progress status, use in a processing loop for resume
    update_progress_log: bool = False,
    # Explicitly load function for multiprocessing
    _dl_val: Callable = _dl_val,
    pixel_apply: Callable = pixel_apply,
    dump_vars: Callable = dump_vars,
) -> dict: ...


@overload
def _preprocessing_sample(
    sample_data: dict,
    process: list[tuple[str, str, str, str, int, Any, int, int]],
    custom_chains: list,
    process_chains: list,
    specpipe_report_directory: str,
    *,
    dump_result: Literal[True] = True,
    return_result_path: Literal[False] = False,
    dump_backup: bool = False,
    return_step_result: Literal[False] = False,
    final_result_only: bool = True,
    is_test_run: bool = False,
    dump_directory: str = "",
    # Update progress status, use in a processing loop for resume
    update_progress_log: bool = False,
    # Explicitly load function for multiprocessing
    _dl_val: Callable = _dl_val,
    pixel_apply: Callable = pixel_apply,
    dump_vars: Callable = dump_vars,
) -> None: ...


@overload
def _preprocessing_sample(
    sample_data: dict,
    process: list[tuple[str, str, str, str, int, Any, int, int]],
    custom_chains: list,
    process_chains: list,
    specpipe_report_directory: str,
    *,
    dump_result: Literal[False] = False,
    return_result_path: bool = True,
    dump_backup: bool = False,
    return_step_result: bool = False,
    final_result_only: bool = True,
    is_test_run: bool = False,
    dump_directory: str = "",
    # Update progress status, use in a processing loop for resume
    update_progress_log: bool = False,
    # Explicitly load function for multiprocessing
    _dl_val: Callable = _dl_val,
    pixel_apply: Callable = pixel_apply,
    dump_vars: Callable = dump_vars,
) -> dict: ...


# Preprocessing of single sample using all chains
# SpecPipe attributes - argumente relation:
#     sample_data = SpecPipe.sample_data | SpecPipe.pretest_data
#     process = SpecPipe.process
#     custom_chains = SpecPipe.custom_chains
#     process_chains = SpecPipe.process_chains
#     specpipe_report_directory = SpecPipe.spec_exp.report_directory
# Sample data format - ROI: {ID, label, target, img_path, roi_coords}
# Sample data format - standalone spec: {ID, label, target, spec1d: tuple}
# Sample data format - test: {img_path, test_img_path, roi_coords, test_roi_coords, roitable, spec1d}
# @validate_call
@simple_type_validator
def _preprocessing_sample(  # noqa: C901
    sample_data: dict,
    process: list[tuple[str, str, str, str, int, Any, int, int]],
    custom_chains: list,
    process_chains: list,
    specpipe_report_directory: str,
    *,
    dump_result: bool = True,
    return_result_path: bool = True,
    dump_backup: bool = False,
    return_step_result: bool = False,
    final_result_only: bool = True,
    is_test_run: bool = False,
    dump_directory: str = "",
    # Update progress status, use in a processing loop for resume
    update_progress_log: bool = False,
    # Explicitly load function for multiprocessing
    _dl_val: Callable = _dl_val,
    pixel_apply: Callable = pixel_apply,
    dump_vars: Callable = dump_vars,
) -> Union[str, dict, tuple[str, dict], None]:
    """
    Preprocessing of single sample using all chains of SpecPipe.

    Parameters
    ----------
    sample_data : dict
        'sample_data' or 'pretest_data' of SpecPipe.

    process : list[tuple[str, str, str, str, int, Callable, int, int]]
        'process' of SpecPipe.

    custom_chains : list
        'custom_chains' of SpecPipe.

    process_chains : list
        'process_chains' of SpecPipe.

    specpipe_report_directory : str
        'report_directory' of SpecExp specified for SpecPipe.

    dump_result : bool, optional
        Whether result is dumped to dill file. The default is True.

    return_result_path : bool, optional
        Whether the result file path is returned. The default is True.

    dump_backup : bool, optional
        Whether backup for the step result is dumped. The backup file is named with the datetime of dumping.

    return_step_result : bool, optional
        Whether the step result is returned. The default is False. If both path and step_result are to be returned, (path, step_result) is returned.

    final_result_only : bool, optional
        Whether only the final result of chains is dumped, if True, only the result from the last preprocessing step is dumped.
        The default is True.

    is_test_run : bool, optional
        True if test data is applied for chain testing. The default is False.

    dump_directory : str, optional
        If '', default directory in specpipe_report_directory will be used as result file directory, or this directory is used for resulting files. The default is ''.

    update_progress_log : bool = False
        Whether to update progress log files, use to enable resume. The default is False.
    """  # noqa: E501
    # Func dependencies for func 'partial' for multiprocessing
    import copy
    import os
    from datetime import datetime

    import numpy as np

    try:
        # Break and resume testing mode
        preprocess_resume_test_num = int(os.getenv("SPECPIPE_PREPROCESS_RESUME_TEST_NUM", "-1"))
        if preprocess_resume_test_num > 0:
            if preprocess_resume_test_num > 1:
                raise ValueError("Preprocessing resume test raise")
            else:
                preprocess_resume_test_num = preprocess_resume_test_num + 1
                os.environ["SPECPIPE_PREPROCESS_RESUME_TEST_NUM"] = str(preprocess_resume_test_num)

        # Validate sample data label
        if (sample_data["label"] == "") or (sample_data["label"] == "-"):
            sample_data_label = sample_data["ID"]
        else:
            sample_data_label = sample_data["label"]

        # Get methods
        methods = np.array(process)

        # Get testing chain
        if len(custom_chains) > 0:
            chains = custom_chains
        elif len(process_chains) > 0:
            chains = process_chains
        else:
            raise ValueError("\nNo process added")

        # Validate chains
        chain_length = len(process_chains[0])
        for chain in process_chains:
            if len(chain) != chain_length:
                raise ValueError(
                    f"Inconsistent steps of processing in chain: {chain} \
                        \nExpected number of steps: {chain_length}, got: {len(chain)}"
                )

        # Chains
        # [(process 1 ID of step 1, process 1 ID of step 2,...), (process 2 ID of step 1, process 1 ID of step 2,...), ...]  # noqa: E501
        # Status vector: Check previous steps,
        # once identical or previous process completed, avoid computing repeatly but use the previous result.
        # [[Step1:[preceding processes 1],[preceding processes 2],...],[Step2:...],...]
        model_ids = [pit[0] for pit in process if pit[3] == "model"]
        if len(model_ids) > 0:
            n_model_step = 1
        else:
            n_model_step = 0
        # Number of preprocessing steps
        preprocess_chain_length = len(chains[0]) - n_model_step
        if preprocess_chain_length < 1:
            raise ValueError("No preprocessing process found.")
        calc_status: list[list] = [[] for _ in range(preprocess_chain_length)]
        status_results: list[list] = [[] for _ in range(preprocess_chain_length)]

        # For every chain implement processing pipeline
        for chain_ind, chain in enumerate(chains):
            chain_result: list = []
            step_procs: list = []
            # For every step exclude modeling step
            for stepi in range(len(chain) - n_model_step):
                step = chain[stepi]
                # Create new step_procs and sample result for the step
                step_procs = copy.deepcopy(step_procs)
                chain_result = copy.deepcopy(chain_result)
                # Note down processes of every step in a chain
                step_procs.append(step)
                # Inherit calculated result to avoid repeating calculation
                if step_procs in calc_status[stepi]:
                    chain_result = [srt[4] for srt in status_results[stepi] if srt[1] == step_procs]
                # New step calculation
                else:
                    # Get method and method info
                    # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
                    method_item = methods[methods[:, 0] == step, :][0]
                    dl_in = _dl_val(method_item[2])[0]
                    dl_out = _dl_val(method_item[3])[0]
                    method_func = method_item[5]
                    # Get input data of the step
                    # Pretest_data: [img_path, test_img_path, roi_coords, test_roi_coords, roitable, spec1d]
                    if len(chain_result) == 0:
                        if dl_in <= 5:
                            step_input_data = sample_data["img_path"]
                        elif dl_in == 6:
                            step_input_data = sample_data["roitable"]
                        elif dl_in == 7:
                            step_input_data = sample_data["spec1d"]
                    else:
                        step_input_data = chain_result[-1]
                    # Preprocessing computing
                    try:
                        if dl_out < 8:
                            # Apply the step process function
                            if dl_in == 0:
                                chain_result.append(method_func(step_input_data))
                            elif dl_in == 1:
                                chain_result.append(pixel_apply(step_input_data, method_func, "spec", progress=False))
                            elif dl_in == 2:
                                chain_result.append(pixel_apply(step_input_data, method_func, "array", progress=False))
                            elif dl_in == 3:
                                chain_result.append(pixel_apply(step_input_data, method_func, "tensor", progress=False))
                            elif dl_in == 4:
                                chain_result.append(
                                    pixel_apply(step_input_data, method_func, "tensor_hyper", progress=False)
                                )
                            elif dl_in == 5:
                                chain_result.append(method_func(step_input_data, sample_data["roi_coords"]))
                            elif (dl_in >= 6) & (dl_in <= 7):
                                chain_result.append(method_func(step_input_data))
                            # Save calculated step results
                            # Status result: (0 - step_id, 1 step_procs, 2 dl_in, 3 dl_out, 4 sample_result)
                            # Step_id as str of procs id
                            step_id = ""
                            for proc_id in step_procs:
                                step_id = step_id + "proc_" + str(proc_id) + "-"
                            # Store step result and calculation status
                            status_results[stepi].append((step_id, step_procs, dl_in, dl_out, chain_result[-1]))
                            calc_status[stepi].append(step_procs)
                    except Exception as e:
                        method_item = tuple(method_item)
                        method_item_out = method_item[1:5] + (method_item[5].__class__.__name__,) + method_item[6:8]
                        raise ValueError(
                            f"\nTest failed for chain: \nChain index: {chain_ind}, \nChain: {chain};\
                                \n\nProcess ID: {step}, \nProcess item: {method_item_out}, \n\nError message: \n{e}"
                        ) from e

        # Dump test preprocessing results of current chain (chain i)
        if final_result_only:
            status_results = status_results[-1:]
        status_results_out = {
            "ID": sample_data["ID"],
            "label": sample_data_label,
            "target": sample_data["target"],
            "status_results": status_results,
        }

        # Dump step results
        if is_test_run:
            dir_name = "test_run"
            file_name = "PreprocessingTestingResult"
        else:
            dir_name = "Preprocessing"
            file_name = f"PreprocessingResult_sample_{sample_data_label}"
        # Output path
        if len(dump_directory) > 0:
            sdir = dump_directory
        else:
            sdir = specpipe_report_directory + f"{dir_name}/Step_results/"
        if not os.path.exists(sdir):
            os.makedirs(sdir)
        if dump_result:
            chain_result_path = sdir + f"{file_name}.dill"
            dump_vars(chain_result_path, status_results_out, backup=dump_backup)

        # Update progress
        step_dir = specpipe_report_directory + "Preprocessing/Step_results/"
        log_dir_path = step_dir + "Preprocess_progress_logs/"
        log_fp = log_dir_path + sample_data["ID"]
        if update_progress_log:
            if not os.path.exists(log_dir_path):
                os.makedirs(log_dir_path)
            if not os.path.exists(log_fp):
                with open(log_fp, "w") as f:
                    f.write("")

        # Return result file path and step results if required
        if dump_result:
            # Return dumped file path and step results
            if return_result_path & return_step_result:
                return chain_result_path, status_results_out
            elif return_result_path & (not return_step_result):
                return chain_result_path
            elif (not return_result_path) & return_step_result:
                return status_results_out
            else:
                return None
        # Return step results only
        else:
            return status_results_out
            if not return_step_result:
                raise warnings.warn(
                    "When dump_result is False, \
                        the result is always returned and the return_step_result argument is ignored.",
                    UserWarning,
                    stacklevel=3,
                )

    # Error handling
    except Exception as e:
        # Log directory
        if is_test_run:
            dir_name = "test_run"
        else:
            dir_name = "Preprocessing"
        errdir = specpipe_report_directory + f"{dir_name}/Step_results/Error_logs/"
        if not os.path.exists(errdir):
            os.makedirs(errdir)
        cts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # PID for multiprocessing
        pid = os.getpid()
        error_log_path = errdir + f"error_{cts}_pid_{pid}.log"
        # Validate sample data label
        if (sample_data["label"] == "") or (sample_data["label"] == "-"):
            sample_data_label = sample_data["ID"]
        else:
            sample_data_label = sample_data["label"]
        # Write error log
        err_msg = f"Failed in the preprocessing of '{sample_data_label}', error message: \n\n{str(e)}"
        with open(error_log_path, "w") as f:
            f.write(err_msg)
        raise ValueError(e) from e


# Model method wrapper, sample_list as method input
class _ModelMethod:
    def __init__(
        self,
        # Model configuration
        model_label: str,
        is_regression: Optional[bool],
        x_shape: Optional[tuple[int]],
        report_dir: str,
        method: object,
        result_backup: bool,
        # Modeling parameters
        validation_method: str,
        unseen_threshold: Optional[float],
        # Model evaluation reporting parameters
        data_split_config: Union[str, dict[str, Any]],
        validation_config: Union[str, dict[str, Any]],
        metrics_config: Union[str, dict[str, Any], None],
        roc_plot_config: Union[str, dict, None],
        scatter_plot_config: Union[str, dict[str, Any], None],
        residual_config: Union[str, dict[str, Any], None],
        residual_plot_config: Union[str, dict[str, Any], None],
        influence_analysis_config: Union[str, dict[str, Any], None],
    ) -> None:
        self.__name__ = model_label
        self.model_label = model_label
        self.is_regression = is_regression
        self.input_shape = x_shape
        self.report_dir = report_dir
        self.method = method
        self.result_backup = result_backup
        self.validation_method = validation_method
        self.unseen_threshold = unseen_threshold
        self.data_split_config = data_split_config
        self.validation_config = validation_config
        self.metrics_config = metrics_config
        self.roc_plot_config = roc_plot_config
        self.scatter_plot_config = scatter_plot_config
        self.residual_config = residual_config
        self.residual_plot_config = residual_plot_config
        self.influence_analysis_config = influence_analysis_config

    """
    Model method wrapper to specify modeling and model evaluation parameters.
    """

    # @validate_call
    @simple_type_validator
    def evaluation(
        self,
        sample_list: list[
            tuple[
                str,
                tuple[int],
                Union[str, int, bool, float],
                Annotated[Any, arraylike_validator(ndim=1)],
            ]
        ],
        data_label: str,
        report_directory: Optional[str] = None,
        modeleva: type = ModelEva,
        silent_all: bool = False,
    ) -> None:
        """
        Evaluation of specified model. Configured at _ModelMethod instance.

        Parameters
        ----------
        sample_list : TYPE
            Standard sample data of SpecPipe for modeling.
        data_label : str
            Label for the specified dataset.
        report_directory : Optional[str], optional
            Report_directory for model evaluation reports. The default is using report_directory of the _ModelMethod instance.
        """  # noqa: E501
        if report_directory is None:
            report_directory = self.report_dir
        model_eva = modeleva(
            sample_list=sample_list,
            model=self.method,
            validation_method=self.validation_method,
            report_directory=report_directory,
            model_label=self.model_label,
            data_label=data_label,
            is_regression=self.is_regression,
            unseen_threshold=self.unseen_threshold,
            result_backup=self.result_backup,
            silent_all=silent_all,
        )
        if self.is_regression:
            model_eva.regressor_evaluation(
                data_split_config=self.data_split_config,
                validation_config=self.validation_config,
                metrics_config=self.metrics_config,
                scatter_plot_config=self.scatter_plot_config,
                residual_config=self.residual_config,
                residual_plot_config=self.residual_plot_config,
                influence_analysis_config=self.influence_analysis_config,
            )
        else:
            model_eva.classifier_evaluation(
                data_split_config=self.data_split_config,
                validation_config=self.validation_config,
                metrics_config=self.metrics_config,
                roc_plot_config=self.roc_plot_config,
                residual_config=self.residual_config,
                influence_analysis_config=self.influence_analysis_config,
            )

    def __call__(self, sample_list: list, data_label: str, report_directory: Optional[str] = None) -> None:
        self.evaluation(sample_list, data_label, report_directory)


# Run modeling on single dataset
# Notes: different from '_test_model',
# Sample_list item: (0 - Sample id, 1 - Original shape, 2 - Target value, 3 - Sample predictor value)
@simple_type_validator
def _model_evaluator(  # noqa: C901
    preprocess_result: list[
        tuple[
            str,
            tuple[int],
            Union[str, int, bool, float],
            Annotated[Any, arraylike_validator(ndim=1)],
        ]
    ],
    preprocess_chain: tuple,
    preprocess_chain_label: str,
    model_processes: list[tuple[str, str, str, str, int, Any, int, int]],
    specpipe_report_directory: str,
    result_directory: str = "",
    # Update progress status, use in a processing loop for resume
    update_progress_log: bool = False,
    # Import applied functions and modules
    _dl_val: Callable = _dl_val,
    load_vars: Callable = load_vars,
    dump_vars: Callable = dump_vars,
    _target_type_validation_for_serialization: Callable = _target_type_validation_for_serialization,
    modeleva: type = ModelEva,
    silent_all: bool = False,
) -> None:
    """
    Evaluation of added models on a single sample_list dataset.
    'preprocess_result' must be the standard 'sample_list' output.
    'preprocess_chain_label' serves as the 'data_label' for modeling data.
    'preprocess_chain_label' must be unique across all sample lists for modeling.
    """
    # Import applied modules
    import os
    import time

    import numpy as np

    # Break and resume testing mode
    model_resume_test_num = int(os.getenv("SPECPIPE_MODEL_RESUME_TEST_NUM", "-1"))
    if model_resume_test_num > 0:
        if model_resume_test_num > 1:
            raise ValueError("Modeling resume test raise")
        else:
            model_resume_test_num = model_resume_test_num + 1
            os.environ["SPECPIPE_MODEL_RESUME_TEST_NUM"] = str(model_resume_test_num)

    # Name reassignment
    sample_list = preprocess_result
    preprocess_chain = tuple(preprocess_chain)
    sample_list_label = preprocess_chain_label

    # Validate report directory
    if result_directory == "":
        result_directory = specpipe_report_directory
    model_result_dir = result_directory + "Modeling/"
    if not os.path.exists(model_result_dir):
        os.makedirs(model_result_dir)

    # Validate sample_list_label - Absolutely unique number if label not provided (commonly it should be given)
    if sample_list_label == "":
        if update_progress_log:
            raise ValueError(
                "Consistent and unique sample_list_label must be provided \
                    if update_progress_log set True for enabling break resuming."
            )
        sample_list_label = str(time.time_ns())[4:-2] + str(os.getpid())

    # Validate model processes
    for procit in model_processes:
        if procit[3] != "model":
            raise ValueError(f"Model process must have output data level of 'model', but got: '{procit[3]}'")
        if not callable(procit[-3]):
            raise ValueError(
                f"Invalid model evaluation method, \
                    given method is not callable : {procit[-3]}, got type : {type(procit[-3])}"
            )

    # Save preprocess chain info for the sample_list
    model_report_dir = model_result_dir + "Model_evaluation_reports/"
    if not os.path.exists(model_report_dir):
        os.makedirs(model_report_dir)
    # Chain file name
    chain_label_file_path = model_report_dir + f"{preprocess_chain_label}.txt"
    # Save chain process file
    with open(chain_label_file_path, "w") as f:
        for pci, pproc in enumerate(preprocess_chain):
            if pci < (len(preprocess_chain) - 1):
                f.write(f"{pproc}\n")
            else:
                f.write(f"{pproc}")

    # Get model processes with input data level index dl_in_ind
    # Process: [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    for modelit in model_processes:
        # Get test model
        model_methodi = modelit[5]
        # Get model input data level
        dl_in_name = _dl_val(modelit[2])[1]
        dl_in_ind = _dl_val(modelit[2])[0]
        # Get model input data shape
        input_dshape = model_methodi.input_shape
        # Validate data shape
        sample_list_shape = (len(sample_list), len(sample_list[0]))
        if input_dshape is None:
            input_dshape = sample_list_shape
        elif np.prod(sample_list_shape) != np.prod(input_dshape):
            raise ValueError(
                f"Cannot reshape sample data with shape {sample_list_shape} \
                    into specified input data shape {input_dshape} of the model.\
                    \nInput step data ID: {sample_list[0]}\nModel label: {modelit[1]}"
            )
        # Modeling
        # Sample_list item: (0 - Sample id, 1 - Original shape, 2 - Target value, 3 - Sample predictor value)
        if dl_in_ind == 7:
            # Regression
            if model_methodi.is_regression:
                model_methodi.evaluation(
                    sample_list=sample_list,
                    data_label="chain_" + sample_list_label,
                    report_directory=model_result_dir,
                    modeleva=modeleva,
                    silent_all=silent_all,
                )
            # Classification
            else:
                model_methodi.evaluation(
                    sample_list=sample_list,
                    data_label="chain_" + sample_list_label,
                    report_directory=model_result_dir,
                    modeleva=modeleva,
                    silent_all=silent_all,
                )
        else:
            raise ValueError(f"Model only accepts data level 'spec1d' as input, but got: {dl_in_name}")

    # Update progress
    log_path = model_report_dir + "modeling_progress_log.dill"
    if os.path.exists(log_path):
        modeling_progress_log = load_vars(log_path)["modeling_progress_log"]
        if preprocess_chain not in modeling_progress_log:
            modeling_progress_log.append(preprocess_chain)
        else:
            warnings.warn(
                f"Sample_list_label must be unique, got duplicated label: {sample_list_label}",
                UserWarning,
                stacklevel=3,
            )
        dump_vars(log_path, {"modeling_progress_log": modeling_progress_log}, backup=False)
    else:
        dump_vars(log_path, {"modeling_progress_log": [preprocess_chain]}, backup=False)


@simple_type_validator
def _model_evaluator_mp(
    cdp: str,
    pchains: list[tuple],
    model_processes: list[tuple[str, str, str, str, int, Any, int, int]],
    specpipe_report_directory: str,
    result_directory: str = "",
    # Update progress status, use in a processing loop for resume
    update_progress_log: bool = False,
    # Import applied functions and modules
    _model_evaluator: Callable = _model_evaluator,
    _dl_val: Callable = _dl_val,
    load_vars: Callable = load_vars,
    dump_vars: Callable = dump_vars,
    _target_type_validation_for_serialization: Callable = _target_type_validation_for_serialization,
    modeleva: type = ModelEva,
    silent_all: bool = True,
) -> None:
    """
    Evaluation of added models on a single sample_list dataset for multiprocessing.
    cdp : chain data path
    pchains : preprocessing chains
    """
    try:
        # Import applied modules
        import os
        from datetime import datetime

        # Load chain data
        pc_it = load_vars(cdp)
        pc_sample_list = pc_it["chain_res"]
        pc_sample_list = _target_type_validation_for_serialization(pc_sample_list)
        pchain = pc_it["chain_procs"]
        # Use preprocess chain ID as chain label
        pproc_chain_label = [f"Preprocessing_#{pci}" for pci, pc in enumerate(pchains) if pc == pchain][0]
        _model_evaluator(
            preprocess_result=pc_sample_list,
            preprocess_chain=pchain,
            preprocess_chain_label=pproc_chain_label,
            model_processes=model_processes,
            specpipe_report_directory=specpipe_report_directory,
            result_directory=result_directory,
            update_progress_log=update_progress_log,
            # Import applied functions
            _dl_val=_dl_val,
            load_vars=load_vars,
            dump_vars=dump_vars,
            modeleva=modeleva,
            silent_all=silent_all,
        )

    # Error handling
    except Exception as e:
        # Validate report directory
        if result_directory == "":
            result_directory = specpipe_report_directory
        model_result_dir = result_directory + "Modeling/"
        if not os.path.exists(model_result_dir):
            os.makedirs(model_result_dir)
        errdir = model_result_dir + "Error_logs/"
        if not os.path.exists(errdir):
            os.makedirs(errdir)
        cts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pid = os.getpid()
        error_log_path = errdir + f"error_{cts}_pid_{pid}.log"
        err_msg = f"\nFailed in the modeling of preprocessing chain from path '{cdp}', error message: \n\n{str(e)}\n"
        with open(error_log_path, "w") as f:
            f.write(err_msg)
        raise ValueError(e) from e


# %% Spectral Modeling Pipeline Class - SpecPipe


class SpecPipe:
    """
    Design and implement a processing and modeling pipeline for spectral experiment datasets.


    Attributes:
    -----------
    spec_exp : SpecExp
        Instance of SpecExp configuring spectral experiment datasets. See SpecExp for details.


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

    - process_chains_to_df (Alias: ls_process_chains, ls_chains)
        List process chains in a dataframe. Return the dataframe and / or print the dataframe of the process chains show in process label.
        If multiple candidate methods are added for a given position in the application sequence, a full-factorial experimental design is employed by default to construct and evaluate all possible method chains.

    - custom_chains_from_df
        Customize processing chains and update chains using chain dataframe.
        Once custom chains are created, SpecPipe will prioritize their execution, bypassing the original full-factorial chains.

    - test_run :
        Run the pipeline of all processing chains using simplified test data. This method is executed automatically prior to each formal run.

    - preprocessing :
        Implement preprocessing steps of all processing chains on the entire dataset and output modeling-ready sample_list data to files.

    - model_evaluation :
        Evaluating added models on sample data from all preprocessing chains.

    - run :
        Run pipeline of given processes on SpecExp instance (corresponding manager of spectral experiment data).
        Full-factorial test is applied to multiple processes of identical sequence.

    """  # noqa: E501

    # @validate_call
    @simple_type_validator
    def __init__(self, spec_exp: SpecExp) -> None:  # noqa: C901
        ## Experiment data manager - SpecExp
        # Validate SpecExp
        if type(spec_exp) is not SpecExp:
            raise TypeError(f"spec_exp must be a SpecExp object, but got: {type(spec_exp)}")

        # Validate report diretory
        if not os.path.isdir(spec_exp._report_directory):
            raise ValueError(f"\nReport directory of given SpecExp is invalid: \n'{spec_exp._report_directory}'")

        # Validate group
        if len(spec_exp.groups) == 0:
            raise ValueError("No group is found in given SpecExp")

        # Validate sample data configs
        if len(spec_exp.standalone_specs_sample) == 0:
            if len(spec_exp.images) == 0:
                raise ValueError("Neither image path nor standalone spectrum is found in given SpecExp")
            elif len(spec_exp.rois_sample) == 0:
                raise ValueError("Neither sample ROI nor standalone spectrum is found in given SpecExp")
            for g in spec_exp.groups:
                group_images = spec_exp.ls_images(group_name=g, return_dataframe=True, print_result=False)
                group_rois = spec_exp.ls_rois(
                    group_name=g, roi_type="sample", return_dataframe=True, print_result=False
                )
                if len(group_images) == 0:
                    raise ValueError(f"Neither image nor standalone spectrum is found in group: '{g}'")
                elif len(group_rois) == 0:
                    raise ValueError(f"Neither image sample ROI nor standalone spectrum is found in group: '{g}'")
        else:
            if len(spec_exp.images) > 0 or len(spec_exp.rois_sample) > 0:
                raise ValueError(
                    "Hybrid samples from both standalone spectra and spectral images \
                        is not allowed by SpecPipe pipeline.\
                        \nPlease provide either pure image samples or standalone spectrum samples"
                )
            for g in spec_exp.groups:
                if (
                    len(
                        spec_exp.ls_standalone_specs(
                            group_name=g,
                            use_type="sample",
                            print_result=False,
                            return_dataframe=True,
                        )
                    )
                    == 0
                ):
                    raise ValueError(f"No spectrum is found in group: '{g}'")

        # Validate sample target values
        if len(spec_exp.sample_targets) == 0:
            raise ValueError("No sample target value is found in given SpecExp")

        # SpecExp
        # SpecExp._groups: [0 group]
        # SpecExp._images: [0 id, 1 group, 2 image_name, 3 image_use_type, 4 image_path]
        # SpecExp._rois: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
        self._spec_exp: SpecExp = spec_exp

        # Sample target values
        # [0 fixed sample id, 1 user assinged labels, 2 Target values]
        self._sample_targets: list[tuple[str, str, Union[str, bool, int, float]]] = spec_exp.sample_targets

        # Report directory
        self._report_directory: str = self.spec_exp._report_directory

        # Processes
        # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
        self._process: list[tuple[str, str, str, str, int, Any, int, int]] = []

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
        if os.path.exists(value):
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
    def spec_exp(self, value: SpecExp) -> None:
        if isinstance(value, SpecExp):
            self._spec_exp = value
        else:
            raise ValueError(f"{self.__class__.__name__}.spec_exp must be a SpecExp instance")

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
    def sample_targets(self) -> list[tuple[str, str, Union[str, bool, int, float]]]:
        return self._sample_targets

    @sample_targets.setter
    def sample_targets(self, value: list[tuple[str, str, Union[str, bool, int, float]]]) -> None:
        raise ValueError("sample_targets cannot be modified in SpecPipe, please update using 'SpecExp' instead")

    @property
    def process(self) -> list[tuple[str, str, str, str, int, Any, int, int]]:
        return self._process

    @process.setter
    def process(self, value: list[tuple[str, str, str, str, int, Any, int, int]]) -> None:
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

    # Process configuration ===================================

    # Generate test data of each level for process validation
    # From data levels:
    # 0 - image (path), \
    # 1 - pixel_spec (1D), 2 - pixel_specs_array (2D), 3 - pixel_specs_tensor (3D), 4 - pixel_hyperspecs_tensor (3D), \
    # 5 - image_ROI (img_path + ROI coords), 6 - ROI_specs (2D), 7 - spec1d (1D spec stats)
    # Pretest_data: [img_path, test_img_path, roi_coords, test_roi_coords, roitable, spec1d]
    # @validate_call
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
            if not os.path.exists(sdir):
                os.makedirs(sdir)

            # Save test image
            test_img_path = sdir + "test_images." + img_path.split(".")[-1]
            with open(sdir + "pre_execution_data.json", "w") as f:
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
            td2.to_csv(sdir + "Pre_execution_data_roi_specs.csv", index=False)

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
            if not os.path.exists(sdir):
                os.makedirs(sdir)

            # Testing spectrum table
            td2 = pd.DataFrame([list(spec1d)], columns=[("Band_" + str(i + 1)) for i in range(len(spec1d))])
            td2.to_csv(sdir + "Pre_execution_data_standalone_specs.csv", index=False)

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
    # @validate_call
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
                    result = method(test_img_path)
                elif dl_in == 1:
                    result = pixel_apply(test_img_path, method, "spec", progress=False)
                elif dl_in == 2:
                    result = pixel_apply(test_img_path, method, "array", progress=False)
                elif dl_in == 3:
                    result = pixel_apply(test_img_path, method, "tensor", progress=False)
                elif dl_in == 4:
                    result = pixel_apply(test_img_path, method, "tensor_hyper", progress=False)
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
                if os.path.exists(result):
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
        application_sequence = max(app_seqs) + 1
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
    # @validate_call
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
                    Method directly applied to raster images, accepting raster image path as input.

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
        if dl_in_ind >= 8:
            raise ValueError("Input data level cannot be 'model' or 8 (corresponding index).")
        dl_out = _dl_val(output_data_level)
        dl_out_name = dl_out[1]
        dl_out_ind = dl_out[0]
        if dl_out_name == "image_roi":
            raise ValueError(
                f"Output data level '{dl_out_name}' is not supported, \
                    as dynamic ROIs are not supported currently. \
                    Please write the generated ROIs to files and start a new SpecPipe \
                    using SpecExp with the resulting ROI files."
            )
        if dl_out_ind < dl_in_ind:
            raise ValueError(
                f"Output_data_level cannot precede the input_data_level in the processing pipeline, \
                    got input data level: '{input_data_level}', output data level: '{output_data_level}'"
            )

        # Validate sequence
        if (application_sequence < 0) | (application_sequence > 999999):
            raise ValueError(
                f"Application sequence must be within [0, 1,000,000), \
                    got: {application_sequence}"
            )

        # Full application sequence
        fapp_seq = 1000000 * dl_in_ind + application_sequence

        # Validate existed process labels
        # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
        existed_proc_num = [0]
        f_fapp_seq, l_fapp_seq = 0, np.inf
        f_out_dl = None
        l_in_dl = None
        if len(self._process) > 0:
            for pr in self._process:
                # Previous process
                if (pr[6] < fapp_seq) & (pr[6] > f_fapp_seq):
                    f_fapp_seq = pr[6]
                    f_out_dl = pr[3]
                # Subsequent process
                if (pr[6] > fapp_seq) & (pr[6] < l_fapp_seq):
                    l_fapp_seq = pr[6]
                    l_in_dl = pr[2]
                # Validate consistency of output with identical input data level and application sequence
                if pr[6] == fapp_seq:
                    if dl_out_name != pr[3]:
                        raise ValueError(
                            f"Methods with identical input data level (here: '{dl_in_name}') \
                                and application sequence (here: '{application_sequence}') \
                                must have identical output data levels. \nGot output data level: \
                                '{dl_out_name}' \nconflicted with process item: \nProcess ID: {pr[0]}\
                                \nOutput data level: '{pr[3]}'"
                        )
                # Get existed process number (repeat number)
                if (pr[2] == dl_in_name) & (pr[4] == application_sequence):
                    existed_proc_num.append(pr[7])

        # Validate I/O data level of previous and subsequent processes
        if f_out_dl is not None:
            if dl_in_ind <= 4:
                if _dl_val(f_out_dl)[0] > 4:
                    raise ValueError(
                        f"The specified input data level '{dl_in_name}' of added process \
                            is inconsistent with the output data level '{f_out_dl}' of the previous process, \
                            the input data level of added process must be image levels (0~4)."
                    )
            else:
                if dl_in_ind >= 6 and dl_in_ind != _dl_val(f_out_dl)[0]:
                    raise ValueError(
                        f"The specified input data level '{dl_in_name}' of added process \
                            is inconsistent with the output data level '{f_out_dl}' of the previous process, \
                            they must be identical."
                    )
        if l_in_dl is not None:
            if _dl_val(l_in_dl)[0] <= 4:
                if dl_out_ind > 4:
                    raise ValueError(
                        f"The specified output data level '{dl_out_name}' of added process \
                            is inconsistent with the input data level '{l_in_dl}' of the subsequent process, \
                            the output data level of added process must be image levels (0~4)."
                    )
            if _dl_val(l_in_dl)[0] >= 6 and dl_out_ind != _dl_val(l_in_dl)[0]:
                raise ValueError(
                    f"The specified output data level '{dl_out_name}' of added process \
                        is inconsistent with the input data level '{l_in_dl}' of the subsequent process, \
                        they must be identical."
                )

        # Generate process number
        proc_num_new = max(existed_proc_num) + 1

        # Build process ID
        proc_id = str(dl_in_ind) + "_" + str(application_sequence) + "_%#" + str(proc_num_new)

        # Validate Process
        if dl_out_ind < 8:
            try:
                proc_method = self._process_validator(method, input_data_level, output_data_level)
            except Exception as e:
                if test_error_raise:
                    raise ValueError(f"Method testing fails: \n{e}") from e
                else:
                    warnings.warn(
                        f"Method fails on '{input_data_level}' testing data: \n{e}", UserWarning, stacklevel=2
                    )

        # Model method constructor
        else:
            # Get report dir
            report_dir = self.report_directory
            # Validate model label
            if model_label is None:
                model_label = method.__class__.__name__
            else:
                process_label = model_label
            # Validate is_regression
            if is_regression is None:
                try:
                    self.sample_targets[0][-1] + 1  # type: ignore[operator]
                    # Behavior-based type check after serialization, safe approach for serialization
                    is_regression = True
                except Exception:
                    is_regression = False
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
            for p_item in self._process:
                p_item = tuple(p_item)
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
    # @validate_call
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
        List process chains. A dataframe of process IDs is returned, each row represents a processing chain.

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
            return df_chains
        else:
            print("No process chain found")
            return None

    # Alias
    process_chains_to_df = ls_process_chains
    ls_chains = ls_process_chains

    # Read process chains from dataframe
    # @validate_call
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

    # Get matched and unmatched process items
    # Format of associated attribute:
    # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    # Data levels:
    # 0 - image (path), \
    # 1 - pixel_spec (1D), 2 - pixel_specs_array (2D), 3 - pixel_specs_tensor (3D), 4 - pixel_hyperspecs_tensor (3D), \
    # 5 - image_ROI (img_path + ROI coords), 6 - ROI_specs (2D), 7 - spec1d (1D spec stats)
    # @validate_call
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
    # @validate_call
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
    # @validate_call
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
    # @validate_call
    @simple_type_validator
    def test_run(
        self,
        test_modeling: bool = True,
        return_result: bool = False,
        dump_result: bool = True,
        dump_backup: bool = False,
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

        # Preprocessing test
        status_results: dict = _preprocessing_sample(  # type: ignore[call-overload]
            sample_data=test_data,
            process=self.process,
            custom_chains=self.custom_chains,
            process_chains=self.process_chains,
            specpipe_report_directory=self.spec_exp.report_directory,
            dump_result=dump_result,
            return_result_path=False,
            dump_backup=dump_backup,
            return_step_result=True,
            final_result_only=False,
            is_test_run=True,
        )
        # Multiple dynamic bool switches passed to '_preprocessing_sample' that fails for type 'Literal' in overloads

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
                    with open(specpipe_report_directory + f"test_run/Preprocess_chain_#{pci}.txt", "w") as f:
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
                        if not os.path.exists(model_report_dir):
                            os.makedirs(model_report_dir)
                        # Build testing sample list
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
    # @validate_call
    @simple_type_validator
    def preprocessing(
        self,
        result_directory: str = "",
        n_processor: int = 1,
        dump_backup: bool = False,
        final_result_only: bool = True,
        keep_chain_results: bool = True,
        resume: bool = False,
        to_csv: bool = True,
        show_progress: bool = True,
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
        """  # noqa: E501
        # Prompt "if __name__ == '__main__':" protection for windows multiprocessing
        if n_processor > 1:
            if os.name == "nt":
                warnings.warn(
                    "Windows users must run multiprocessing within block \n\nif __name__ == '__main__': \n\n\
                    Please make sure all of your main codes in the script are placed within this block.",
                    UserWarning,
                    stacklevel=2,
                )

        # Added chain testing
        if not self.tested:
            self.test_run(test_modeling=True, return_result=False, dump_result=False, dump_backup=False)

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

    @simple_type_validator
    def _preprocessor(  # noqa: C901
        self,
        result_directory: str = "",
        n_processor: int = 1,
        dump_backup: bool = False,
        final_result_only: bool = True,
        resume: bool = False,
        show_progress: bool = True,
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
        if not os.path.exists(step_dir):
            os.makedirs(step_dir)
        # log dir for resume
        log_dir_path = step_dir + "Preprocess_progress_logs/"
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)

        # Check running log and subset sample data
        if not resume:
            rest_sample_data = self.sample_data
        else:
            if not os.path.exists(log_dir_path):
                os.makedirs(log_dir_path)
                finished_samples = []
            else:
                finished_samples = [
                    f.split(".")[0] for f in os.listdir(log_dir_path) if os.path.isfile(log_dir_path + f)
                ]
                existed_samples = [sd["ID"] for sd in self.sample_data]
                finished_samples = [sdid for sdid in finished_samples if sdid in existed_samples]
            if len(finished_samples) > 0:
                rest_sample_data = [sd for sd in self.sample_data if sd["ID"] not in finished_samples]
            else:
                rest_sample_data = self.sample_data

        ## Preprocessing of all data and generate sample_list data of all chains
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
                )
                self._preprocess_result_path.append(pti)

        # Parallel compute
        else:
            # Initialize errorlogs dir for err handling in parallel computing
            errorlog_path = result_directory + "Preprocessing/Step_results/Error_logs/"
            if os.path.exists(errorlog_path):
                shutil.rmtree(errorlog_path)
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
            if os.path.exists(errorlog_path):
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
                elif not os.path.exists(pti):
                    raise ValueError(f"\nGot invalid path: {pti}")
            # Update preprocess result file paths
            self._preprocess_result_path = preprocess_result_paths

        # Clear log after finishing whole preprocessing
        if os.path.exists(log_dir_path):
            shutil.rmtree(log_dir_path)

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
        if not os.path.exists(preprocess_result_dir):
            os.makedirs(preprocess_result_dir)

        # Validate preprocessing result file paths
        sd_paths = [
            f"{preprocess_result_dir}Step_results/PreprocessingResult_sample_{sd['label']}.dill"
            for sd in self.sample_data
        ]
        for sdp in sd_paths:
            if not os.path.exists(sdp):
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
                sdata = load_vars(spath)
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
            dump_vars(res_path_dill, {"chain_ind": pci, "chain_procs": pchain, "chain_res": pre_results})

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
        if os.path.exists(step_dir):
            for item in os.listdir(step_dir):
                if "PreprocessingChainResult_" in str(item):
                    item_path = os.path.join(step_dir, item)
                    try:
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                    except Exception as e:
                        print(f"Error in removing '{item_path}': \n{e}")
        else:
            warnings.warn(
                f"Step_results path is invalid:\n{result_directory}\nNo 'PreprocessingChainResult' is cleared.",
                UserWarning,
                stacklevel=3,
            )

    # Run modeling on single dataset
    # Sample_list item: (0 - Sample id, 1 - Original shape, 2 - Target value, 3 - Sample predictor value)
    @simple_type_validator
    def model_evaluation(  # noqa: C901
        self,
        n_processor: int = 1,
        resume: bool = False,
        report_directory: str = "",
        show_progress: bool = True,
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
        """  # noqa: E501

        # Prompt "if __name__ == '__main__':" protection for windows multiprocessing
        if n_processor > 1:
            if os.name == "nt":
                warnings.warn(
                    "Windows users must run multiprocessing within block \n\nif __name__ == '__main__': \n\n\
                    Please make sure all of your main codes in the script are placed within this block.",
                    UserWarning,
                    stacklevel=2,
                )

        # Close existing pyplot to save memory
        plt.close("all")

        # Validate report directory
        result_directory = report_directory
        if result_directory == "":
            result_directory = self._spec_exp.report_directory
        # Preprocessing dir
        preprocess_result_dir = result_directory + "Preprocessing/"
        if not os.path.exists(preprocess_result_dir):
            raise ValueError(f"\nPreprocessing result directory not found, got path:\n{preprocess_result_dir}")
        # Modeling dir
        model_result_dir = result_directory + "Modeling/"
        if not os.path.exists(model_result_dir):
            os.makedirs(model_result_dir)

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
            if not os.path.exists(cdp):
                raise ValueError(f"\nPreprocessing result file of chain {pchains[pci]} not found, path : \n{cdp}\n")
            cprocs = load_vars(cdp)["chain_procs"]
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
            if not os.path.exists(log_path):
                rest_cd_paths = cd_paths
            else:
                modeling_progress_log = load_vars(log_path)["modeling_progress_log"]
                rest_cd_paths = []
                for cdp in cd_paths:
                    cprocs = load_vars(cdp)["chain_procs"]
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
                    pc_it = load_vars(cdp)
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
                    if not os.path.exists(model_result_dir):
                        os.makedirs(model_result_dir)
                    errdir = model_result_dir + "Error_logs/"
                    if not os.path.exists(errdir):
                        os.makedirs(errdir)
                    cts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    pid = os.getpid()
                    error_log_path = errdir + f"error_{cts}_pid_{pid}.log"
                    err_msg = f"\nFailed in the modeling of preprocessing chain from path '{cdp}', \
                                error message: \n\n{str(e)}\n"
                    with open(error_log_path, "w") as f:
                        f.write(err_msg)
                    raise ValueError(e) from e

        # Multiprocessing modeling
        else:
            # Initialize errorlogs dir for err handling in parallel computing
            errorlog_path = model_result_dir + "Error_logs/"
            if os.path.exists(errorlog_path):
                shutil.rmtree(errorlog_path)
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
            if os.path.exists(errorlog_path):
                raise ValueError(
                    f"\nPreprocessing errors, please check error logs in the following path:\n{errorlog_path}"
                )

        # Clear log
        self._clear_model_log(log_path)

    @staticmethod
    def _clear_model_log(log_path: str) -> None:
        # Clear log when finished
        if os.path.exists(log_path):
            try:
                os.remove(log_path)
            except PermissionError as e:
                raise PermissionError(f"\nNo permission to clear existed running log : \n'{log_path}'.\n") from e
            except Exception as e:
                raise ValueError(f"\nError in clearing existed running log : \n{e}\n") from e

    @simple_type_validator
    def run(
        self,
        result_directory: str = "",
        n_processor: int = -1,
        pipe_parallel_for_modeling: bool = False,
        dump_backup: bool = False,
        step_result: bool = True,
        keep_chain_results: bool = True,
        resume: bool = False,
        sample_data_to_csv: bool = True,
        show_progress: bool = True,
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
            The default is -1, which automatically uses sequential processing for Windows and (maximum number of available CPUs - 1) processors for other OS.

            Windows Note: When using n_processor > 1 on Windows, all main codes in the working script must be placed within block "if __name__ == '__main__':".
            This requirement comes from 'pathos', which uses dill for object serialization and is essential for parallel execution of the package functions.

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
        """  # noqa: E501

        # Validate processor
        if n_processor < 0:
            if os.name == "nt":
                n_processor = 1
            else:
                n_processor = max(1, cpu_count() - 1)

        # Prompt "if __name__ == '__main__':" protection for windows multiprocessing
        if n_processor > 1:
            if os.name == "nt":
                warnings.warn(
                    "Windows users must run multiprocessing within block \n\nif __name__ == '__main__': \n\n\
                    Please make sure all of your main codes in the script are placed within this block.",
                    UserWarning,
                    stacklevel=2,
                )

        # Test process
        print("\n========= Test added chains =========\n")
        self.test_run()

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
            )
        else:
            print("\nNo model added, pipeline complete with preprocessing results.\n")
