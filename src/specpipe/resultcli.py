# -*- coding: utf-8 -*-
"""
File results reporters for console interface

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# OS
import os

# File
import dill

# Plot
import matplotlib.pyplot as plt

# Local
from .specio import simple_type_validator, unc_path

# CLI requirements
# ------
# Decoupled from SpecPipe for isolated access e.g. cross-console usage
# Does not cover all file results - exclusively fast access to core contents
# ------
# Core result of each chain to dill (.__specpipe_dill_data/.__specpipe_core_result_<file name>.dill)
# Groupstats result to dill (.__specpipe_dill_data/.__specpipe_result_summary_<file name>.dill)
# chain in id, chain in name, tables in df, plt object, model object
# all summary tables in df (No full summary files -> error: not complete pipeline)
# <name>__specpipe__.dill scaning and directly read back the corresponding data -> as a dict
# The dict: {summary table name: the table, ..., chain_order_as_name: chain result dict, ...}
# Chain result dict: {chain_id: chain_in_id, chain_name: chain_in_name, plt_name: plt_obj, table_name: table, ...}


# %% Reporter helpers


# Get file paths
@simple_type_validator
def _get_file_path(target_dir_path: str, match_string: str = "") -> list[str]:
    """Get file path within a target_dir_path matching filename and the 'match_string'."""
    file_paths = []
    for root, _, files in os.walk(unc_path(target_dir_path)):
        for file in files:
            if match_string in str(file):
                file_path = str(os.path.join(root, file))
                file_paths.append(unc_path(file_path))
    return file_paths


# Get dir paths
@simple_type_validator
def _get_dir_path(target_dir_path: str, match_string: str = "") -> list[str]:
    """Get directory path within a target_dir_path matching directory name and the 'match_string'."""
    dir_paths = []
    for root, dirs, _ in os.walk(unc_path(target_dir_path)):
        for directory in dirs:
            if match_string == str(directory):
                dir_path = str(os.path.join(root, directory))
                dir_paths.append(unc_path(dir_path))
    return dir_paths


# %% Reporters


# Reading group statistics
@simple_type_validator
def group_stats_report(specpipe_report_dir: str) -> dict:
    """Access major SpecPipe group statistics reports from SpecPipe report directory"""
    resulting_report: dict = {}
    data_paths = _get_file_path(specpipe_report_dir, ".__specpipe_result_summary_")
    for data_path in data_paths:
        data_name = str(os.path.basename(data_path)).replace(".__specpipe_result_summary_", "").replace(".dill", "")
        with open(unc_path(data_path), 'rb') as f:
            resulting_report[data_name] = dill.load(f)
    return resulting_report


# Reading core chain results
@simple_type_validator
def core_chain_report(specpipe_report_dir: str) -> list[dict]:
    """Access major SpecPipe core processing chain reports from SpecPipe report directory"""
    # Get chain report dir paths
    data_dir_paths = _get_dir_path(specpipe_report_dir, ".__specpipe_dill_data")

    # Filter chain dir paths
    chain_dir_paths = []
    for dir_path in data_dir_paths:
        if ('test_run' not in dir_path) and (
            ".__specpipe_core_result_Chain_process_info.dill" in os.listdir(unc_path(dir_path))
        ):
            chain_dir_paths.append(dir_path)

    # Turn off plot display in loading
    iplot: bool = False
    if plt.isinteractive():
        iplot = True
        plt.ioff()

    # Get chain results
    resulting_report: list = []
    for dir_path in chain_dir_paths:

        # Chain report file paths
        chain_report_paths = _get_file_path(dir_path, ".__specpipe_core_result_")

        # Get chain info
        chain_info_path = dir_path + "/.__specpipe_core_result_Chain_process_info.dill"
        with open(unc_path(chain_info_path), 'rb') as f:
            chain_info = dill.load(f)
            model_name = chain_info.iloc[-1, 1]
        chain_report: dict = {}
        chain_report["Chain_processes"] = chain_info

        # Get chain reports
        for data_path in chain_report_paths:
            data_name = str(os.path.basename(data_path)).replace(".__specpipe_core_result_", "").replace(".dill", "")
            # Remove model name
            data_name = data_name.replace("_" + model_name, "")
            if data_name != 'Chain_process_info':
                with open(unc_path(data_path), 'rb') as f:
                    chain_report[data_name] = dill.load(f)

        resulting_report.append(chain_report)

    # Turn back plot interactive display after loading
    if iplot:
        plt.ion()

    return resulting_report
