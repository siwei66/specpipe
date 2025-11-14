# -*- coding: utf-8 -*-
"""
File results reporters for console interface

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# OS
import os

# Data
import numpy as np
import pandas as pd

# CLI requirements

# Decoupled from SpecPipe for isolated access e.g. cross-console usage
# Does not cover all file results - exclusively fast access to core contents

# Core result of each chain to dill (.__specpipe_dill_data/.__specpipe_core_result_<file name>.dill)
# Groupstats result to dill (.__specpipe_dill_data/.__specpipe_result_summary_<file name>.dill)
# chain in id, chain in name, tables in df, plt object, model object
# all summary tables in df (No full summary files -> error: not complete pipeline)

# <name>__specpipe__.dill scaning and directly read back the corresponding data -> as a dict
# The dict: {summary table name: the table, ..., chain_order_as_name: chain result dict, ...}
# Chain result dict: {chain_id: chain_in_id, chain_name: chain_in_name, plt_name: plt_obj, table_name: table, ...}


#%% Results 