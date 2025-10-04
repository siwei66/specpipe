# -*- coding: utf-8 -*-
"""
Spectra data I/O-related utilities for SpecPipe

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Basics
import fnmatch
import time

# File operations
import glob
import inspect

# OS output
import io
import os
import re
import warnings
from contextlib import redirect_stdout
from datetime import datetime

# Typing
from functools import wraps
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

# Python variable r/w
import dill
import geopandas as gpd

# import zstandard as zstd
# Calculation
import numpy as np

# import numpy.typing as npt
import pandas as pd
import torch

# XML parsing
from bs4 import BeautifulSoup
from pydantic import AfterValidator, validate_call
from pyproj import CRS

# Geoshape
from shapely.geometry import MultiPolygon, Polygon

# %% simple_type_validator - Basic validator with serilization compatibility


def simple_type_validator(func: Callable) -> Callable:  # type: ignore[no-untyped-def]  # noqa: C901
    """
    Python function runtime native type validator for serilization of multiprocessing
    """

    @wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]  # noqa: C901
        hints = get_type_hints(func, include_extras=True)
        sig = inspect.signature(func)

        def check_type(  # type: ignore[no-untyped-def]  # noqa: C901
            value: Any, expected_type: Any
        ) -> tuple[bool, str]:
            # Error msg in check_type
            err_msg = ""

            # Early return for None values
            if value is None:
                # Check if None is allowed (Optional[T] or Union[T, None])
                origin = get_origin(expected_type)
                if origin is Union and type(None) in get_args(expected_type):
                    return True, err_msg
                return False, err_msg

            # Handle Any type - should always pass
            if expected_type is Any:
                return True, err_msg

            # Handle special typing constructs
            origin = get_origin(expected_type)

            # Handle simple types
            if origin is None:  # Simple type (int, str, etc.)
                # Fix for serialization error for int
                if isinstance(value, expected_type):
                    return True, err_msg
                elif type(value) is expected_type:
                    return True, err_msg
                else:
                    return False, err_msg

            # Union[T1, T2, ...] or Optional[T] (which is Union[T, None])
            if origin is Union:
                return any(check_type(value, t)[0] for t in get_args(expected_type)), err_msg

            # Callable[[args], return]
            if origin is Callable:
                return callable(value), err_msg

            # Handle containers (list, tuple, set, etc.)
            if origin in (list, tuple, set, frozenset):
                if not isinstance(value, origin):
                    return False, err_msg

                type_args = get_args(expected_type)
                if not type_args:  # Unparameterized (just `tuple`, `list`, etc.)
                    return True, err_msg

                # Handle tuples (fixed-length)
                if origin is tuple:
                    if len(type_args) > 1:
                        if Ellipsis in type_args:
                            if (len(type_args) == 2) & (type_args[0] is not Ellipsis):
                                return (
                                    all(check_type(x, t)[0] for x, t in zip(value, [type_args[0]] * len(value))),
                                    err_msg,
                                )
                            else:
                                raise ValueError(
                                    f"Invalid tuple annotation with Ellipsis: \
                                        expected exactly one type before '...', got: {type_args}"
                                )
                        if len(value) != len(type_args):
                            return False, err_msg
                        return all(check_type(x, t)[0] for x, t in zip(value, type_args)), err_msg
                    elif len(type_args) == 1:
                        return (
                            all(check_type(x, t)[0] for x, t in zip(value, [type_args[0]] * len(value))),
                            err_msg,
                        )
                    else:
                        return isinstance(value, expected_type), err_msg

                # Handle list, set, etc. (all elements must match the first type arg)
                return all(check_type(x, type_args[0])[0] for x in value), err_msg

            # Handle Annotated[T, ...]
            if origin is Annotated:
                base_type, *validators = get_args(expected_type)
                if not check_type(value, base_type)[0]:
                    return False, err_msg
                for validator in validators:
                    try:
                        validator(value)
                    except Exception as e:
                        err_msg = f"\n\nValidator error: \n{e}"
                        return False, err_msg
                return True, err_msg

            # Handle unrecognized types
            return isinstance(value, origin), err_msg

        # Validate all arguments
        bound_args = sig.bind(*args, **kwargs)
        for name, value in bound_args.arguments.items():
            if name in hints:
                is_valid, err_msg = check_type(value, hints[name])
                if not is_valid:
                    expected = hints[name]
                    raise TypeError(
                        f"Validation error for {name}\n\n "
                        f"Expected type: {expected}\n\n "
                        f"Got type: {type(value)}\n\n "
                        f"Got value: \n{repr(value)} "
                        f"{err_msg} "
                    )

        return func(*args, **kwargs)

    return wrapper


# %% Validator for numpy array-like


@simple_type_validator
def arraylike_validator(  # noqa: C901
    ndim: Optional[int] = None,
    shape: Optional[tuple[int, ...]] = None,
    as_type: Union[type, str, None] = None,
    d_type: Union[type, str, None] = None,
) -> Callable:
    """
    Pydantic validator for array-like.

    Parameters
    ----------
    ndim : int, optional
        ndim of the array. If not given, the criteria will not be applied.
        The default is None.

    shape : tuple[int,...], optional
        shape of the array. If not given, the criteria will not be applied. If both are given, both are applied.
        0 represents variable length, indicating the dimension can have any size.
        The default is None.

    dtype: type or str
        Validate simple datatypes of the values of the arraylike.

    as_type: type or str
        Convert simple datatypes of the values of the arraylike.
    """
    # Validate ndim
    if ndim is not None:
        if ndim < 0:
            raise ValueError(f"ndim cannot be negative, got: {ndim}")

    # Validate shape
    if shape is not None:
        for dimk in shape:
            if dimk < 0:
                raise ValueError(f"shape dimension cannot be negative. Got shape: {shape}\n")

    def arraylike_val(array_like_data: Any) -> np.ndarray:  # noqa: C901  # type: ignore[no-untyped-def]
        v = array_like_data

        # Validate conversion
        if isinstance(v, np.ndarray):
            arr = v
        elif isinstance(v, list) or isinstance(v, tuple) or isinstance(v, pd.DataFrame) or isinstance(v, pd.Series):
            try:
                arr = np.array(v)
            except Exception as e:
                raise ValueError(f"Given data '{v}' cannot be converted to numpy.ndarray\nGot error: {e}\n") from e
        elif isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
        else:
            raise TypeError(
                f"Given data \n{v}\n with data type \n'{type(v)}'\n cannot be \
                            converted to numpy.ndarray."
            )

        # Validate ndim
        if ndim is not None:
            if arr.ndim != ndim:
                raise ValueError(f"Given data has an incompatible ndim. Expected: {ndim}, Got: {arr.ndim}\n")

        # Validate shape
        if shape is not None:
            if len(arr.shape) != len(shape):
                raise ValueError(f"Given data has an incompatible ndim. Expected: {len(shape)}, Got: {arr.ndim}\n")
            for dimkt in enumerate(shape):
                if (dimkt[1] != 0) & (dimkt[1] != arr.shape[dimkt[0]]):
                    raise ValueError(f"Given data has an incompatible shape. Expected: {shape}, Got: {arr.shape}\n")

        # Convert dtype
        if as_type is not None:
            try:
                arr = arr.astype(as_type)
            except Exception as e:
                raise ValueError(f"Failed to convert array data type to float: \n{str(e)}\n") from e

        # Validate dtype
        if d_type is not None:
            dtype_err = False
            # Check np numeric dtypes
            if arr.dtype != np.dtype(d_type):
                # Check np str
                if "U" in str(np.dtype(d_type)):
                    np_str = str(np.dtype(d_type))
                    if np_str[-1] in [str(i) for i in range(10)]:
                        if np_str != "<U0":
                            dtype_err = True
                        elif "U" not in str(arr.dtype):
                            dtype_err = True
                    else:
                        if "U" not in str(arr.dtype):
                            dtype_err = True
                else:
                    dtype_err = True
            if dtype_err:
                raise TypeError(f"Expect array data type: {np.dtype(d_type)}, but got: {arr.dtype}")

        return arr

    return arraylike_val


# %% Validator for pandas dataframe

pd_dtypes = [
    # Numeric types
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    # Boolean
    "bool",
    # Text/String
    "object",
    "O",
    "string",
    # Datetime/temporal
    "datetime64[ns]",
    "datetime64[ns, tz]",
    "timedelta64[ns]",
    # Categorical
    "category",
    # Nullable types (Pandas 1.0+)
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Float32",
    "Float64",
    "boolean",
    # Sparse
    "Sparse[int8]",
    "Sparse[int16]",
    "Sparse[int32]",
    "Sparse[int64]",
    "Sparse[float16]",
    "Sparse[float32]",
    "Sparse[float64]",
]


@simple_type_validator
def _pd_dtype_cond(pd_dtype: Any, target_type: Union[type, str]) -> bool:  # type: ignore[no-untyped-def]
    """
    Validate if a pandas dtype is compatible with target type.
    """
    # Native dtypes
    if (type(target_type) is type) or str(target_type).lower() in [
        "bool",
        "int",
        "float",
        "str",
        "string",
    ]:
        if (target_type is bool) | (str(target_type).lower() == "bool"):
            cond = pd.api.types.is_bool_dtype(pd_dtype)
        elif (target_type is int) | (str(target_type).lower() == "int"):
            cond = pd.api.types.is_integer_dtype(pd_dtype)
        elif (target_type is float) | (str(target_type).lower() == "float"):
            cond = pd.api.types.is_float_dtype(pd_dtype)
        elif (target_type is str) | (str(target_type).lower() == "str") | (str(target_type).lower() == "string"):
            cond = pd.api.types.is_string_dtype(pd_dtype)
        else:
            raise ValueError("If target_type is specified in type, target_type must be bool, int, float or str")

    elif type(target_type) is str:
        # Numeric dtypes
        if target_type.lower() == "numeric":
            cond = pd.api.types.is_numeric_dtype(pd_dtype)
        elif target_type not in pd_dtypes:
            raise ValueError("target_type is not a valid data type of pandas dataframe")
        # Other pandas dtypes in str
        else:
            cond = str(pd_dtype) == target_type

    else:
        raise TypeError(f"Invalid type of target_type, got: {type(target_type)}, expected: type or str")

    return bool(cond)


@simple_type_validator
def dataframe_validator(  # noqa: C901
    colname_dtypes_dict: Optional[dict[str, Union[type, str]]] = None,
    shape: Optional[tuple[int, int]] = None,
    dtype: Optional[Union[type, str]] = None,
    ncol: Optional[int] = None,
    nrow: Optional[int] = None,
    index: Optional[list[Union[int, str]]] = None,
) -> Callable:
    """
    Pydantic validator for Pandas dataframe.

    Parameters
    ----------
    colname_dtypes_dict : dict[str,Union[type,str]], optional
        Dtypes of the columns. The default is None.

    shape : tuple[int,int], optional
        Shape of the dataframe. If not given, the criteria will not be applied. If both are given, both are applied.
        0 represents variable length, indicating the dimension can have any size.
        The default is None.

    dtype : type
        Global dtype of the dataframe. The default is None.

    ncol :
        Number of columns of the dataframe. The default is None.

    nrow :
        Number of rows of the dataframe. The default is None.

    index : list[Union[int,str]], optional
        Dataframe index. The default is None.
    """
    if shape is not None:
        for dimk in shape:
            if dimk <= 0:
                raise ValueError(f"shape dimension must be positive. Got shape: {shape}\n")

    if nrow is not None:
        if nrow <= 0:
            raise ValueError(f"nrow must be positive. Got: {nrow}\n")

    if ncol is not None:
        if ncol <= 0:
            raise ValueError(f"ncol must be positive. Got: {ncol}\n")

    @simple_type_validator
    def dataframe_val(dataframe: pd.DataFrame) -> pd.DataFrame:  # noqa: C901
        v = dataframe

        # Validate columns and dtypes
        if colname_dtypes_dict is not None:
            columns = list(colname_dtypes_dict.keys())
            dtype_values = list(colname_dtypes_dict.values())
            if dtype is not None:
                raise ValueError(
                    f"Redundant dtype specification.\
                        \nGlobal dtype : {dtype}\ncolname_dtypes_dict : {colname_dtypes_dict}"
                )
            if list(v.columns) != columns:
                raise ValueError(
                    f"Given dataframe has an incompatible column names.\
                        \nExpected: {columns}, \nGot: {v.columns}\n"
                )
            for coln, dtp, dte in zip(v.columns, v.dtypes, dtype_values):
                # Validate non-nested type
                nested = False
                tpcond = False
                if (type(dte) is type) | (type(dte) is str):
                    tpcond = _pd_dtype_cond(dtp, dte)

                # Validate nested type
                else:

                    @simple_type_validator
                    def nesttyp(v1: dte) -> None:  # type: ignore[valid-type]
                        pass

                    for vcoli in v[coln]:
                        try:
                            nesttyp(vcoli)
                            nested = True
                        except Exception as e:
                            raise ValueError(
                                f"Given dataframe has an incompatible nested data type:\
                                    \nColumn: {coln}\nValue: {vcoli}\n\nError:\n{e}\n"
                            ) from e

                if (not tpcond) and (not nested):
                    raise TypeError(
                        f"Given dataframe has an incompatible column data type:\nColumn: {coln},\
                            \nExpected dtype: {dte},\nGot dtype: {dtp}\n"
                    )

        elif dtype is not None:
            val_type = True

            for dtp in v.dtypes:
                val_type = val_type & _pd_dtype_cond(dtp, dtype)

            if not val_type:
                raise TypeError(
                    f"Given dataframe has incompatible data types:\nExpected: {dtype},\nGot dtype: {v.dtypes}\n"
                )

        # Validate shape
        if shape is not None:
            if v.shape != shape:
                raise ValueError(f"Given dataframe has an incompatible shape. Expected: {shape}, Got: {v.shape}\n")

        if nrow is not None:
            if v.shape[0] != nrow:
                raise ValueError(
                    f"Given dataframe has an incompatible number of rows. Expected: {nrow}, Got: {v.shape[0]}\n"
                )

        if ncol is not None:
            if v.shape[1] != ncol:
                raise ValueError(
                    f"Given dataframe has an incompatible number of columns. Expected: {ncol}, Got: {v.shape[1]}\n"
                )

        # Validate index
        if index is not None:
            if list(v.index) != list(index):
                raise TypeError(
                    f"\nGiven dataframe has an incompatible index:\n\n\
                        Expected index: {index},\n\n Got index: {list(v.index)}\n"
                )

        return v

    return dataframe_val


# %% Validator for dictionary value types


# Dictionary value type validator
@simple_type_validator
def dict_value_validator(  # noqa: C901
    value_type_list: Optional[list[Any]] = None,
    key_type_list: Optional[list[Any]] = None,
    value_type_dict: Optional[dict[Any, type]] = None,
) -> Callable:
    """
    Validator for dictionary value and key types.

    Parameters
    ----------
    value_type_list : list[type]
        Dictionary value types, the types is ordered.

    key_type_list : Optional[list[type]], optional
        Key types, the types is ordered. The default is None.

    value_type_dict : Optional[dict[Any, type]], optional
        Dictionary value types according to key, order is not required.
        Simultaneously specifying value_type_list and value_type_dict is not allowed.
        The default is None.
    """
    # Validate value type specification
    if value_type_list is not None and value_type_dict is not None:
        raise ValueError(
            "Simultaneously specifying value_type_list and value_type_dict is not allowed."
            f"\ngot value_type_list: {value_type_list}"
            f"\ngot value_type_dict: {value_type_dict}"
        )

    # Validate key type and value types
    if key_type_list is not None:
        if value_type_list is not None:
            if len(key_type_list) != len(value_type_list):
                raise ValueError(
                    "Inconsistent length of key_type_list and value_type_list"
                    f"\ngot key_type_list: {key_type_list}, length: {len(key_type_list)}"
                    f"\ngot value_type_list: {value_type_list}, length: {len(value_type_list)}"
                )

        if value_type_dict is not None:
            key_list = list(value_type_dict.keys())
            if len(key_type_list) != len(key_list):
                raise ValueError(
                    "Inconsistent length of key_type_list and value_type_list"
                    f"\ngot key_type_list: {key_type_list}, length: {len(key_type_list)}"
                    f"\ngot value_type_list: {value_type_dict}, length: {len(key_list)}"
                )

    # Validate types
    @simple_type_validator
    def dict_val(dictionary: dict) -> dict:  # noqa: C901
        dict_key_list = list(dictionary.keys())
        dict_value_list = list(dictionary.values())

        # Validate value types
        if value_type_list is not None:
            # Length validation
            if len(dict_key_list) != len(value_type_list):
                raise ValueError(
                    "Inconsistent length of given dictionary and value_type_list"
                    f"\ngot dictionary: {dictionary}, length: {len(dictionary)}"
                    f"\ngot value_type_list: {value_type_list}, length: {len(value_type_list)}"
                )

            for i in range(len(dict_key_list)):
                value = dict_value_list[i]
                value_type = value_type_list[i]

                @simple_type_validator
                def value_type_val(v: value_type) -> None:  # type: ignore[valid-type]
                    pass

                value_type_val(value)

        # Validate key types
        if key_type_list is not None:
            # Length validation
            if len(dict_key_list) != len(key_type_list):
                raise ValueError(
                    "Inconsistent length of given dictionary and key_type_list"
                    f"\ngot dictionary: {dictionary}, length: {len(dictionary)}"
                    f"\ngot key_type_list: {key_type_list}, length: {len(key_type_list)}"
                )

            for i in range(len(dict_key_list)):
                value = dict_key_list[i]
                value_type = key_type_list[i]

                @simple_type_validator
                def key_type_val(v: value_type) -> None:  # type: ignore[valid-type]
                    pass

                key_type_val(value)

        # Validate dict value types
        if value_type_dict is not None:
            # Length validation
            if set(dict_key_list) != set(value_type_dict.keys()):
                raise ValueError(
                    "Inconsistent keys of given dictionary and value_type_dict"
                    f"\ngot dictionary keys: {dictionary.keys()}"
                    f"\ngot value_type_dict keys: {value_type_dict.keys()}"
                )

            for key in list(dict_key_list):
                value = dictionary[key]
                value_type = value_type_dict[key]

                @simple_type_validator
                def type_val(v: value_type) -> None:  # type: ignore[valid-type]
                    pass

                type_val(value)

        return dictionary

    return dict_val


# %% Search file in dir


@validate_call
def search_file(  # noqa: C901
    directory_path: str, search_pattern: str, end_with: str = "", exclude_list: Optional[list[str]] = None
) -> list[str]:
    """
    Search and list files in a directory.

    Parameters
    ----------
    path : str
        Directory path to search.

    search_pattern : str
        pattern to search. Wildcard "*"/"[]"/"?"/"[!]" are supported.

    end_with : str
        Filter file names ending with this pattern, extension is included. Default is end_with name extension when exists.

    exclude_list : list[str]
        Filter file names that contains any strings in the exclude_list.

    Returns
    -------
    list[str]
        A list of file paths that match the search criteria.

    """  # noqa: E501
    # Init exclude_list
    if exclude_list is None:
        ex_list = []
    else:
        ex_list = exclude_list

    wildcardcheck = 0
    for wildcard in ["*", "?", "[", "]", "!"]:
        if wildcard in search_pattern:
            wildcardcheck = wildcardcheck + 1
    if wildcardcheck == 0:
        results = []
        for root, _, file_names in os.walk(directory_path):
            for file_name in file_names:
                if search_pattern is None or search_pattern in file_name:
                    results.append(os.path.join(root, file_name))
    else:
        search_path = os.path.join(directory_path, search_pattern)
        results = glob.glob(search_path)
        results = [str(res).replace("\\", "/") for res in results]

    # filter paths with unexpected ends
    if len(end_with) == 0:
        if "." in search_pattern:
            end_with = search_pattern.split(".")[-1]
    if "*" not in end_with:
        results_filtered = []
        for i in range(len(results)):
            result = results[i]
            if result[(len(result) - len(end_with)) : len(result)] == end_with:
                results_filtered.append(result)
        results = results_filtered
    elif end_with != "*":
        end_with_pattern = end_with.split("*")
        results_filtered = []
        for i in range(len(results)):
            result = results[i]
            result_ext = result.split(".")[-1]
            if len(end_with_pattern) == 2:
                pattern_ext1 = end_with_pattern[0]
                pattern_ext2 = end_with_pattern[1]
                if result_ext.startswith(pattern_ext1) & result_ext.endswith(pattern_ext2):
                    results_filtered.append(result)
            else:
                raise ValueError(
                    f"the number of '*' in file extension pattern cannot be larger than 1, "
                    f"but got pattern: {search_pattern}, "
                    f"with extension pattern: {end_with}"
                )
        results = results_filtered

    # filter paths excluding exclude_list items
    if len(ex_list) > 0:
        results_filtered2 = []
        for i in range(len(results)):
            resulti = results[i]
            result_name = resulti.split("/")[-1]
            iselect = True
            for eit in ex_list:
                if eit in result_name:
                    iselect = False
            if iselect:
                results_filtered2.append(resulti)
        results = results_filtered2

    # use same path format, replace \\ with /
    for ptid in range(len(results)):
        results[ptid] = str(results[ptid]).replace("\\", "/")

    return results


# %% filtering using wild cards


@overload
def names_filter(
    names: dict, pattern: str, dict_value_as_filename: bool = False, return_ids: bool = False
) -> tuple[dict[str, str], dict[str, str]]: ...


@overload
def names_filter(
    names: list[str],
    pattern: str,
    dict_value_as_filename: bool = False,
    return_ids: Literal[False] = False,
) -> tuple[list[str], list[str]]: ...


@overload
def names_filter(
    names: list[str],
    pattern: str,
    dict_value_as_filename: bool = False,
    return_ids: Literal[True] = True,
) -> tuple[list[int], list[int]]: ...


@validate_call
def names_filter(
    names: Union[list[str], dict],
    pattern: str,
    dict_value_as_filename: bool = False,
    return_ids: bool = False,
) -> Union[tuple[list[str], list[str]], tuple[list[int], list[int]], tuple[dict[str, str], dict[str, str]]]:
    """
    In a list or dictionary of names, filter names by a pattern.
    Returns matched names and unmatched names.

    Parameters
    ----------
    names : list[str] or dict[str,str]
        File names to filter.

    pattern : str
        Pattern to search. Unix-like patterns is supported (using fnmatch).

    dict_value_as_filename : bool
        For dictionary of names, set True if dictionary values instead of keys are the file names to filter.

    return_ids : bool
        For list of names, if True, return name indices of filtered results instead of result values. The default is False.

    Returns
    -------
    Union[tuple[list[str],list[str]], tuple[list[int],list[int]], tuple[dict[str,str],dict[str,str]]]
        Lists or dictionaries of matched name items and unmatched name items.

    Raises
    ------
    ValueError
        If names is not recognized as list or dictionary.

    """  # noqa: E501

    if type(names) is list:
        # For names
        selected_list = [fn for fnid, fn in enumerate(names) if fnmatch.fnmatch(fn, pattern)]
        removed_list = [fn for fnid, fn in enumerate(names) if not fnmatch.fnmatch(fn, pattern)]

        # For IDs
        selected_ids = [fnid for fnid, fn in enumerate(names) if fnmatch.fnmatch(fn, pattern)]
        removed_ids = [fnid for fnid, fn in enumerate(names) if not fnmatch.fnmatch(fn, pattern)]

        # Return
        if not return_ids:
            return selected_list, removed_list
        else:
            return selected_ids, removed_ids

    elif type(names) is dict:
        if dict_value_as_filename:
            fns_to_remove_ids: list = [fnid for fnid in names.keys() if fnmatch.fnmatch(names[fnid], pattern)]
        else:
            fns_to_remove_ids = [fn for fn in names.keys() if fnmatch.fnmatch(fn, pattern)]
        selected_dict: dict = {}
        removed_dict: dict = names
        for fnid in fns_to_remove_ids:
            selected_dict[fnid] = names[fnid]
            del removed_dict[fnid]
        return selected_dict, removed_dict

    else:
        raise ValueError("\n provided names is not recognized as a list or dictionary.")


# %% Extract ENVI ROI coordinates


# Read multipolygon vertex coordinate pairs from ENVI ROI files
@validate_call
def envi_roi_coords(roi_xml_path: str) -> list[dict[str, Any]]:
    """

    Get vertex coordinates of Polygon ROIs from ENVI ROI xml file.

    Parameters
    ----------
    rpath : str
        Path of ENVI xml ROI file.

    Returns
    -------
    list[dict[str, Any]]
        A list with dictionaries of ROI name, ROI geometry type and lists of vertex coordinate pairs in tuples.

    Raises
    ------
    ValueError
        If no polygon for a ROI is found.

    """
    # Read ENVI ROI xml file
    with open(roi_xml_path, "r") as f:
        roi_data = f.read()
    soup = BeautifulSoup(roi_data, "xml")
    sroi = soup.find_all("Region")
    rn = len(sroi)
    if rn < 1:
        raise ValueError(f"No ROI is found in the provided xml file, got file content: \n\n{soup.prettify()}")
    # Parsing
    roi_list = []
    for n in range(rn):
        roii = sroi[n]
        roiname = roii.get("name")
        roipolys = roii.find_all("Polygon")
        if len(roipolys) == 1:
            polytype = "Polygon"
        elif len(roipolys) > 1:
            polytype = "MultiPolygon"
        else:
            raise ValueError(f"no polygon for ROI {roiname} is found")
        poly_coords = []
        for polygs in roipolys:
            roicoord_found = polygs.find("Coordinates")
            if roicoord_found is not None:
                roicoord = roicoord_found.text.strip()
            else:
                raise ValueError("No 'Coordinates' found in given xml file.")
            coord = re.findall(r"-?\d+\.?\d*[eE][+-]?\d+|-?\d+\.?\d*", roicoord)
            coordpairs = []
            for i in range(int(len(coord) / 2)):
                coordpairs.append(
                    (
                        float(coord[2 * i].replace("e", "E")),
                        float(coord[2 * i + 1].replace("e", "E")),
                    )
                )
            poly_coords.append(coordpairs)
        roi_list.append({"name": roiname, "type": polytype, "coordinates": poly_coords})
    return roi_list


# %% Extract shp ROI Coordinates


# Read QGIS shp polygon / multipolygon ROI Coordinate Pairs
@validate_call
def shp_roi_coords(roi_shp_path: str) -> list[dict[str, Any]]:
    """

    Get vertex coordinates of Polygon and MultiPolygon ROIs from shapefile.

    Parameters:
    -----------
    shapefile_path : str
        Path to the input shapefile

    Returns:
    --------
    list[dict[str, Any]]
        A list with dictionaries of ROI name, ROI geometry type and lists of vertex coordinate pairs in tuples.

    Raises
    ------
    ValueError
        If geom_type is not supported.

    """
    # Read ROI shapefile as geodataframe
    gdf = gpd.read_file(roi_shp_path)
    # Parsing
    roi_list = []
    for id1, rdata in gdf.iterrows():
        geometry = rdata.geometry
        name = rdata.get("name", f"ROI_{id1}")
        if geometry.geom_type == "Polygon":
            coordinates = list(geometry.exterior.coords)
            roi_list.append({"name": name, "type": "Polygon", "coordinates": [coordinates]})
        elif geometry.geom_type == "MultiPolygon":
            multi_coords = []
            for poly in geometry.geoms:
                poly_coords = list(poly.exterior.coords)
                multi_coords.append(poly_coords)
            roi_list.append({"name": name, "type": "MultiPolygon", "coordinates": multi_coords})
        else:
            raise ValueError(f"geom_type {geometry.geom_type} is not supported")
    return roi_list


# %% Write variables to dill file and load variables from dill file


# Write variables to dill file
@validate_call
def dump_vars(target_file_path: str, var_dict: dict[str, Any], backup: bool = True) -> None:
    """
    Dump variables to dill file.

    Parameters
    ----------
    file_path : str
        Full path for the target output file.

    var_dict : dict[str, Any]
        Dictionary of the variables to save.
        Set keys as the variable names and value as the corresponding variable values.

    Returns
    -------
    None.

    """
    # Validate extension
    target_file_path1 = os.path.splitext(target_file_path)[0] + ".dill"

    # Dill dump
    temp_path = target_file_path1 + ".tmp"
    with open(temp_path, "wb") as f:
        dill.dump(var_dict, f)
    os.replace(temp_path, target_file_path1)

    # Dump backup file
    if backup:
        # Path for backup
        cts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        target_file_path_b = os.path.splitext(target_file_path1)[0] + "_" + cts + ".dill"
        # Dill dump
        with open(target_file_path_b, "wb") as f:
            dill.dump(var_dict, f)


# Load variables from dill file
@validate_call
def load_vars(source_file_path: str) -> dict[str, Any]:
    """
    Load variables from dill file.

    Parameters
    ----------
    file_path : str
        path of dill file to load.

    Raises
    ------
    ValueError
        If specified source file is not a 'dill' file.

    ValueError
        If source_file_path does not exist.

    Returns
    -------
    data : dict
        Stored python variables.

    """
    # Validate extension
    if os.path.splitext(source_file_path)[1] != ".dill":
        raise ValueError(f"The specified source file must be a 'dill' file, got source_file_path: \n{source_file_path}")

    # Validate existence
    if not os.path.exists(source_file_path):
        raise ValueError(f"source_file_path does not exist: {source_file_path}")

    with open(source_file_path, "rb") as f:
        data: dict[str, Any] = dill.load(f)

    return data


# %% Write to large csv


# Write pandas dataframe to csv with auto compression
@validate_call
def df_to_csv(  # type: ignore[no-untyped-def]
    dataframe: Annotated[Any, AfterValidator(dataframe_validator())],
    path: str,
    index: bool = False,
    return_path: bool = True,
    overwrite: bool = True,
    compress_nvalue_threshold: int = 1000000,
    compress_shape_threshold: tuple[int, int] = (1048576 - 1, 16384 - 1),
    compression_format: str = "zstd",
    **kwargs,
) -> Optional[str]:
    """
    Write large Pandas dataframe to CSV file using automatical compression and return write path.
    """
    # Compresion formats {parameter : ext}
    ext = {"gzip": ".gz", "bz2": ".bz2", "zip": ".zip", "xz": ".xz", "zstd": ".zst", "infer": ""}

    # Validate compression configs
    if compression_format.lower() not in list(ext.keys()):
        raise ValueError(
            f"Compression_format must be one of 'gzip', 'bz2', 'zip', 'xz' and 'zstd', got : {compression_format}"
        )

    # Validate compression
    nvalues = 1
    for d in dataframe.shape:
        nvalues = nvalues * d

    # For small table, compression is not applied by default
    if (
        (nvalues <= compress_nvalue_threshold)
        and (dataframe.shape[0] < compress_shape_threshold[0])
        and (dataframe.shape[1] < compress_shape_threshold[1])
    ):
        compression_format = "infer"

    # Validate path
    if (path.split(".")[-1]).lower() != "csv":
        if not (
            ((path.split(".")[-1]).lower() == ext[compression_format][1:]) and ((path.split(".")[-2]).lower() == "csv")
        ):
            raise ValueError(f"Invalid csv file path, got {path}, file extension must be '.csv'")
    if ((path.split(".")[-1]).lower() == "csv") and (compression_format != "infer"):
        path = path + ext[compression_format]
    if not overwrite:
        if os.path.exists(path):
            raise ValueError(f"File path '{path}' already exists while overwrite is set {overwrite}")

    # Validate other parameters
    # Get accepted parameters
    sig = inspect.signature(dataframe.to_csv)
    accepted_params = sig.parameters.keys()
    # Filter kwargs and only allow accepted parameters
    filtered_params = {k: v for k, v in kwargs.items() if k in accepted_params}
    filtered_params["path_or_buf"] = path
    filtered_params["compression"] = compression_format
    filtered_params["index"] = index

    # Write CSV
    dataframe.to_csv(**filtered_params)
    if return_path:
        return path
    else:
        return None


# Read compressed csv
@validate_call
def df_from_csv(path: str, **kwargs) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    """
    Automatically read large CSV file with compression to Pandas dataframe.
    """
    # Compresion formats {parameter : ext}
    extr = {"gz": "gzip", "bz2": "bz2", "zip": "zip", "xz": "xz", "zst": "zstd"}

    # Validate path
    if not os.path.exists(path):
        raise ValueError(f"File path '{path}' is invalid.")
    if ((path.split(".")[-1]).lower() != "csv") and (
        not (((path.split(".")[-2]).lower() == "csv") and ((path.split(".")[-1]).lower() in list(extr.keys())))
    ):
        raise ValueError(
            f"Invalid CSV file '{path}', \
                the file extension must be one of '.csv', '.csv.gz', '.csv.bz2', '.csv.zip', '.csv.xz' and '.csv.zst'"
        )

    # Parse compression
    if (path.split(".")[-1]).lower() != "csv":
        compression_format = extr[(path.split(".")[-1]).lower()]
    else:
        compression_format = "infer"

    # Validate other parameters
    # Get accepted parameters
    sig = inspect.signature(pd.read_csv)
    accepted_params = sig.parameters.keys()
    # Filter kwargs and only allow accepted parameters
    filtered_params = {k: v for k, v in kwargs.items() if k in accepted_params}
    filtered_params["filepath_or_buffer"] = path
    filtered_params["compression"] = compression_format

    # Read CSV
    return pd.read_csv(**filtered_params)


# %% Write ROI coord lists to ROI files


# ROI coords to ENVI ROI xml
@simple_type_validator
def roi_to_envi_xml(  # noqa: C901
    file_path: str,
    name: str = "",
    coordinates: Optional[list[list[tuple[Union[int, float], Union[int, float]]]]] = None,
    crs: Union[str, CRS] = "none",
    color: Optional[tuple[int, int, int]] = None,
    roi_type: str = "polygon",
    roi_list: Optional[list[dict[str, Any]]] = None,
    return_path: bool = True,
) -> Optional[str]:
    """
    Write the vertex coordinate pairs of (multipart-)polygon Region of Interest (ROI) to ENVI xml ROI file.

    Parameters
    ----------
    file_path : str
        File path.

    name : str
        ROI name.

    coordinates : list[list[tuple[float,float]]]
        List of vertex coordinate pair list.

    crs : Union[str, CRS]
        Coordinate system of ROI. The default is 'none'.

    color : Optional[tuple[int, int, int]], optional
        List of color tuples of the ROIs in RGB values.
        List of RGB color tuples for the ROIs.
        Each color is specified as a tuple of three integers (R, G, B), where each value must be in the range 0 to 255.
        The default value applies random colors.

    roi_type : str, optional
        Type of ROI to write.
        The default is 'polygon'.
        Only 'polygon' type is supported in the current version, and this parameter has no effect.
        The polygon geometry type includes polygons, rectangles and ellipses in ENVI, as they are all represented as polygons.

    roi_list : Optional[list[dict[str, Any]]]
        If multiple ROIs are to be written, provide the ROI parameters as a list of dictionaries. This will replace the individual arguments above.
        The expected format for the list is:
            [{'name' : name, 'crs': crs, 'color': color, 'type': roi_type, 'coordinates': coordinates}, ...]
        The expected type of coordinates is:
            list[list[tuple[Union[int, float], Union[int, float]]]]
            (list of lists of polygon coordinate pairs in tuple)

    Returns
    -------
    str
        Path of generated ENVI xml ROI file.
    """  # noqa: E501

    # Initialize coordinate list
    if coordinates is None:
        coord_list = []
    else:
        coord_list = coordinates

    # Validate ROI item dict in roi_list for ENVI xml
    if roi_list is not None:
        roi_it_validator = dict_value_validator(
            [
                str,
                str,
                Optional[tuple[int, int, int]],
                str,
                list[list[tuple[Union[int, float], Union[int, float]]]],
            ]
        )
        for roi_item in roi_list:
            _ = roi_it_validator(roi_item)

    # Validate path
    file_path = str(file_path).replace("\\", "/")
    path_dir = os.path.dirname(file_path)
    if not os.path.exists(path_dir):
        warnings.warn(
            f"The specified path directory does not exist, the directory is created: {path_dir}",
            UserWarning,
            stacklevel=2,
        )
        os.makedirs(path_dir)
    file_path = os.path.splitext(file_path)[0] + ".xml"

    # ROI type current version fixed to 'polygon'
    roi_type = "polygon"

    # Validate ROI parameters
    if roi_list is None:
        if name == "":
            raise ValueError("ROI name is not specified.")
        if len(coord_list) == 0:
            raise ValueError("ROI polygon vertex coordinates is not provided.")
        roi_list = [{"name": name, "crs": crs, "color": color, "type": roi_type, "coordinates": coord_list}]

    # Validate ROI name, color and coordinates
    roi_names = []
    for roi in roi_list:
        # Validate ROI name
        roin = roi["name"]
        if roin == "":
            raise ValueError("ROI name is not specified.")
        if roin in roi_names:
            raise ValueError(f"ROI name must be unique, got duplicated name: {roin}.")
        roi_names.append(roin)
        # Validate color
        if roi["color"] is not None:

            @simple_type_validator
            def val_color(v: tuple[int, int, int]) -> tuple[int, int, int]:
                return v

            roi["color"] = val_color(roi["color"])
            for cv in roi["color"]:
                if cv < 0 or cv > 255:
                    raise ValueError(f"RGB values must be in the range of 0 to 255, got: {roi['color']}")
        else:
            roi["color"] = tuple(np.random.randint(0, 256, 3))
        # Validate coordiantes
        for poly in roi["coordinates"]:
            if poly[0] != poly[-1]:
                poly.append(poly[0])
            if len(poly) < 4:
                raise ValueError(f"At least 3 vertices must be defined for a polygon geometry, but got: {poly}")

    # Write ROI xml file
    with open(file_path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>' + "\n")
        f.write('<RegionsOfInterest version="1.1">' + "\n")
        for roi in roi_list:
            roin = roi["name"]
            roic = roi["color"]
            roic_str = f"{roic[0]},{roic[1]},{roic[2]}"
            coord_sys = roi["crs"]
            f.write(f'  <Region name="{roin}" color="{roic_str}">' + "\n")
            f.write("    <GeometryDef>" + "\n")
            f.write(f"      <CoordSysStr>{coord_sys}</CoordSysStr>" + "\n")
            for poly in roi["coordinates"]:
                f.write("      <Polygon>" + "\n")
                f.write("        <Exterior>" + "\n")
                f.write("          <LinearRing>" + "\n")
                f.write("            <Coordinates>" + "\n")
                coord_str = ""
                for i, coord_pair in enumerate(poly):
                    coord_str = coord_str + str(coord_pair[0]) + " " + str(coord_pair[1])
                    if i < len(poly) - 1:
                        coord_str = coord_str + " "
                f.write(coord_str + "\n")
                f.write("            </Coordinates>" + "\n")
                f.write("          </LinearRing>" + "\n")
                f.write("        </Exterior>" + "\n")
                f.write("      </Polygon>" + "\n")
            f.write("    </GeometryDef>" + "\n")
            f.write("  </Region>" + "\n")
        f.write("</RegionsOfInterest>")

    if return_path:
        return file_path
    else:
        return None


# ROI coords to shp
@simple_type_validator
def roi_to_shp(  # noqa: C901
    file_path: str,
    crs: Union[str, CRS],
    name: str = "",
    coordinates: Optional[list[list[tuple[Union[int, float], Union[int, float]]]]] = None,
    roi_type: str = "polygon",
    roi_list: Optional[list[dict[str, Any]]] = None,
    return_path: bool = True,
) -> Optional[str]:
    """
    Write the vertex coordinate pairs of (multipart-)polygon Region of Interest (ROI) to a shapefile.

    Parameters
    ----------
    file_path : str
        File path.

    crs : Union[str, CRS]
        Coordinate system (CRS) of the ROIs.
        Note: CRS must be provided. Non-georeferenced image may require additional alignment.

    name : str
        ROI name.

    coordinates : list[list[tuple[float,float]]]
        List of vertex coordinate pair list.

    roi_type : str, optional
        Type of ROI to write.
        The default is 'polygon'.
        Only 'polygon' type is supported in the current version, and this parameter has no effect.

    roi_list : Optional[list[dict[str, Any]]]
        If multiple ROIs are to be written, provide the ROI parameters as a list of dictionaries. This will replace the individual arguments above.
        The expected format for the list is:
            [{'name' : name, 'type': roi_type, 'coordinates': coordinates}, ...]
        The expected type of coordinate list is:
            list[list[tuple[Union[int, float], Union[int, float]]]]
            (list of lists of polygon coordinate pairs in tuple)

    Returns
    -------
    str
        Path of generated ROI shapefile.
    """  # noqa: E501

    # Initialize coordinate list
    if coordinates is None:
        coord_list = []
    else:
        coord_list = coordinates

    # Validate ROI item dict in roi_list for shapefile
    if roi_list is not None:
        roi_it_validator = dict_value_validator([str, str, list[list[tuple[Union[int, float], Union[int, float]]]]])
        for roi_item in roi_list:
            _ = roi_it_validator(roi_item)

    # ROI type (Only polygon for current version)
    roi_type = "polygon"

    # Validate path
    file_path = str(file_path).replace("\\", "/")
    path_dir = os.path.dirname(file_path)
    if not os.path.exists(path_dir):
        warnings.warn(
            f"The specified path directory does not exist, the directory is created: {path_dir}",
            UserWarning,
            stacklevel=2,
        )
        os.makedirs(path_dir)
    file_path = os.path.splitext(file_path)[0] + ".shp"

    # Validate ROI parameters
    if roi_list is None:
        if name == "":
            raise ValueError("ROI name is not specified.")
        if len(coord_list) == 0:
            raise ValueError("ROI polygon vertex coordinates is not provided.")
        roi_list = [{"name": name, "type": roi_type, "coordinates": coord_list}]

    # Validate ROI name and coordinates
    roi_names = []
    for roi in roi_list:
        # Validate ROI name
        roin = roi["name"]
        if roin == "":
            raise ValueError("ROI name is not specified.")
        if roin in roi_names:
            raise ValueError(f"ROI name must be unique, got duplicated name: {roin}.")
        roi_names.append(roin)
        # Validate coordiantes
        for poly in roi["coordinates"]:
            if poly[0] != poly[-1]:
                poly.append(poly[0])
            if len(poly) < 4:
                raise ValueError(f"At least 3 vertices must be defined for a polygon geometry, but got: {poly}")

    # Write shp
    geometries = []
    attributes: dict = {"name": [], "type": []}

    for roi in roi_list:
        roin = roi["name"]
        geom_type = roi["type"]
        coords = roi["coordinates"]
        if geom_type == "polygon":
            polygons = [Polygon(poly) for poly in coords]
            geometries.append(MultiPolygon(polygons))
            attributes["name"].append(roin)
            attributes["type"].append(geom_type)
        else:
            raise ValueError(f"geom_type must be 'polygon', but got: '{geom_type}'")

    # Create gdf
    gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs=crs)

    # Save to shapefile
    gdf.to_file(file_path, driver="ESRI Shapefile")

    if return_path:
        return file_path
    else:
        return None


# %% Silent decorator to mute print and progress bar of functions


def silent(func: Callable) -> Callable:
    """
    Decorator to suppress print
    """

    @wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        # Disable tqdm
        try:
            from tqdm import tqdm

            original_tqdm_init = tqdm.__init__

            def silent_tqdm_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                kwargs["disable"] = True
                return original_tqdm_init(self, *args, **kwargs)

            tqdm.__init__ = silent_tqdm_init
        except ImportError:
            pass
        # Redirect output
        with redirect_stdout(io.StringIO()):
            result = func(*args, **kwargs)
        # Reset tqdm
        tqdm.__init__ = original_tqdm_init
        return result

    return wrapper


# %% Robust listdir


def lsdir_robust(
    path: str, fetch_number_gt: int = 0, *, retry: int = 5, time_wait_min: float = 0.5, time_wait_max: float = 20
) -> list:
    """
    Substitution of listdir with retry for file-related testing using GitHub workflow actions.
    """
    # Validate configs
    retry = max(int(retry), 1)
    time_wait_min = max(time_wait_min, 0.1)
    time_wait_max = max(time_wait_min, 0.2)
    fetch_number_gt = max(int(fetch_number_gt), 0)

    # Fetch loop
    for run_i in range(retry):
        # OS listdir method
        try:
            result = os.listdir(path)
            if result is not None:
                result1: list = list(result)
                if len(result1) > fetch_number_gt:
                    return result1

        except OSError:
            pass

        # Glob method
        try:
            pattern_path = os.path.join(path, "*")
            result = glob.glob(pattern_path)
            if result is not None:
                result1 = list(result)
                if len(result1) > fetch_number_gt:
                    return result1

        except OSError:
            pass

        if run_i < (retry - 1):
            # Wait time
            time_wait_coef: float = (time_wait_max / time_wait_min) ** (1 / max(1, retry - 1))
            wait_time: float = time_wait_min * (time_wait_coef**run_i)
            time.sleep(wait_time)

    # Not fetch required number of result
    if result is not None:
        return list(result)
    else:
        return []
