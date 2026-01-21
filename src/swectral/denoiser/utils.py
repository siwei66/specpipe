# -*- coding: utf-8 -*-
"""
Kernel smoothing helper - RollWindow - (chain) apply functions to rolling window

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Sequential data denoising
import inspect
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Annotated, Any, Callable, Union, Optional
import warnings

from ..specio import simple_type_validator, arraylike_validator

# # For local test
# from specpipe.specio import simple_type_validator, arraylike_validator


# %% Helper: data standardizers - 1D data to 2D and 2D back to 1D


def _to_2d_arr(
    data_array: Union[Annotated[Any, arraylike_validator(ndim=2)], Annotated[Any, arraylike_validator(ndim=1)]],
) -> np.ndarray:
    """Standardize 1D and 2D arraylike to 2D array data"""
    data_array = np.asarray(data_array)
    if data_array.ndim == 1:
        return np.asarray(data_array.reshape(1, -1))
    elif data_array.ndim == 2:
        return np.asarray(data_array)
    else:
        raise ValueError(f"Expected 1D or 2D array, but got shape: {data_array.shape}")


def _back_1d_arr(data_array: Union[Annotated[Any, arraylike_validator(ndim=2)]]) -> np.ndarray:
    """Standardize 1D and 2D arraylike to 2D array data"""
    data_array = np.asarray(data_array)
    if data_array.ndim == 2:
        if data_array.shape[0] == 1 or data_array.shape[1] == 1:
            return np.asarray(data_array.reshape(-1))
        else:
            return np.asarray(data_array)
    else:
        raise ValueError(f"Expected 2D array, but got shape: {data_array.shape}")


# %% Common denoising techniques - kernel smoothing methods - helper: rolling window application of functions


# Rolling window for rolling computation along a table axis
class RollWindow:
    """
    Apply a function to rolling window of a 2d array-like along axis 0 or 1.

    Attributes
    ----------
    window_size : int
        Size of rolling window, must be at least 2.

    axis : int, optional
        Window is rolling along the assigned axis. The default is 1.

    roll_mode : str, optional
        Rolling mode. Available options:

        - "center" - window is created centered on current data point, better applicable to static data series.
        - "end" - for time series, window is created end with current data point, better applicable to time series.
        - "knn" - k-nearst neighbor mode, k is the window size.

        For "knn", the window is constructed with available data centered on current data point, except the edge windows.
        The edge windows are constructed on the k nearst neighbor values.

        The default is "center".

    padding : str, optional
        Padding approach for rolling mode ``"center"`` and ``"end"``. Available options:

        - "none" - No padding applied. For rolling mode "center" and "end", the resulting data series will be window_size - 1 smaller than the original data series without padding.
        - "nan" - Missing values are set to numpy nan after applying the function.
        - "ext_edge" - Edge value of the first and last window are applied to pad before applying the smoothing function.
        - "constant_edge" - Missing values are filled with edge value after applying function.
        - "extrapolation" - Missing values are linearly extrapolated from edge values with length of window_size/2 (center mode), window_size-1 (end mode) after function application.

        The default is "none".

    numtype : str, optional
        Number type of given data, supports number type of Numpy. Default is ``"float32"``.

    axis_double_definition_warning : bool, optional
        If True, the duplicate definition warning will prompt when function has ``axis`` argument.

        The default is True. Set false for known application.

    Methods
    -------
    _apply
        Apply function with arguments and keyword arguments to the rolling window.
    _chain_apply
        Apply a list of functions to the rolling window with a list of arguments and keyword arguments matched by list index.
    """  # noqa: E501

    @simple_type_validator
    def __init__(
        self,
        window_size: int,
        axis: int = 1,
        roll_mode: str = 'center',
        padding: str = 'none',
        numtype: Union[str, type] = 'float32',
        axis_double_definition_warning: bool = True,
    ) -> None:
        # Private internal attributes
        self.__window_ind: int = 0
        self.__data_array_shape: tuple[int, ...] = (0, 0)

        # Public attributes
        # Validate window_size
        self.numtype: Union[str, type] = numtype
        if window_size < 2:
            raise ValueError('window_size must be at least 2!')
        self.window_size: int = window_size
        # Validate axis
        if axis not in [0, 1]:
            raise ValueError(f"axis must be 0 or 1, got: {axis}")
        self.axis: int = axis

        # Validate roll_mode
        roll_mode = str(roll_mode).lower()
        if roll_mode not in ['center', 'end', 'knn']:
            raise ValueError(f"'roll_mode' must be one of 'center', 'end' and 'knn', got: {roll_mode}")
        self.roll_mode: str = roll_mode

        # Validate padding approach
        padding = str(padding).lower()
        if padding not in ['none', 'nan', 'ext_edge', 'constant_edge', 'extrapolation']:
            raise ValueError(
                f"'padding' must be one of 'none', 'nan', 'ext_edge', 'constant_edge' and 'extrapolation', \
                    got: {roll_mode}"
            )
        self.padding: str = padding

        self.axis_double_definition_warning: bool = axis_double_definition_warning

    @property
    def _window_ind(self) -> int:
        return self.__window_ind

    @_window_ind.setter
    def _window_ind(self, value: int) -> None:
        raise ValueError("window_ind cannot be modified.")

    @property
    def _data_array_shape(self) -> tuple[int, ...]:
        return self.__data_array_shape

    @_data_array_shape.setter
    def _data_array_shape(self, value: tuple[int, int]) -> None:
        raise ValueError("data_array_shape cannot be modified.")

    # Apply function to rolling window
    @simple_type_validator
    def _apply(  # noqa: C901
        self,
        data_array: Annotated[Any, arraylike_validator(ndim=2)],
        func: Callable,
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Apply a function to rolling window of a 2d array or dataframe along axis 0 or 1.

        Parameters
        ----------
        data_array : 2D array-like
            Data for processing, data series as row or column.
        func : callable
            Applied function for the window.
        args : list, optional
            A list of positional parameters for the applied function in addition to your data.
            The data must be the first parameter.
            The default is [].
        kwargs : dict, optional
            A dict of key word parameters for the applied function.

        Returns
        -------
        np.ndarray
            Array of rolling calculation result.

        Examples
        --------
        Prepare an exemplary 2D data array::

            >>> arr = np.array([[1, 2, 3, 4, 5, 6], [2, 2, 4, 4, 6, 6]])

        Define a function to apply::

            >>> def func_to_apply(arr):
            ...     return np.mean(arr, axis=0, keepdims=True)

        Apply the defined function::

            >>> RollWindow(3)._apply(arr, func_to_apply)

        Use different window size::

            >>> RollWindow(4)._apply(arr, func_to_apply)

        Pad with ``numpy.nan``::

            >>> RollWindow(3, padding='nan')._apply(arr, func_to_apply)

        Specify rolling mode::

            >>> RollWindow(3, roll_mode='knn')._apply(arr, func_to_apply)

        Compute along a different axis::

            >>> RollWindow(3, axis=0)._apply(arr.T, func_to_apply)
        """
        if args is None:
            arg_values: list = []
        if kwargs is None:
            kwarg_values: dict = {}
        result = self._chain_apply(data_array, [func], [arg_values], [kwarg_values])
        assert isinstance(result, np.ndarray)
        return result

    # Apply helper: _padder
    @staticmethod
    @simple_type_validator
    def _padder(
        result: np.ndarray, window_size: int, arr: np.ndarray, padding: str, pdist0: int, pdist1: int
    ) -> np.ndarray:
        if (padding == 'none') | (padding == 'ext_edge'):
            pass
        elif padding == 'nan':
            result = np.pad(
                array=np.array(result), pad_width=((pdist0, pdist1), (0, 0)), mode='constant', constant_values=np.nan
            )
        elif padding == 'constant_edge':
            result = np.pad(array=np.array(result), pad_width=((pdist0, pdist1), (0, 0)), mode='edge')
        elif padding == 'extrapolation':
            # select edge training data for extrapolation
            train_range0 = min(window_size, result.shape[0])
            # training dataset
            X_train0 = np.array([range(0, train_range0)] * arr.shape[1]).T  # noqa: N806
            y_train0 = np.array(result)[0:train_range0, :]
            # training
            ex_model0 = LinearRegression()
            ex_model0.fit(X_train0, y_train0)
            # predict pad before
            X_estimate0 = np.array([range(-pdist0, 0)] * arr.shape[1]).T  # noqa: N806
            pads0 = pd.DataFrame(ex_model0.predict(X_estimate0))
            if pdist1 > 0:
                # select edge training data for extrapolation
                train_range1 = max(window_size, result.shape[0])
                # training dataset
                X_train1 = np.array(  # noqa: N806
                    [range(result.shape[0] - train_range1, result.shape[0])] * arr.shape[1]
                ).T
                y_train1 = np.array(result)[(result.shape[0] - train_range1) : result.shape[0], :]
                # training
                ex_model1 = LinearRegression()
                ex_model1.fit(X_train1, y_train1)
                # predict pad after
                X_estimate1 = np.array(  # noqa: N806
                    [range(result.shape[0], result.shape[0] + pdist1)] * arr.shape[1]
                ).T
                pads1 = ex_model1.predict(X_estimate1)
                # apply padding
                result = np.concatenate((pads0, result, pads1))
            else:
                result = np.concatenate((pads0, result))
        else:
            warnings.warn('\nUndefined padding! No padding is applied.\n', UserWarning, stacklevel=2)
        return result

    # Apply functions in chain with args
    @simple_type_validator
    def _chain_apply(  # noqa: C901
        self,
        data_array: Annotated[Any, arraylike_validator(ndim=2)],
        function_list: list[Callable],
        arg_lists: Optional[list[list]] = None,
        kwarg_dicts: Optional[list[dict]] = None,
    ) -> np.ndarray:
        """
        Apply a list of functions in chain, effect as a combined function.

        Parameters
        ----------
        data_array : 2D array-like
            Input data array to process.
        function_list : list of callables
            List of functions.
        arg_lists : list of list or None, optional
            Lists of additional positional arguments of the functions in a list. The functions and corresponding arg lists are matched by list index.
            If provided, arg_lists must have the same number of lists with the number of functions.
            Please note the data must be the first argument in all functions.
            The default is [].
        kwarg_dicts : list of dict or None, optional
            Dictionaries of keyword arguments of the functions in a list. The functions and corresponding kwarg dictionaries are matched by list index.
            If provided, kwarg_dicts must have the same number of dictionaries with the number of functions.
            The default is [].

        Returns
        -------
        np.ndarray
            Result after applying a chain of functions to the input data.

        Raises
        ------
        ValueError
            If number of arguments in addition to input data exceeds a supported maximum of 5.

        Examples
        --------
        Prepare an exemplary 2D data array::

            >>> arr = np.array([[1, 2, 3, 4, 5, 6], [2, 2, 4, 4, 6, 6]])

        Define functions to apply::

            >>> def func1(arr):
            ...     return np.std(arr, axis=0, keepdims=True)
            >>> def func2(arr):
            ...     return np.mean(arr, axis=0, keepdims=True)

        Sequentially apply two functions::

            >>> RollWindow(3)._chain_apply(arr, [func1, func2])

        Pass additional parameters to the functions to apply::

            >>> RollWindow(3)._chain_apply(
            ...     arr,
            ...     [np.std, np.mean],
            ...     kwarg_dicts=[{'axis': 0, 'keepdims': True}, {'axis': 0, 'keepdims': True}]
            ... )
        """  # noqa: E501

        # Validate arg lists
        if arg_lists is not None:
            assert isinstance(arg_lists, list)
            if len(arg_lists) != len(function_list):
                raise ValueError(
                    f"If provided, arg_lists must have the same number of lists with the number of functions, \
                        got number of arg lists: {len(arg_lists)}, \
                        got number of functions: {len(function_list)}"
                )
        # Validate kwarg dicts
        if kwarg_dicts is not None:
            assert isinstance(kwarg_dicts, list)
            if len(kwarg_dicts) != len(function_list):
                raise ValueError(
                    f"If provided, kwarg_dicts must have the same number of dictionaries with the number of functions, \
                        got number of kwarg dictionaries: {len(kwarg_dicts)}, \
                        got number of functions: {len(function_list)}"
                )

        # Get attributes
        window_size = self.window_size
        axis = self.axis
        roll_mode = self.roll_mode
        padding = self.padding
        axis_double_definition_warning = self.axis_double_definition_warning
        arr = np.array(data_array).astype(self.numtype)

        # Other config validation
        if window_size < 2:
            raise ValueError('window_size must be at least 2!')
        if axis == 1:
            arr = arr.T
        if (roll_mode == 'knn') & (padding != 'none'):
            warnings.warn(
                '\nRolling mode knn does not need padding, padding is not applied!\n', UserWarning, stacklevel=2
            )
            padding = 'none'

        # Set internal data shape in calculation
        self.__data_array_shape = arr.shape

        # Padding distance
        if roll_mode == 'center':
            pdist0 = int((window_size - 1) / 2)  # padding distance before
            pdist1 = int(np.ceil((window_size - 1) / 2))  # padding distance after
        elif roll_mode == 'end':
            pdist0 = int(window_size - 1)
            pdist1 = 0
        else:
            pdist0 = 0
            pdist1 = 0

        # Pre-calculation Padding
        # Extending edge padding before applying MA kernel
        if padding == 'ext_edge':
            arr = np.array(np.pad(array=np.array(arr), pad_width=((pdist0, pdist1), (0, 0)), mode='edge')).astype(
                self.numtype
            )

        # Function application to rolling window
        result = np.zeros_like(arr).astype('float64')

        # Rolling range
        if (roll_mode == 'center') | (roll_mode == 'end'):
            itr = range(pdist0, arr.shape[0] - pdist1)
        elif roll_mode == 'knn':
            itr = range(arr.shape[0])
        else:
            raise ValueError('undefined roll_mode! Please choose between center/end/knn')

        # Apply function to rolling window
        init_switch = 0
        for i in itr:
            self.__window_ind = i
            # Create window data
            if (roll_mode == 'center') | (roll_mode == 'end'):
                arr_current_window = arr[(i - pdist0) : (i + pdist1 + 1), :]  # window_size = pdist0 + pdist1 + 1
            elif roll_mode == 'knn':
                w1 = int((window_size - 1) / 2)
                w2 = int(np.ceil((window_size - 1) / 2))
                if (i >= w1) & (i <= (arr.shape[0] - w2 - 1)):
                    arr_current_window = arr[(i - w1) : (i + w2 + 1), :]
                elif i < (w1 - 1):
                    arr_current_window = arr[0:window_size, :]
                elif i > (arr.shape[0] - w2 - 1):
                    arr_current_window = arr[(arr.shape[0] - window_size) : arr.shape[0], :]

            # Chain apply of functions
            for i in range(len(function_list)):
                # Get function to apply
                func = function_list[i]
                # Validate function to apply
                if axis_double_definition_warning:
                    if 'axis' in inspect.signature(func).parameters.keys():
                        warnings.warn(
                            (
                                "\nPossible double definition of calculation axis found. "
                                + "Double definition of calculation axis must be avoided, "
                                + "it is recommended to fix the computation axis to 0 "
                                + "(along the row) in the function to apply.\n"
                            ),
                            UserWarning,
                            stacklevel=2,
                        )
                # Get args and kwargs
                if arg_lists is None:
                    args = []
                else:
                    assert isinstance(arg_lists, list)
                    args = arg_lists[i]
                if kwarg_dicts is None:
                    kwargs = {}
                else:
                    assert isinstance(kwarg_dicts, list)
                    kwargs = kwarg_dicts[i]
                # Apply function to window data
                arr_current_window = func(arr_current_window, *args, **kwargs)

            # Store result of arr_current_window
            arr_current_window = np.array(arr_current_window).reshape(1, -1).astype(self.numtype)
            if init_switch == 0:
                result = arr_current_window
                init_switch = 1
            else:
                result = np.concatenate((result, arr_current_window))

        # Post-calculation Padding
        result = self._padder(
            result=result, window_size=window_size, arr=arr, padding=padding, pdist0=pdist0, pdist1=pdist1
        )

        # axis reset
        if axis == 1:
            result = result.T

        assert isinstance(result, np.ndarray)
        return result


# %% Create test array


# Create test array
@simple_type_validator
def create_test_array(random: bool = False, shape: tuple[int, int] = (14, 12)) -> np.ndarray:
    """Create test 2D array-like data"""
    if random:
        np.random.seed(66)
        return np.random.random(shape)
    else:
        # Fixed array data with edge case
        arr = np.array(
            [
                [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                [10, 8, 20, 16, 100, 17, 10, 12, 14, 15, 10, 11],
                [10, 12, 14, 15, 10, 11, 9, 8, 20, 16, 77, 17],
                [10, 8, 20, 16, 7, 17, 10, 12, 14, 15, 10, 11],
                [10, 12, 14, 15, 10, 11, 9, 8, 20, 16, 77, 17],
                [10, 8, 20, 16, 7, 17, -100, 8, 20, 16, 7, 17],
                [10, 12, 14, 15, 10, 11, 9, 8, 20, 16, 77, 17],
                [10, 12, 14, 15, 10, 11, 10, 8, 20, 16, 7, 17],
                [10, 8, 20, 16, 77, 17, 10, 12, 14, 15, 10, 11],
                [10, 12, 14, 15, 10, 11, -100, 8, 20, 16, 7, 17],
                [10, 12, 14, 15, 10, 11, 9, 8, 20, 16, 77, 17],
                [10, 12, 14, 15, 10, 11, 10, 8, 20, 16, 7, 17],
                [10, 12, 14, 15, 10, 11, 9, 8, 20, 16, 77, 17],
                [10, 12, 14, 15, 10, 11, 10, 8, 20, 16, 7, 17],
            ]
        )
        if shape != arr.shape:
            core_arr = arr[0 : min(shape[0], arr.shape[0]), 0 : min(shape[1], arr.shape[1])]
            # Tile the array
            tiles_rows = (shape[0] + core_arr.shape[0] - 1) // core_arr.shape[0]
            tiles_cols = (shape[1] + core_arr.shape[1] - 1) // core_arr.shape[1]
            tiled = np.tile(arr, (tiles_rows, tiles_cols))
            return tiled[0 : shape[0], 0 : shape[1]]
        else:
            return arr
