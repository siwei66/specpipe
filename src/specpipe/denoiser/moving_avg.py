# -*- coding: utf-8 -*-
"""
1D data series denoising - kernel smoothing - moving average smoothing

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Sequential data denoising

import numpy as np
from typing import Union, Annotated, Any, Optional

from ..specio import simple_type_validator, arraylike_validator
from .utils import RollWindow, _to_2d_arr, _back_1d_arr
from .outlier import ArrayOutlier


# %% Common denoising techniques - kernel smoothing methods - moving average smoothing


# Moving average smoothing
class MovingAvg(RollWindow):
    """
    Apply moving average data smoothing to 2-dimensional array of 1-dimensional data series.
    The averaging calculation is applied to the rolling window.

    Attributes
    ----------
    window_size : int
        Size of rolling window.

    axis : int, optional
        Window is rolling along the assigned axis. The default is 1.

    mode : str, optional
        Rolling mode for kernel application. Available options:

        - "center" - window is created centered on current data point, better applicable to static data series.
        - "end" - for time series, window is created end with current data point, better applicable to time series.
        - "knn" - k-nearst neighbor mode, k is the window size.

        For "knn", the window is constructed with available data centered on current data point, except the edge windows.
        The edge windows are constructed on the k nearst neighbor values.

        The default is "center".

    padding : str, optional
        Padding approach for the output. Available options:

        - "none" - No padding applied, the resulting data series will be window_size - 1 smaller than the original data series.
        - "nan" - Missing values are set to numpy nan after applying the function.
        - "ext_edge" - Edge value of the first and last window are applied to pad before applying the smoothing function.
        - "constant_edge" - Missing values are filled with edge value after applying function.
        - "extrapolation" - Missing values are linearly extrapolated from edge values with length of window_size/2 (center mode), window_size-1 (end mode) after function application.

        The default is "extrapolation".

    numtype : str or type, optional
        Number type of given data, supported number type of Numpy. Default is float32.

    outlier_replacer : ArrayOutlier instance or None, optional
        Outlier removing object with defined attributes. The default is None.

    window_weights : 1D array-like, optional
        Weights array for weighted average. The default is None.

    gaussian_sigma : float, optional
        Sigma for Gaussian kernel. By default it is calculated as (window size - 1) / 6.

    axis_double_definition_warning : bool, optional
        If True, the duplicate definition warning will prompt when function has "axis" argument.

        The default is True. Set false for known application.

    Methods
    -------
    simple_moving_average
        Implemente simple moving average of input 2d data array.
    moving_median
        Implemente moving median of input 2d data array.
    weighted_moving_average
        Implemente weighted moving average of input 2d data array.
    gaussian_filter
        Implemente Gaussian smoothing of input 2d data array.

    Examples
    --------
    Basic usage with ``window_size``::

        >>> ma = MovingAvg(3)

    Use different ``window_size``::

        >>> ma = MovingAvg(4)

    Pad with ``numpy.nan``::

        >>> ma = MovingAvg(3, padding='nan')

    Specify rolling mode::

        >>> ma = MovingAvg(3, mode='knn')

    Compute along a different axis::

        >>> ma = MovingAvg(3, axis=0)
    """  # noqa: E501

    @simple_type_validator
    def __init__(
        self,
        window_size: int,
        axis: int = 1,
        mode: str = 'center',
        padding: str = 'extrapolation',
        numtype: Union[str, type] = 'float32',
        outlier_replacer: Optional[ArrayOutlier] = None,
        *,
        window_weights: Union[str, None, Annotated[Any, arraylike_validator(ndim=1)]] = None,
        gaussian_sigma: float = -1.0,
        axis_double_definition_warning: bool = True,
    ) -> None:
        super().__init__(
            window_size=window_size,
            axis=axis,
            roll_mode=mode,
            padding=padding,
            numtype=numtype,
            axis_double_definition_warning=axis_double_definition_warning,
        )
        self.numtype = numtype
        if window_weights is None:
            window_weight_values = np.array([1] * window_size).astype(self.numtype)
        elif isinstance(window_weights, str):
            if str(window_weights).lower() == 'gaussian':
                if gaussian_sigma == -1.0:
                    gaussian_sigma = (window_size - 1) / 6
                else:
                    gaussian_sigma = float(gaussian_sigma)
                # Create an array of indices centered around the middle
                x = np.arange(window_size) - (window_size - 1) / 2
                # Compute the Gaussian function
                window_weight_values = np.exp(-(x**2) / (2 * gaussian_sigma**2))
                # Normalize the kernel so that it sums to 1
                window_weight_values /= window_weight_values.sum()
            else:
                raise ValueError(
                    f"Invalid window_weights name: {window_weights}, \
                                 window_weights must be 1D array-like or 'gaussian'"
                )
        else:
            window_weight_values = np.array(window_weights).astype(self.numtype)
            if len(window_weights) != window_size:
                raise ValueError('Length of window size and weights does not match.')
        self.window_weights = window_weight_values
        self.gaussian_sigma = gaussian_sigma
        self.outlier_replacer: Optional[ArrayOutlier] = None
        if isinstance(outlier_replacer, ArrayOutlier):
            # Fix computation axis 0
            outlier_replacer.axis = 0
            self.outlier_replacer = outlier_replacer
        elif outlier_replacer is None:
            self.outlier_replacer = outlier_replacer
        else:
            raise ValueError("'outlier_replacer' must be an instance of class ArrayOutlier.")

    @simple_type_validator
    def _sma_kernel(self, window_array: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Simple moving average kernel.

        Parameters
        ----------
        window_array : numpy.ndarray
            1D window data in a 2D array.

        Returns
        -------
        numpy.ndarray
            Calculation results.

        Raises
        ------
        ValueError
            window_array does not match window_size.
        """
        if window_array.shape[0] != self.window_size:
            raise ValueError('window_array does not match window_size.')
        window_array = np.array(np.nanmean(window_array, axis=0))
        assert isinstance(window_array, np.ndarray)
        return window_array

    @simple_type_validator
    def simple_moving_average(
        self,
        data_array: Union[Annotated[Any, arraylike_validator(ndim=2)], Annotated[Any, arraylike_validator(ndim=1)]],
    ) -> np.ndarray:
        """
        Implemente simple moving average of input 2d data array.

        Parameters
        ----------
        data_array : numpy.ndarray
            input data series in 2d data array.

        Returns
        -------
        numpy.ndarray
            Smoothing result.

        Examples
        --------
        >>> ma = MovingAvg(5)
        >>> ma.simple_moving_average([1, 2, 3, 4, 5, 6, 77, 88, 9, 10])
        >>> ma.simple_moving_average([[1, 2, 3, 4, 5, 6, 77, 88, 9, 10], [1, 22, 33, 4, 5, 6, 7, 8, 9, 10]])

        Add to prepared ``SpecPipe`` instance ``pipe`` for ROI pixel spectrum processing::

            >>> pipe.add_process(6, 6, 0, MovingAvg(5).simple_moving_average)

        Add to prepared ``SpecPipe`` instance ``pipe`` for the processing of 1D sample data::

            >>> pipe.add_process(7, 7, 0, MovingAvg(5).simple_moving_average)
        """
        data_array = _to_2d_arr(data_array).astype(self.numtype)
        if self.outlier_replacer is None:
            result = super()._apply(data_array, self._sma_kernel)
        else:
            result = super()._chain_apply(
                data_array=data_array, function_list=[self.outlier_replacer.replace, self._sma_kernel]
            )
        return _back_1d_arr(result)

    @simple_type_validator
    def _median_kernel(self, window_array: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Moving median kernel.

        Parameters
        ----------
        window_array : numpy.ndarray
            1D window data in a 2D array.

        Returns
        -------
        numpy.ndarray
            Calculation results.

        Raises
        ------
        ValueError
            window_array does not match window_size.
        """
        if self.window_size < 3:
            raise ValueError('window_size must be at least 3.')
        if window_array.shape[0] != self.window_size:
            raise ValueError('window_array does not match window_size.')
        window_array = np.array(np.nanmedian(window_array, axis=0))
        assert isinstance(window_array, np.ndarray)
        return window_array

    @simple_type_validator
    def moving_median(
        self,
        data_array: Union[Annotated[Any, arraylike_validator(ndim=2)], Annotated[Any, arraylike_validator(ndim=1)]],
    ) -> np.ndarray:
        """
        Implemente moving median of input 2d data array.

        Parameters
        ----------
        data_array : numpy.ndarray.
            input data series in 2d data array.

        Returns
        -------
        numpy.ndarray
            Smoothing result.

        Examples
        --------
        >>> ma = MovingAvg(5)
        >>> ma.moving_median([1, 2, 3, 4, 5, 6, 77, 88, 9, 10])
        >>> ma.moving_median([[1, 2, 3, 4, 5, 6, 77, 88, 9, 10], [1, 22, 33, 4, 5, 6, 7, 8, 9, 10]])

        Add to prepared ``SpecPipe`` instance ``pipe`` for ROI pixel spectrum processing::

            >>> pipe.add_process(6, 6, 0, MovingAvg(5).moving_median)

        Add to prepared ``SpecPipe`` instance ``pipe`` for the processing of 1D sample data::

            >>> pipe.add_process(7, 7, 0, MovingAvg(5).moving_median)
        """
        data_array = _to_2d_arr(data_array).astype(self.numtype)
        if self.outlier_replacer is None:
            result = super()._apply(data_array, self._median_kernel)
        else:
            result = super()._chain_apply(
                data_array=data_array, function_list=[self.outlier_replacer.replace, self._median_kernel]
            )
        return _back_1d_arr(result)

    @simple_type_validator
    def _wma_kernel(self, window_array: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        weighted moving average kernel.

        Parameters
        ----------
        window_array : numpy.ndarray
            1D window data in a 2D array.

        Returns
        -------
        numpy.ndarray
            Calculation results.

        Raises
        ------
        ValueError
            window_size must be at least 3.
        ValueError
            window_array does not match window_size.
        ValueError
            weights for weighted average is missing.
        ValueError
            Weights must be 1D array or list.
        ValueError
            The length of weights must be same as length of data series to be calculated.
        """
        if self.window_size < 3:
            raise ValueError('window_size must be at least 3.')
        if window_array.shape[0] != self.window_size:
            raise ValueError('window_array does not match window_size.')
        window_array = np.array(window_array)
        window_weights = self.window_weights
        # Validate window weights
        if len(window_weights) != window_array.shape[0]:
            raise ValueError(
                f"The window_weights length must be same as window_array length at the computation axis. \
                    got window_weights: {window_weights}, got window_array shape: {window_array.shape}"
            )
        # calculated weighted avg ignoring nan
        window_array[window_array == 0] = 1e-38  # prevent 0 in division
        window_weights[window_weights == 0] = 1e-38  # prevent 0 in division
        dp = window_array * window_weights.reshape(-1, 1)
        nanw = window_array / window_array * window_weights.reshape(-1, 1)  # weight sum excluding nan
        wavg = np.array(np.nansum(dp, axis=0) / np.nansum(nanw, axis=0))
        assert isinstance(wavg, np.ndarray)
        return wavg

    @simple_type_validator
    def _gaussian_kernel(self, window_array: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Moving median kernel.

        Parameters
        ----------
        window_array : numpy.ndarray
            1D window data in a 2D array.
        gaussian_sigma : float
            sigma applied in gaussian kernel, if not given, the sigma will be (window size-1)/6 by default.

        Returns
        -------
        float
            Calculation results.

        Raises
        ------
        ValueError
            window_size must be at least 3.
        ValueError
            window_array does not match window_size.
        """
        if self.window_size < 3:
            raise ValueError('window_size must be at least 3.')
        if window_array.shape[0] != self.window_size:
            raise ValueError('window_array does not match window_size.')
        gaussian_sigma = self.gaussian_sigma
        if gaussian_sigma == -1.0:  # Default gaussian sigma
            gaussian_sigma = (self.window_size - 1) / 6
        else:
            gaussian_sigma = float(gaussian_sigma)
        # Create an array of indices centered around the middle
        x = np.arange(self.window_size) - (self.window_size - 1) / 2
        # Compute the Gaussian function
        window_weights = np.exp(-(x**2) / (2 * gaussian_sigma**2))
        # Normalize the kernel so that it sums to 1
        window_weights /= window_weights.sum()
        # calculated weighted avg ignoring nan
        window_array[window_array == 0] = 1e-38  # prevent 0 in division
        window_weights[window_weights == 0] = 1e-38  # prevent 0 in division
        dp = window_array * window_weights.reshape(-1, 1)
        nanw = window_array / window_array * window_weights.reshape(-1, 1)  # weight sum excluding nan
        wavg = np.array(np.nansum(dp, axis=0) / np.nansum(nanw, axis=0))
        assert isinstance(wavg, np.ndarray)
        return wavg

    @simple_type_validator
    def weighted_moving_average(
        self,
        data_array: Union[Annotated[Any, arraylike_validator(ndim=2)], Annotated[Any, arraylike_validator(ndim=1)]],
    ) -> np.ndarray:
        """
        Implemente weighted moving average of input 2d data array.

        Parameters
        ----------
        data_array : 1D array-like or 2D array-like
            1D data array or 2D data array of 1D series data.

        Returns
        -------
        numpy.ndarray
            Smoothing result.

        Examples
        --------
        >>> ma = MovingAvg(5)
        >>> ma.weighted_moving_average([1, 2, 3, 4, 5, 6, 77, 88, 9, 10])
        >>> ma.weighted_moving_average([[1, 2, 3, 4, 5, 6, 77, 88, 9, 10], [1, 22, 33, 4, 5, 6, 7, 8, 9, 10]])

        Add to prepared ``SpecPipe`` instance ``pipe`` for ROI pixel spectrum processing::

            >>> pipe.add_process(6, 6, 0, MovingAvg(5).weighted_moving_average)

        Add to prepared ``SpecPipe`` instance ``pipe`` for the processing of 1D sample data::

            >>> pipe.add_process(7, 7, 0, MovingAvg(5).weighted_moving_average)
        """
        data_array = _to_2d_arr(data_array).astype(self.numtype)
        if self.outlier_replacer is None:
            result = super()._apply(data_array, self._wma_kernel)
        else:
            result = super()._chain_apply(
                data_array=data_array, function_list=[self.outlier_replacer.replace, self._wma_kernel]
            )

        return _back_1d_arr(result)

    @simple_type_validator
    def gaussian_filter(
        self,
        data_array: Union[Annotated[Any, arraylike_validator(ndim=2)], Annotated[Any, arraylike_validator(ndim=1)]],
    ) -> np.ndarray:
        """
        Implemente Gaussian smoothing of input 2d data array.

        Parameters
        ----------
        data_array : 1D array-like or 2D array-like
            1D data array or 2D data array of 1D series data.

        Returns
        -------
        numpy.ndarray
            Smoothing result.

        Examples
        --------
        >>> ma = MovingAvg(5)
        >>> ma.gaussian_filter([1, 2, 3, 4, 5, 6, 77, 88, 9, 10])
        >>> ma.gaussian_filter([[1, 2, 3, 4, 5, 6, 77, 88, 9, 10], [1, 22, 33, 4, 5, 6, 7, 8, 9, 10]])

        Add to prepared ``SpecPipe`` instance ``pipe`` for ROI pixel spectrum processing::

            >>> pipe.add_process(6, 6, 0, MovingAvg(5).gaussian_filter)

        Add to prepared ``SpecPipe`` instance ``pipe`` for the processing of 1D sample data::

            >>> pipe.add_process(7, 7, 0, MovingAvg(5).gaussian_filter)
        """
        data_array = _to_2d_arr(data_array).astype(self.numtype)
        if self.outlier_replacer is None:
            result = super()._apply(data_array, self._gaussian_kernel)
        else:
            result = super()._chain_apply(
                data_array=data_array, function_list=[self.outlier_replacer.replace, self._gaussian_kernel]
            )
        return _back_1d_arr(result)
