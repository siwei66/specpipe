# -*- coding: utf-8 -*-
"""
1D data series denoising - kernel smoothing - local polynomial smoothing

Copyright (c) 2025 Siwei Luo. MIT License.
"""

import numpy as np
import warnings
from typing import Union, Optional, Annotated, Any, Callable

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from ..specio import simple_type_validator, arraylike_validator
from .outlier import ArrayOutlier
from .utils import RollWindow, _to_2d_arr, _back_1d_arr


# %% Common denoising techniques - kernel smoothing methods


class LocalPolynomial(RollWindow):
    """
    Apply local polynomial data smoothing to 2D array of 1D series data.
    Local polynomial regression is applied to the rolling window to smooth the data series.

    Attributes
    ----------
    window_size : int
        Size of rolling window, must be at least 2.

    polynomial_order : int
        Order of used polynomial regression.

    axis : int, optional
        Window is rolling along the assigned axis. The default is 1.

    mode : str, optional
        Rolling mode for kernel application. Available options:

        - "center" - window is created centered on current data point, better applicable to static data series.
        - "end" - for time series, window is created end with current data point, better applicable to time series.
        - "knn" - k-nearst neighbor mode, k is the window size.

        For "knn", the window is constructed with available data centered on current data point, except the edge windows.
        The edge windows are constructed on the k nearst neighbor values.

        Note: "knn" cannot be applied to "savitzky_golay_filter".

        The default is "center".

    padding : str, optional
        Padding approach for rolling mode "center" and "end". Available options:

        - "none" - No padding applied. For rolling mode "center" and "end", the resulting data series will be window_size - 1 smaller than the original data series without padding.
        - "nan" - Missing values are set to numpy nan after applying the function. "nan" cannot be applied to "savitzky_golay_filter".
        - "ext_edge" - Edge value of the first and last window are applied to pad before applying the smoothing function.
        - "constant_edge" - Missing values are filled with edge value after applying function.
        - "extrapolation" - Missing values are linearly extrapolated from edge values with length of window_size/2 (center mode), window_size-1 (end mode) after function application.

        The default is "ext_edge".

    numtype : str, optional
        Number type of given data, supported number type of Numpy. Default is float32.

    outlier_replacer : ArrayOutlier object, optional
        Outlier removing object with defined attributes. The default is None.

    wfunc_x : str or function
        Weight distribution function for LOWESS filter at x axis.

        The weight function can be a distribution name or a custom function.

        Available weight distribution names:

        "tricubic" / "triangular" / "cosine" / "gaussian" / "epanechnikov" / "exponential" / "uniform"

        The default is "tricubic".
        Configure distribution feature using shape parameter wdist_param_x.

    wdist_param_x : float
        Shape parameter of weight distribution at x axis for LOWESS filter.

        Default is ``-1.0``, which applies the default shape parameter for the selected weight distribution.

        The shape parameter affects the weight computation as follows:

            For ``"triangular"`` weight distribution:

                ``sample_weights = 1 - abs(distance_to_focal_point) / b``

                ``b`` is the shape parameter. Default is the distance of window boundary to focal point.

            For ``"cosine"`` weight distribution:

                ``sample_weights = cos(Pi * distance_to_focal_point / (2 * s))``

                ``s`` is the shape parameter. Default is the distance of window boundary to focal point.

            For ``"gaussian"`` weight distribution:

                ``sample_weights = exp(-(distance_to_focal_point ** 2) / (2 * (sigma ** 2)))``

                ``sigma`` is the shape parameter. Default is 1.0.

            For ``"epanechnikov"`` weight distribution:

                ``sample_weights = 1 - (distance_to_focal_point / h) ** 2``

                ``h`` is the shape parameter. Default is the distance of window boundary to focal point.

            For ``"exponential"`` weight distribution:

                ``sample_weights = exp(-abs(distance_to_focal_point / h))``

                ``h`` is the shape parameter. Default is the distance of window boundary to focal point.

    wfunc_y : str or function
        Weight distribution function for LOWESS filter at y axis. The weight function can be a distribution name or a custom function.

        Available weight distribution names:

        "tricubic" / "triangular" / "cosine" / "gaussian" / "epanechnikov" / "exponential" / "uniform"

        The default is "uniform".
        Configure distribution feature using shape parameter wdist_param_y.

    wdist_param_y : float
        Shape parameter of weight distribution at y axis for LOWESS filter.

        Default is ``-1.0``, which applies the default shape parameter for the selected weight distribution.
        See ``wdist_param_x`` for details.

    axis_double_definition_warning : bool, optional
        If True, the duplicate definition warning will prompt when function has "axis" argument.

        The default is True. Set false for known application.

    Methods
    -------
    savitzky_golay_filter
        Implemente Savitzky-Golay smoothing of input 2D data array.
    simple_polynomial_filter
        Implemente simple polynomial smoothing of input 2D data array.
    lowess_filter
        Implemente LOWESS smoothing of input 2D data array.

    Examples
    --------
    Basic usage with ``window_size`` and ``polynomial_order``::

        >>> lp = LocalPolynomial(3, 1)

    Use different ``window_size``::

        >>> lp = LocalPolynomial(4, 1)

    Use different ``polynomial_order``::

        >>> lp = LocalPolynomial(3, 2)

    Pad with ``numpy.nan``::

        >>> lp = LocalPolynomial(3, 1, padding='nan')

    Specify rolling mode::

        >>> lp = LocalPolynomial(3, 1, mode='knn')

    Compute along a different axis::

        >>> lp = LocalPolynomial(3, 1, axis=0)
    """  # noqa: E501

    @simple_type_validator
    def __init__(
        self,
        window_size: int,
        polynomial_order: int,
        axis: int = 1,
        mode: str = 'center',
        padding: str = 'ext_edge',
        numtype: str = 'float32',
        outlier_replacer: Optional[ArrayOutlier] = None,
        *,
        wfunc_x: Union[str, Callable] = 'tricubic',
        wdist_param_x: float = -1.0,
        wfunc_y: Union[str, Callable] = 'uniform',
        wdist_param_y: float = -1.0,
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
        if window_size < 3:
            raise ValueError('Window_size must be at least 3.')
        if polynomial_order < 1:
            raise ValueError('Polynomial order must be at least 1.')
        if polynomial_order >= window_size:
            raise ValueError('Polynomial_order must be less than window_size.')
        self.polynomial_order = polynomial_order
        self.wfunc_x = wfunc_x
        self.wdist_param_x = wdist_param_x
        self.wfunc_y = wfunc_y
        self.wdist_param_y = wdist_param_y
        self.outlier_replacer = outlier_replacer
        self.numtype = numtype
        # Private attribute for reducing redundant calculation
        self.sg_coeff = np.array([])
        self.focal_ind0 = int((self.window_size - 1) / 2)  # for knn and center mode
        self.focal_ind1 = int(self.window_size - 1)  # for end mode
        self.focal_ind = 0
        self.sample_weight_x = np.full(window_size, 1)  # Default uniform distribution for equal weights
        self.sample_weight_y = np.full(window_size, 1)
        self.sample_weight_calc_switch_x = 0

    # SG kernel
    @simple_type_validator
    def _savitzky_golay_kernel(self, window_array: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Savitzky-Golay kernel.

        Parameters
        ----------
        window_array : numpy.ndarray
            2D window data of 1D data series.

        Returns
        -------
        numpy.ndarray
            Resulting smoothed data.

        Raises
        ------
        ValueError
            If window_size for savitzky-golay kernel is not odd.
        ValueError
            If window_array does not match window_size.
        ValueError
            If window_array has nan value.
        """
        window_size = self.window_size
        if window_size % 2 == 0:
            raise ValueError('window_size must be an odd integer for Savitzky-Golay filter.')
        if window_array.shape[0] != self.window_size:
            raise ValueError('window_array does not match window_size.')
        polyorder = self.polynomial_order
        if np.any(np.isnan(window_array)):
            raise ValueError(
                'window_array contains NaN values. \
                             Savitzky-Golay filter cannot process data with NaN values.'
            )
        if self.sg_coeff.shape != (polyorder, window_size):
            # Half window size
            hw = int((window_size - 1) / 2)
            # Create Vandermonde matrix
            vandm = np.vander(np.arange(-hw, hw + 1), N=polyorder + 1)
            # Calculate coefficients
            coeffs = np.linalg.pinv(vandm)
            self.sg_coeff = coeffs
        else:
            coeffs = self.sg_coeff
        smoothed = np.sum(np.dot(coeffs, window_array), axis=0).reshape(-1, 1).T
        return np.array(smoothed)

    # Perform SG filter on 2-d array of 1-d series data
    @simple_type_validator
    def savitzky_golay_filter(
        self,
        data_array: Union[Annotated[Any, arraylike_validator(ndim=2)], Annotated[Any, arraylike_validator(ndim=1)]],
    ) -> np.ndarray:
        """
        Implemente Savitzky-Golay smoothing of input 2D data array.

        Parameters
        ----------
        data_array : 1D array-like or 2D array-like
            1D data array or 2D data array of 1D series data.

        Returns
        -------
        numpy.ndarray
            Resulting smoothed data array.

        Examples
        --------
        >>> lp = LocalPolynomial(5, polynomial_order=2)
        >>> lp.savitzky_golay_filter([1, 2, 3, 4, 5, 6, 77, 88, 9, 10])
        >>> lp.savitzky_golay_filter([[1, 2, 3, 4, 5, 6, 77, 88, 9, 10], [1, 22, 33, 4, 5, 6, 7, 8, 9, 10]])

        Add to prepared ``SpecPipe`` instance ``pipe`` for ROI pixel spectrum processing::

            >>> pipe.add_process(6, 6, 0, LocalPolynomial(5, polynomial_order=2).savitzky_golay_filter)

        Add to prepared ``SpecPipe`` instance ``pipe`` for the processing of 1D sample data::

            >>> pipe.add_process(7, 7, 0, LocalPolynomial(5, polynomial_order=2).savitzky_golay_filter)
        """
        data_array = _to_2d_arr(data_array).astype(self.numtype)

        # Validate rolling mode and padding approach
        if self.roll_mode == 'knn':
            raise ValueError("Mode 'knn' is not applicable to Savitzky-Golay filter, 'center' mode is applied instead.")
        if self.padding == 'nan':
            raise ValueError(
                "'nan' padding is not applicable to Savitzky-Golay filter. 'ext_edge' padding is applied instead."
            )

        # Apply kernel
        if self.outlier_replacer is None:
            result = super()._apply(data_array, self._savitzky_golay_kernel)
        else:
            result = super()._chain_apply(
                data_array=data_array, function_list=[self.outlier_replacer.replace, self._savitzky_golay_kernel]
            )

        return _back_1d_arr(result)

    # Polynomial regression on 1-d series data, return estimation of focal point
    @simple_type_validator
    def _polynomial_estimator(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Perform polynomial regression on local window data.

        Parameters
        ----------
        y : numpy.ndarray
            Responses.
        x : numpy.ndarray
            Predictors.

        Returns
        -------
        float
            Estimation values.
        """
        x1 = x[~np.isnan(y)]
        y1 = y[~np.isnan(y)]
        poly_features = PolynomialFeatures(degree=self.polynomial_order)
        x_poly = poly_features.fit_transform(x.reshape(-1, 1))
        x_poly1 = poly_features.fit_transform(x1.reshape(-1, 1))
        model = LinearRegression()
        model.fit(x_poly1, y1)
        yb = model.predict(x_poly)
        return float(yb[self.focal_ind])

    # Perform polynomial regression on all column data series in a 2D array
    @simple_type_validator
    def _simple_polynomial_kernel(self, window_array: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Polynomial kernel.

        Parameters
        ----------
        window_array : numpy.ndarray
            2D window data of 1D data series.

        Returns
        -------
        numpy.ndarray
            Resulting smoothed data.

        Raises
        ------
        ValueError
            If window_array does not match window_size.
        ValueError
            If number of non-NaN values less than polynomial_order+1.
        """
        if window_array.shape[0] != self.window_size:
            raise ValueError('window_array does not match window_size.')
        if np.count_nonzero(~np.isnan(window_array)) < self.polynomial_order + 1:
            raise ValueError(
                f"Number of non-NaN values less than {self.polynomial_order+1}, "
                + f"not enough for fitting polynomial kernel with order {self.polynomial_order}."
            )
        # set focal point index in the window
        window_size = self.window_size
        if self.roll_mode == 'knn':
            self.focal_ind = self.focal_ind0  # middle range focus on center
            if self._window_ind < self.focal_ind:  # start
                self.focal_ind = self._window_ind
            elif self._window_ind > self._data_array_shape[0] - window_size + self.focal_ind:  # end
                self.focal_ind = self._window_ind + window_size - self._data_array_shape[0]
        elif self.roll_mode == 'center':
            self.focal_ind = self.focal_ind0
        elif self.roll_mode == 'end':
            self.focal_ind = self.focal_ind1
        # Regression by column
        x = np.linspace(0, window_size - 1, window_size)
        y_pred_list = []
        for i in range(window_array.shape[1]):
            y_pred_list.append(self._polynomial_estimator(window_array[:, i], x))
        y_pred = np.array(y_pred_list).reshape(-1, 1).T
        assert isinstance(y_pred, np.ndarray)
        return y_pred

    # Perform simple polynomial filter on 2-d array of 1-d series data
    @simple_type_validator
    def simple_polynomial_filter(
        self,
        data_array: Union[Annotated[Any, arraylike_validator(ndim=2)], Annotated[Any, arraylike_validator(ndim=1)]],
    ) -> np.ndarray:
        """
        Implemente simple polynomial smoothing of input 2D data array.

        Parameters
        ----------
        data_array : 1D array-like or 2D array-like
            1D data array or 2D data array of 1D series data.

        Returns
        -------
        numpy.ndarray
            Resulting smoothed data array.

        Examples
        --------
        >>> lp = LocalPolynomial(5, polynomial_order=2)
        >>> lp.simple_polynomial_filter([1, 2, 3, 4, 5, 6, 77, 88, 9, 10])
        >>> lp.simple_polynomial_filter([[1, 2, 3, 4, 5, 6, 77, 88, 9, 10], [1, 22, 33, 4, 5, 6, 7, 8, 9, 10]])

        Add to prepared ``SpecPipe`` instance ``pipe`` for ROI pixel spectrum processing::

            >>> pipe.add_process(6, 6, 0, LocalPolynomial(5, polynomial_order=2).simple_polynomial_filter)

        Add to prepared ``SpecPipe`` instance ``pipe`` for the processing of 1D sample data::

            >>> pipe.add_process(7, 7, 0, LocalPolynomial(5, polynomial_order=2).simple_polynomial_filter)
        """
        data_array = _to_2d_arr(data_array).astype(self.numtype)

        # Apply kernel
        if self.outlier_replacer is None:
            result = super()._apply(data_array, self._simple_polynomial_kernel)
        else:
            result = super()._chain_apply(
                data_array=data_array, function_list=[self.outlier_replacer.replace, self._simple_polynomial_kernel]
            )

        return _back_1d_arr(result)

    # weight function for lowess
    @staticmethod
    @simple_type_validator
    def _tricubic(sample_distances_to_focal: Annotated[np.ndarray, arraylike_validator(ndim=1)]) -> np.ndarray:
        """Tricubic kernel function mapping distances from focus to weights."""
        fdist = sample_distances_to_focal
        data_boundary = max(abs(fdist))
        if data_boundary == 0:  # Return uniform weights for perfectly smooth data
            return np.full(fdist.shape, 1.0)
        else:
            sample_weights = (1 - (fdist / data_boundary) ** 3) ** 3
            sample_weights[sample_weights < 0] = 0.0
            return np.array(sample_weights)

    @staticmethod
    @simple_type_validator
    def _triangular(
        sample_distances_to_focal: Annotated[np.ndarray, arraylike_validator(ndim=1)], b: float = -1.0
    ) -> np.ndarray:
        """Triangular kernel function mapping distances from focus to weights."""
        fdist = sample_distances_to_focal
        data_boundary = float(max(abs(fdist)))
        if data_boundary == 0:  # Return uniform weights for perfectly smooth data
            return np.full(fdist.shape, 1.0)
        if b == -1:
            b = data_boundary
        if b < data_boundary:
            raise ValueError(f'Kernel parameter b must be at least {data_boundary}')
        sample_weights = 1 - abs(fdist) / b
        sample_weights[sample_weights < 0] = 0.0
        return np.array(sample_weights)

    @staticmethod
    @simple_type_validator
    def _cosine(
        sample_distances_to_focal: Annotated[np.ndarray, arraylike_validator(ndim=1)], s: float = -1.0
    ) -> np.ndarray:
        """Cosine kernel function mapping distances from focus to weights."""
        fdist = sample_distances_to_focal
        data_boundary = float(max(abs(fdist)))
        if data_boundary == 0:  # Return uniform weights for perfectly smooth data
            return np.full(fdist.shape, 1.0)
        if s == -1:
            s = data_boundary
        if s < data_boundary:
            raise ValueError(f'Kernel parameter s must be at least {data_boundary}')
        sample_weights = np.cos(np.pi * fdist / 2 / s)
        sample_weights[sample_weights < 0] = 0.0
        return np.array(sample_weights)

    @staticmethod
    @simple_type_validator
    def _gaussian(
        sample_distances_to_focal: Annotated[np.ndarray, arraylike_validator(ndim=1)], sigma: float = -1.0
    ) -> np.ndarray:
        """Gaussian kernel function mapping distances from focus to weights."""
        fdist = sample_distances_to_focal
        if sigma == -1:
            sigma = 1.0
        if sigma <= 0:
            raise ValueError('Kernel parameter sigma must be positive')
        sample_weights = np.exp(-(fdist**2) / (2 * (sigma**2)))
        return np.array(sample_weights)

    @staticmethod
    @simple_type_validator
    def _epanechnikov(
        sample_distances_to_focal: Annotated[np.ndarray, arraylike_validator(ndim=1)], h: float = -1.0
    ) -> np.ndarray:
        """Epanechnikov kernel function mapping distances from focus to weights."""
        fdist = sample_distances_to_focal
        data_boundary = float(max(abs(fdist)))
        if data_boundary == 0:  # Return uniform weights for perfectly smooth data
            return np.full(fdist.shape, 1.0)
        if h == -1:
            h = data_boundary
        if abs(h) < data_boundary:
            raise ValueError(f'Kernel parameter c must be at least {data_boundary}')
        sample_weights = 1 - (fdist / h) ** 2
        sample_weights[sample_weights < 0] = 0.0
        return np.array(sample_weights)

    @staticmethod
    @simple_type_validator
    def _exponential(
        sample_distances_to_focal: Annotated[np.ndarray, arraylike_validator(ndim=1)], h: float = -1.0
    ) -> np.ndarray:
        """Exponential kernel function mapping distances from focus to weights."""
        fdist = sample_distances_to_focal
        if h == -1:
            h = 1
        sample_weights = np.exp(-abs(fdist / h))
        return np.array(sample_weights)

    # Polynomial regression on 1-d series data, return estimation of focal point
    @simple_type_validator
    def _lowess_estimator(self, y: np.ndarray, x: np.ndarray) -> float:  # noqa: C901
        """
        Perform locally weighted regression.

        Parameters
        ----------
        y : numpy.ndarray
            Responses.
        x : numpy.ndarray
            Predictors.

        Returns
        -------
        float
            Estimation values.
        """
        # attributes pass
        xweight = self.wfunc_x
        yweight = self.wfunc_y
        # Remove nan
        x1 = x[~np.isnan(y)]
        y1 = y[~np.isnan(y)]
        # calculate data point x weight in regression
        fdist_x = np.array(range(self.window_size)) - self.focal_ind  # x distances to focal
        if self.sample_weight_calc_switch_x < 2:
            if type(self.wfunc_x) is str:
                if xweight == 'uniform':
                    self.sample_weight_x = np.full(len(fdist_x), 1.0)
                elif xweight == 'tricubic':
                    self.sample_weight_x = self._tricubic(fdist_x)
                elif xweight == 'triangular':
                    self.sample_weight_x = self._triangular(fdist_x, self.wdist_param_x)
                elif xweight == 'cosine':
                    self.sample_weight_x = self._cosine(fdist_x, self.wdist_param_x)
                elif xweight == 'gaussian':
                    self.sample_weight_x = self._gaussian(fdist_x, self.wdist_param_x)
                elif xweight == 'epanechnikov':
                    self.sample_weight_x = self._epanechnikov(fdist_x, self.wdist_param_x)
                elif xweight == 'exponential':
                    self.sample_weight_x = self._exponential(fdist_x, self.wdist_param_x)
                else:
                    raise ValueError('Invalid x weight function name, please provide weight function instead.')
            else:
                try:
                    assert callable(xweight)
                    self.sample_weight_x = xweight(fdist_x)
                except Exception as e:
                    raise ValueError('Failure in custom weight function application.') from e
        sample_weight1_x = self.sample_weight_x[~np.isnan(y)]  # remove nan
        # calculate data point y weight in regression
        if (y[self.focal_ind] is not None) & (~np.isnan(y[self.focal_ind])):
            fdist_y = y - y[self.focal_ind]
            if type(self.wfunc_y) is str:
                if yweight == 'uniform':
                    self.sample_weight_y = np.full(len(fdist_y), 1.0)
                elif yweight == 'tricubic':
                    self.sample_weight_y = self._tricubic(fdist_y)
                elif yweight == 'triangular':
                    self.sample_weight_y = self._triangular(fdist_y, self.wdist_param_y)
                elif yweight == 'cosine':
                    self.sample_weight_y = self._cosine(fdist_y, self.wdist_param_y)
                elif yweight == 'gaussian':
                    self.sample_weight_y = self._gaussian(fdist_y, self.wdist_param_y)
                elif yweight == 'epanechnikov':
                    self.sample_weight_y = self._epanechnikov(fdist_y, self.wdist_param_y)
                elif yweight == 'exponential':
                    self.sample_weight_y = self._exponential(fdist_y, self.wdist_param_y)
                else:
                    raise ValueError('Invalid y weight function name, please provide weight function instead.')
            else:
                try:
                    assert callable(yweight)
                    self.sample_weight_y = yweight(fdist_y)
                except Exception as e:
                    raise ValueError('Failure in custom weight function application.') from e
        else:
            warnings.warn('\n y weights ignored due to NaN values.\n', UserWarning, stacklevel=2)
        sample_weight1y = self.sample_weight_y[~np.isnan(y)]
        sample_weight1 = np.sqrt(sample_weight1_x * sample_weight1y)
        # Polynomial feature preparation
        poly_features = PolynomialFeatures(degree=self.polynomial_order)
        x_poly = poly_features.fit_transform(x.reshape(-1, 1))
        x_poly1 = poly_features.fit_transform(x1.reshape(-1, 1))
        # Model fitting and applying
        model = LinearRegression()
        model.fit(x_poly1, y1, sample_weight=sample_weight1)
        yb = model.predict(x_poly)
        yb = np.array(yb)
        return float(yb[self.focal_ind])

    # lowess kernel
    @simple_type_validator
    def _lowess_kernel(self, window_array: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Locally weighted scatterplot smoothing kernel.

        Parameters
        ----------
        window_array : numpy.ndarray
            Window 2D array of 1D series data.

        Returns
        -------
        numpy.ndarray
            Resulting smoothed window array.

        Raises
        ------
        ValueError
            If window_array does not match window_size.
        ValueError
            If number of non-NaN values less than polynomial_order+1.
        """
        if window_array.shape[0] != self.window_size:
            raise ValueError('Window_array does not match window_size.')
        if np.count_nonzero(~np.isnan(window_array)) < self.polynomial_order + 1:
            raise ValueError(
                f"Number of non-NaN values less than {self.polynomial_order + 1}, "
                + f"which is not enough for fitting polynomial kernel with order {self.polynomial_order}."
            )
        # set focal point index in the window
        window_size = self.window_size
        if self.roll_mode == 'knn':
            self.focal_ind = self.focal_ind0  # middle range focus on center
            if self._window_ind < self.focal_ind:  # start
                self.focal_ind = self._window_ind
            elif self._window_ind > self._data_array_shape[0] - window_size + self.focal_ind:  # end
                self.focal_ind = self._window_ind + window_size - self._data_array_shape[0]
        elif self.roll_mode == 'center':
            self.focal_ind = self.focal_ind0
        elif self.roll_mode == 'end':
            self.focal_ind = self.focal_ind1
        # weight calc switch
        if self.focal_ind != self.focal_ind0:
            self.sample_weight_calc_switch_x = 0
        else:
            self.sample_weight_calc_switch_x = self.sample_weight_calc_switch_x + 1
        # Regression by column
        x = np.linspace(0, window_size - 1, window_size)
        y_pred_list = []
        for i in range(window_array.shape[1]):
            y_pred_list.append(self._lowess_estimator(window_array[:, i], x))
        y_pred = np.array(y_pred_list).reshape(-1, 1).T
        assert isinstance(y_pred, np.ndarray)
        return y_pred

    # Perform lowess filter on 2-d array of 1-d series data
    @simple_type_validator
    def lowess_filter(
        self,
        data_array: Union[Annotated[Any, arraylike_validator(ndim=2)], Annotated[Any, arraylike_validator(ndim=1)]],
    ) -> np.ndarray:
        """
        Implemente LOWESS smoothing of input 2D data array.

        Parameters
        ----------
        data_array : 1D array-like or 2D array-like
            1D data array or 2D data array of 1D series data.

        Returns
        -------
        numpy.ndarray
            Resulting smoothed data array.

        Examples
        --------
        >>> lp = LocalPolynomial(5, polynomial_order=2)
        >>> lp.lowess_filter([1, 2, 3, 4, 5, 6, 77, 88, 9, 10])
        >>> lp.lowess_filter([[1, 2, 3, 4, 5, 6, 77, 88, 9, 10], [1, 22, 33, 4, 5, 6, 7, 8, 9, 10]])

        Add to prepared ``SpecPipe`` instance ``pipe`` for ROI pixel spectrum processing::

            >>> pipe.add_process(6, 6, 0, LocalPolynomial(5, polynomial_order=2).lowess_filter)

        Add to prepared ``SpecPipe`` instance ``pipe`` for the processing of 1D sample data::

            >>> pipe.add_process(7, 7, 0, LocalPolynomial(5, polynomial_order=2).lowess_filter)
        """
        data_array = _to_2d_arr(data_array).astype(self.numtype)

        # Apply kernel
        if self.outlier_replacer is None:
            result = super()._apply(data_array, self._lowess_kernel)
        else:
            result = super()._chain_apply(
                data_array=data_array, function_list=[self.outlier_replacer.replace, self._lowess_kernel]
            )

        return _back_1d_arr(result)
