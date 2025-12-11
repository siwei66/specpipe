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

    axis : int, optional
        Window is rolling along the assigned axis. The default is 1.

    mode : str, optional
        Rolling mode for kernel application.
        'center' - window is created centered on current data point, better applicable to static data series.
        'end' - for time series, window is created end with current data point, better applicable to time series.
        'knn' - k-nearst neighbor mode, k is the window size.
                For 'knn', the window is constructed with available data centered on current data point, except the edge windows.
                For the edge windows, the window is constructed on the k nearst neighbor values.
                'knn' cannot be applied to 'savitzky_golay_filter'.
        The default is 'center'.

    padding : str, optional
        Padding approach for rolling mode "center" and "end", choose between:
            'none' - No padding applied. For rolling mode "center" and "end", the resulting data series will be window_size - 1 smaller than the original data series without padding.
            'nan' - Missing values are set to numpy nan after applying the function. 'nan' cannot be applied to 'savitzky_golay_filter'.
            'ext_edge' - Edge value of the first and last window are applied to pad before applying the smoothing function.
            'constant_edge' - Missing values are filled with edge value after applying function.
            'extrapolation' - Missing values are linearly extrapolated from edge values with length of window_size/2 (center mode), window_size-1 (end mode) after function application.
        The default is 'ext_edge'.

    numtype : str, optional
        Number type of given data, supported number type of Numpy. Default is float32.

    outlier_replacer : ArrayOutlier object, optional
        Outlier removing object with defined attributes. The default is None.

    polynomial_order : int
        Order of used polynomial regression.

    wfunc_x : str or function
        Weight distribution function for LOWESS filter at x axis. The weight function can be a distribution name or a custom function.
        Available weight distribution names:
            'tricubic' / 'triangular' / 'cosine' / 'gaussian' / 'epanechnikov' / 'exponential' / 'uniform'
        The default is 'tricubic'.
        Configure distribution feature using shape parameter wdist_param_x.

    wdist_param_x : float
        Shape parameter of weight distribution at x axis for LOWESS filter.

    wfunc_y : str or function
        Weight distribution function for LOWESS filter at y axis. The weight function can be a distribution name or a custom function.
        Available weight distribution names:
            'tricubic' / 'triangular' / 'cosine' / 'gaussian' / 'epanechnikov' / 'exponential' / 'uniform'
        The default is 'uniform'.
        Configure distribution feature using shape parameter wdist_param_y.

    wdist_param_y : float
        Shape parameter of weight distribution at y axis for LOWESS filter.

    axis_double_definition_warning : bool, optional
        If True, the duplicate definition warning will prompt when function has 'axis' argument. The default is True.
        Set false for known application.


    Methods
    -------
    savitzky_golay_filter
        Implemente Savitzky-Golay smoothing of input 2D data array.

    simple_polynomial_filter
        Implemente simple polynomial smoothing of input 2D data array.

    lowess_filter
        Implemente LOWESS smoothing of input 2D data array.
    """  # noqa: E501

    @simple_type_validator
    def __init__(
        self,
        window_size: int,
        axis: int = 1,
        mode: str = 'center',
        padding: str = 'ext_edge',
        numtype: str = 'float32',
        outlier_replacer: Optional[ArrayOutlier] = None,
        *,
        polynomial_order: int,
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
    def savitzky_golay_kernel(self, window_array: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Savitzky-Golay kernel.

        Parameters
        ----------
        window_array : np.ndarray
            2D window data of 1D data series.

        Returns
        -------
        window_array
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
        data_array : 1D or 2D arraylike
            1D data array or 2D data array of 1D series data.

        Returns
        -------
        np.ndarray
            Resulting smoothed data array.
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
            result = super().apply(data_array, self.savitzky_golay_kernel)
        else:
            result = super().chain_apply(
                data_array=data_array, function_list=[self.outlier_replacer.replace, self.savitzky_golay_kernel]
            )

        return _back_1d_arr(result)

    # Polynomial regression on 1-d series data, return estimation of focal point
    @simple_type_validator
    def polynomial_estimator(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Perform polynomial regression on local window data.

        Parameters
        ----------
        y : np.ndarray
            Responses.
        x : np.ndarray
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
    def simple_polynomial_kernel(self, window_array: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Polynomial kernel.

        Parameters
        ----------
        window_array : np.ndarray
            2D window data of 1D data series.

        Returns
        -------
        window_array
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
            if self.window_ind < self.focal_ind:  # start
                self.focal_ind = self.window_ind
            elif self.window_ind > self.data_array_shape[0] - window_size + self.focal_ind:  # end
                self.focal_ind = self.window_ind + window_size - self.data_array_shape[0]
        elif self.roll_mode == 'center':
            self.focal_ind = self.focal_ind0
        elif self.roll_mode == 'end':
            self.focal_ind = self.focal_ind1
        # Regression by column
        x = np.linspace(0, window_size - 1, window_size)
        y_pred_list = []
        for i in range(window_array.shape[1]):
            y_pred_list.append(self.polynomial_estimator(window_array[:, i], x))
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
        data_array : 1D or 2D arraylike
            1D data array or 2D data array of 1D series data.

        Returns
        -------
        np.ndarray
            Resulting smoothed data array.
        """
        data_array = _to_2d_arr(data_array).astype(self.numtype)

        # Apply kernel
        if self.outlier_replacer is None:
            result = super().apply(data_array, self.simple_polynomial_kernel)
        else:
            result = super().chain_apply(
                data_array=data_array, function_list=[self.outlier_replacer.replace, self.simple_polynomial_kernel]
            )

        return _back_1d_arr(result)

    # weight function for lowess
    @staticmethod
    @simple_type_validator
    def tricubic(sample_distances_to_focal: Annotated[np.ndarray, arraylike_validator(ndim=1)]) -> np.ndarray:
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
    def triangular(
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
    def cosine(
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
    def gaussian(
        sample_distances_to_focal: Annotated[np.ndarray, arraylike_validator(ndim=1)], sigma: float = -1.0
    ) -> np.ndarray:
        """Gaussian kernel function mapping distances from focus to weights."""
        fdist = sample_distances_to_focal
        if sigma == -1:
            sigma = 1.0
        if sigma <= 0:
            raise ValueError('Kernel parameter sigma must be positive')
        sample_weights = np.exp(-(fdist**2) / (2 * sigma**2))
        return np.array(sample_weights)

    @staticmethod
    @simple_type_validator
    def epanechnikov(
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
    def exponential(
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
    def lowess_estimator(self, y: np.ndarray, x: np.ndarray) -> float:  # noqa: C901
        """
        Perform locally weighted regression.

        Parameters
        ----------
        y : np.ndarray
            Responses.
        x : np.ndarray
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
                    self.sample_weight_x = self.tricubic(fdist_x)
                elif xweight == 'triangular':
                    self.sample_weight_x = self.triangular(fdist_x, self.wdist_param_x)
                elif xweight == 'cosine':
                    self.sample_weight_x = self.cosine(fdist_x, self.wdist_param_x)
                elif xweight == 'gaussian':
                    self.sample_weight_x = self.gaussian(fdist_x, self.wdist_param_x)
                elif xweight == 'epanechnikov':
                    self.sample_weight_x = self.epanechnikov(fdist_x, self.wdist_param_x)
                elif xweight == 'exponential':
                    self.sample_weight_x = self.exponential(fdist_x, self.wdist_param_x)
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
                    self.sample_weight_y = self.tricubic(fdist_y)
                elif yweight == 'triangular':
                    self.sample_weight_y = self.triangular(fdist_y, self.wdist_param_y)
                elif yweight == 'cosine':
                    self.sample_weight_y = self.cosine(fdist_y, self.wdist_param_y)
                elif yweight == 'gaussian':
                    self.sample_weight_y = self.gaussian(fdist_y, self.wdist_param_y)
                elif yweight == 'epanechnikov':
                    self.sample_weight_y = self.epanechnikov(fdist_y, self.wdist_param_y)
                elif yweight == 'exponential':
                    self.sample_weight_y = self.exponential(fdist_y, self.wdist_param_y)
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
    def lowess_kernel(self, window_array: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Locally weighted scatterplot smoothing kernel.

        Parameters
        ----------
        window_array : np.ndarray
            Window 2D array of 1D series data.

        Returns
        -------
        np.ndarray
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
            if self.window_ind < self.focal_ind:  # start
                self.focal_ind = self.window_ind
            elif self.window_ind > self.data_array_shape[0] - window_size + self.focal_ind:  # end
                self.focal_ind = self.window_ind + window_size - self.data_array_shape[0]
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
            y_pred_list.append(self.lowess_estimator(window_array[:, i], x))
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
        data_array : 1D or 2D arraylike
            1D data array or 2D data array of 1D series data.

        Returns
        -------
        np.ndarray
            Resulting smoothed data array.
        """
        data_array = _to_2d_arr(data_array).astype(self.numtype)

        # Apply kernel
        if self.outlier_replacer is None:
            result = super().apply(data_array, self.lowess_kernel)
        else:
            result = super().chain_apply(
                data_array=data_array, function_list=[self.outlier_replacer.replace, self.lowess_kernel]
            )

        return _back_1d_arr(result)
