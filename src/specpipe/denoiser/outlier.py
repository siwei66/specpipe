# -*- coding: utf-8 -*-
"""
Small 1D data outlier replacing tool for denoisers

Copyright (c) 2025 Siwei Luo. MIT License.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Union, Optional, Any, Annotated
from scipy.stats import median_abs_deviation

from ..specio import simple_type_validator, arraylike_validator, RealNumber

# # For local test
# from specpipe.specio import simple_type_validator, arraylike_validator, RealNumber


# %% Outlier removal


@simple_type_validator
def replace_outlier(
    data: Union[Annotated[Any, arraylike_validator(ndim=2)], pd.DataFrame],
    test_method: str = 'iqr',
    to: str = 'neighbor',
    axis: int = 0,
    *,
    dixon_alpha: float = 0.05,
    iqr_multiplier: float = 1.5,
    modified_z_threshold: float = 3.5,
    numtype: Union[str, type] = 'float32',
    generate_report: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, list]]:
    """
    Replace outliers in a 2D array or dataframe of 1D data.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame
        2D array or dataframe of 1D data.
    test_method : str
        The method of outlier test. See ``ArrayOutlier`` for details.
    to : str
        The outlier replacement strategy. See ``ArrayOutlier`` for details.
    axis : int
        Calculate along the axis. The default is 0.
    dixon_alpha : float
        Two-tail significance level for Dixon's Q test, the default is 0.05.
    iqr_multiplier : float
        Multiplier applied to the interquartile range (IQR) to define the lower and upper bounds for outlier detection.
        The default is 1.5.
    modified_z_threshold : float
        Threshold value used in modified z-score–based outlier detection.
        Observations with an absolute modified z-score exceeding this value are classified as outliers.
        The default is 3.5.
    numtype : str
        Numpy-supported numeric data type for test computation and output.
        Default is ``"float32"``.
    generate_report : bool
        Whether to generate reports of outlier tests.
        If True, the generation can be time-consuming for large datasets.
        The default is False.

    Returns
    -------
    numpy.ndarray or tuple of (numpy.ndarray, list)
        Data with outlier replaced or data with outlier replaced and outlier detection reports.

    See Also
    --------
    ArrayOutlier
    ArrayOutlier.replace

    Examples
    --------
    Basic usage for outlier replacement using default settings::

        >>> replace_outlier([[1, 2, 3, 99, 5, 6], [2, 2, 4, 4, 6, 6]])

    Specify outlier detection method::

        >>> replace_outlier([[1, 2, 3, 99, 5, 6], [2, 2, 4, 4, 6, 6]], test_method='dixon')

    Customize outlier detection method::

        >>> replace_outlier([[1, 2, 3, 99, 5, 6], [2, 2, 4, 4, 6, 6]], test_method='dixon', dixon_alpha=0.1)

    Specify replacement strategy::

        >>> replace_outlier([[1, 2, 3, 99, 5, 6], [2, 2, 4, 4, 6, 6]], to='median')

    Retrieve report in addition to result of replacement::

        >>> result, report = replace_outlier([[1, 2, 3, 99, 5, 6], [2, 2, 4, 4, 6, 6]], generate_report=True)
    """  # noqa: E501
    data = np.array(data)
    if not generate_report:
        result = ArrayOutlier(
            test_method=test_method,
            to=to,
            axis=axis,
            dixon_alpha=dixon_alpha,
            iqr_multiplier=iqr_multiplier,
            modified_z_threshold=modified_z_threshold,
            numtype=numtype,
            generate_report=False,
        ).replace(data=data)
        assert isinstance(result, np.ndarray)
        return result
    else:
        outlier_remover = ArrayOutlier(
            test_method=test_method,
            to=to,
            axis=axis,
            dixon_alpha=dixon_alpha,
            iqr_multiplier=iqr_multiplier,
            modified_z_threshold=modified_z_threshold,
            numtype=numtype,
            generate_report=True,
        )
        result = outlier_remover.replace(data=data)
        assert isinstance(result, np.ndarray)
        assert isinstance(outlier_remover.report[0], list)
        return result, outlier_remover.report[0]


# Current progress only covers methods for small sample size.
class ArrayOutlier:
    """
    Identify and replace outliers in 1D data lines in the dataframe or 2D array.

    Attributes
    ----------
    test_method : str
        The method of outlier test. Available options:

        - "dixon" - Dixon's Q test,
        - "iqr" - interquartile range,
        - "modified_z" - Modified Z-score.

        The default is "iqr".

    to : str
        The outlier replacement strategy. The outlier can be replaced by:

        - "nan" - the outlier is removed and not calculated.
        - "mean" - mean value of nonoutliers.
        - "median" - median of nonoutliers.
        - "neighbor" - the closest nonoutlier value of the outlier. If two are availble, average of the two neighbors are used.

        The default is "neighbor".

    axis : int
        Calculate along the axis. The default is 0.

    dixon_alpha : float
        Two-tail significance level for Dixon's Q test, the default is 0.05.

    iqr_multiplier : float, optional
        Multiplier applied to the interquartile range (IQR) to define the lower and upper bounds for outlier detection.

        The default is 1.5.

    modified_z_threshold : float, optional
        Threshold value used in modified z-score–based outlier detection.
        Observations with an absolute modified z-score exceeding this value are classified as outliers.

        The default is 3.5.

    numtype : str
        Numpy-supported numeric data type for test computation and output, default is "float32".

    generate_report : bool
        Whether to generate reports of outlier tests.

        The generation can be time-consuming for large datasets.
        Repeated calls to ArrayOutlier.replace() accumulate reports in the ArrayOutlier.report, which can lead to significant memory growth.

        The default is False.

    report : list or list
        List of reports of each "replace" exection if generate_report is True.

    Methods
    -------
    dixon_q
        Apply Dixon's Q test to get outlier and nonoutlier indices of 1D data series.
    iqr
        Apply IQR approach to get outlier and nonoutlier indices of 1D data series.
    modified_z
        Apply modified z score approach to get outlier and nonoutlier indices of 1D data series.
    replace
        Replace detected outliers.

    Examples
    --------
    Use default settings::

        >>> outlier = ArrayOutlier()

    Specify outlier detection method::

        >>> outlier = ArrayOutlier(test_method='dixon')

    Customize outlier detection method::

        >>> outlier = ArrayOutlier(test_method='dixon', dixon_alpha=0.1)

    Specify replacement strategy::

        >>> outlier = ArrayOutlier(to='median')

    Retrieve report in addition to result of replacement::

        >>> outlier = ArrayOutlier(generate_report=True)
        >>> report = outlier.report
    """  # noqa: E501

    @simple_type_validator
    def __init__(
        self,
        test_method: str = 'iqr',
        to: str = 'neighbor',
        axis: int = 0,
        *,
        dixon_alpha: float = 0.05,
        iqr_multiplier: float = 1.5,
        modified_z_threshold: float = 3.5,
        numtype: Union[str, type] = 'float32',
        generate_report: bool = False,
    ) -> None:
        # Validate test method
        test_method = str(test_method).lower()
        if test_method not in ['dixon', 'iqr', 'modified_z']:
            raise ValueError(f"'test_method' must be one of 'dixon' / 'iqr' and 'modified_z', got: {test_method}")
        self.test_method: str = test_method

        # Validate replace strategy
        to = str(to).lower()
        if to not in ['nan', 'mean', 'median', 'neighbor']:
            raise ValueError(f"'to' must be one of 'nan' / 'mean' / 'median' and 'neighbor', got: {to}")
        self.to: str = to

        # Validate axis
        if (axis != 0) & (axis != 1):
            raise ValueError("Axis must be 0 or 1.")
        self.axis: int = axis

        self.dixon_alpha: float = dixon_alpha
        self.iqr_multiplier: float = iqr_multiplier
        self.modified_z_threshold: float = modified_z_threshold
        self.numtype: Union[str, type] = numtype
        self.generate_report: bool = generate_report

        # Internel attr
        self.report: list = []

    # Dixon's Q Test critical values
    @simple_type_validator
    def _dixon_critical(self, sample_size: int, alpha: Optional[float] = None) -> float:
        """
        Calculate the critical value for Dixon's Q test.
        The critical values are defined according to:
            D. B. Rorabacher,
            "Statistical treatment for rejection of deviant values: critical values of Dixon’s 'Q' parameter and related subrange ratios at the 95% confidence level"
            Anal. Chem., vol. 63, no. 2, pp. 139–146,
            Jan. 1991,
            DOI: 10.1021/ac00002a010.

        Parameters
        ----------
        sample_size : int
            Sample_number.
        alpha : float
            Two tailed significance level. If None, uses the attribute dixon_alpha.

        Returns
        -------
        Q_critical : float
            Critical value for Dixon's Q test.

        Raises
        ------
        ValueError
            If `sample_size` beyond range 3~30.
        ValueError
            If two-tail significance level `alpha` beyond range 0.005~0.1.
        """  # noqa: E501
        n = sample_size
        if alpha is None:
            alpha = self.dixon_alpha
        # Convert 2-tailed alpha to general significance = 1 - confidence level
        alpha_g = 2 * alpha
        if (n < 3) | (n > 30):
            raise ValueError(f"Sample size for Dixon's Q test must be between 3 and 30, got: {n}")
        elif (alpha_g < 0.01) | (alpha_g > 0.2):
            raise ValueError(f"Dixon's Q test significance level must be between 0.005 and 0.1, got: {alpha}")
        else:
            dixon_critical_table = np.array(
                [
                    [0.886, 0.941, 0.97, 0.976, 0.988, 0.994],
                    [0.679, 0.765, 0.829, 0.846, 0.889, 0.926],
                    [0.557, 0.642, 0.71, 0.729, 0.78, 0.821],
                    [0.482, 0.56, 0.625, 0.644, 0.698, 0.74],
                    [0.434, 0.507, 0.568, 0.586, 0.637, 0.68],
                    [0.399, 0.468, 0.526, 0.543, 0.59, 0.634],
                    [0.37, 0.437, 0.493, 0.51, 0.555, 0.598],
                    [0.349, 0.412, 0.466, 0.483, 0.527, 0.568],
                    [0.332, 0.392, 0.444, 0.46, 0.502, 0.542],
                    [0.318, 0.376, 0.426, 0.441, 0.482, 0.522],
                    [0.305, 0.361, 0.41, 0.425, 0.465, 0.503],
                    [0.294, 0.349, 0.396, 0.411, 0.45, 0.488],
                    [0.285, 0.338, 0.384, 0.399, 0.438, 0.475],
                    [0.277, 0.329, 0.374, 0.388, 0.426, 0.463],
                    [0.269, 0.32, 0.365, 0.379, 0.416, 0.452],
                    [0.263, 0.313, 0.356, 0.37, 0.407, 0.442],
                    [0.258, 0.306, 0.349, 0.363, 0.398, 0.433],
                    [0.252, 0.3, 0.342, 0.356, 0.391, 0.425],
                    [0.247, 0.295, 0.337, 0.35, 0.384, 0.418],
                    [0.242, 0.29, 0.331, 0.344, 0.378, 0.411],
                    [0.238, 0.285, 0.326, 0.338, 0.372, 0.404],
                    [0.234, 0.281, 0.321, 0.333, 0.367, 0.399],
                    [0.23, 0.277, 0.317, 0.329, 0.362, 0.393],
                    [0.227, 0.273, 0.312, 0.324, 0.357, 0.388],
                    [0.224, 0.269, 0.308, 0.32, 0.353, 0.384],
                    [0.22, 0.266, 0.305, 0.316, 0.349, 0.38],
                    [0.218, 0.263, 0.301, 0.312, 0.345, 0.376],
                    [0.215, 0.26, 0.298, 0.309, 0.341, 0.372],
                ]
            ).astype(self.numtype)
            slevel = [0.2, 0.1, 0.05, 0.04, 0.02, 0.01]
            # Linear interpolation
            for i in range(6):
                if alpha_g == slevel[i]:
                    Q_critical = dixon_critical_table[n - 3, i]  # noqa: N806
                    return float(Q_critical)
                elif (alpha_g < slevel[i]) & (alpha_g > slevel[i + 1]) & (i < 5):
                    critr = dixon_critical_table[n - 3, :]
                    Q_critical = critr[i] + (critr[i + 1] - critr[i]) / (slevel[i + 1] - slevel[i]) * (  # noqa: N806
                        alpha_g - slevel[i]
                    )
                    return float(np.round(Q_critical, 3))
            raise ValueError(f"Significance level beyond definition, must be in range [0.01, 0.2], got: {alpha_g}")

    # Dixon's Q Test
    @simple_type_validator
    def dixon(
        self, data_series: Union[list, Annotated[Any, arraylike_validator(ndim=1)]]
    ) -> tuple[np.ndarray, np.ndarray, Optional[list]]:
        """
        Perform Dixon's Q test to identify outliers in a dataset.

        Parameters
        ----------
        data_series : list or 1D array-like
            Series of data for outlier detection. Dixon's Q Test requires a sample size between 3~30.

        Returns
        -------
        A tuple of:

            outlier_indices : numpy.ndarray
                Numpy arrays of outlier indices.
            nonoutlier_indices : numpy.ndarray
                Numpy arrays of non-outlier indices.
            test_report : list or None
                List of test report if generated.

        Raises
        ------
        ValueError
            If sample size beyond range 3~30.

        Examples
        --------
        >>> outlier = ArrayOutlier()
        >>> outlier_ind, non_outlier_ind, report = outlier.dixon([1, 2, 3, 99, 5, 6])
        """
        data_series = np.array(data_series).astype(self.numtype)

        generate_report = self.generate_report
        alpha = self.dixon_alpha

        # Sample size check
        arrlen = len(data_series)
        if (arrlen < 3) | (arrlen > 30):
            raise ValueError("Sample size for Dixon's Q test must be between 3 and 30.")

        n = len(data_series)
        indices = np.arange(n)

        # Calculate the critical value
        Q_critical = self._dixon_critical(n, alpha)  # noqa: N806

        # Sort the data and keep track of the original indices
        sorted_indices = np.argsort(data_series)
        data_sorted = data_series[sorted_indices]

        # Calculate Q statistic for the smallest and largest values
        Q_denominator = data_sorted[-1] - data_sorted[0]  # noqa: N806

        if Q_denominator != 0:
            Q_min = (data_sorted[1] - data_sorted[0]) / Q_denominator  # noqa: N806
            Q_max = (data_sorted[-1] - data_sorted[-2]) / Q_denominator  # noqa: N806
        else:
            Q_min, Q_max = 0, 0  # noqa: N806

        # Determine outliers and their indices
        outlier_indices_list = []
        if Q_min > Q_critical:
            outlier_indices_list.append(sorted_indices[0])
        if Q_max > Q_critical:
            outlier_indices_list.append(sorted_indices[-1])
        outlier_indices = np.array(outlier_indices_list).astype('int16')

        if len(outlier_indices) > 0:
            # Non-outlier indices
            nonoutlier_indices = indices[indices != outlier_indices[0]]
        else:
            nonoutlier_indices = indices
        assert isinstance(nonoutlier_indices, np.ndarray)
        nonoutlier_indices = np.array(nonoutlier_indices).astype('int16')

        if generate_report:
            test_report2 = []
            test_report2.append(['sample size: ', n])
            test_report2.append(['Q critical value: ', Q_critical])
            test_report2.append(['Q_min: ', Q_min])
            test_report2.append(['Q_max: ', Q_max])
            test_report2.append(['Outlier indices: ', list(outlier_indices)])
            if len(outlier_indices) > 0:
                test_report2.append(['Outliers: ', list(data_series[outlier_indices])])
            else:
                test_report2.append(['Outliers: ', 'NA'])

        if generate_report:
            return outlier_indices, nonoutlier_indices, test_report2
        else:
            return outlier_indices, nonoutlier_indices, None

    # Interquartile range criterion
    @simple_type_validator
    def iqr(
        self, data_series: Union[list, Annotated[Any, arraylike_validator(ndim=1)]]
    ) -> tuple[np.ndarray, np.ndarray, Optional[list]]:
        """
        Identify outliers using the Interquartile Range (IQR) criterion and return their indices.

        Parameters
        ----------
        data_series : list or 1D array-like
            List or 1D array of a data series for outlier detection. The length must be at least 5.

        Returns
        -------
        A tuple of:

            outlier_indices : numpy.ndarray
                Numpy arrays of outlier indices.
            nonoutlier_indices : numpy.ndarray
                Numpy arrays of non-outlier indices.
            test_report : list or None
                List of test report if generated.

        Raises
        ------
        ValueError
            If sample size < 5.

        Examples
        --------
        >>> outlier = ArrayOutlier()
        >>> outlier_ind, non_outlier_ind, report = outlier.iqr([1, 2, 3, 99, 5, 6])
        """
        generate_report = self.generate_report
        iqr_multiplier = self.iqr_multiplier

        # Convert data to a numpy array
        data_series = np.array(data_series).astype(self.numtype)

        # Sample size check
        n = data_series.shape[0]
        if n < 5:
            raise ValueError("Sample size to compute interquartile range (IQR) must be at least 5.")

        # Calculate the first quartile (Q1) and third quartile (Q3)
        Q1 = np.percentile(data_series, 25)  # noqa: N806
        Q3 = np.percentile(data_series, 75)  # noqa: N806

        # Calculate the Interquartile Range (IQR)
        IQR = Q3 - Q1  # noqa: N806

        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        # Identify indices
        outlier_indices = np.where((data_series <= lower_bound) | (data_series >= upper_bound))[0]
        nonoutlier_indices = np.where((data_series >= lower_bound) & (data_series <= upper_bound))[0]

        if generate_report:
            test_report2 = []
            test_report2.append(['sample size: ', n])
            test_report2.append(['IQR: ', IQR])
            test_report2.append(['Q1: ', Q1])
            test_report2.append(['Q3: ', Q3])
            test_report2.append(['lower_bound: ', lower_bound])
            test_report2.append(['upper_bound: ', upper_bound])
            test_report2.append(['Outlier indices: ', list(outlier_indices)])
            if len(outlier_indices) > 0:
                test_report2.append(['Outliers: ', list(data_series[outlier_indices])])
            else:
                test_report2.append(['Outliers: ', 'NA'])

        if generate_report:
            return outlier_indices, nonoutlier_indices, test_report2
        else:
            return outlier_indices, nonoutlier_indices, None

    # Modified Z-score approach
    @simple_type_validator
    def modified_z(
        self, data_series: Union[list, Annotated[Any, arraylike_validator(ndim=1)]]
    ) -> tuple[np.ndarray, np.ndarray, Optional[list]]:
        """
        Identify outliers using the modified z score approach and return their indices.

        Parameters
        ----------
        data_series : 1D list or numpy array
            The data series to test for outliers. The lenth should be at least 12.

            Please be aware that the function does not check for data normality that is required by the approach.

        Returns
        -------
        A tuple of:

            outlier_indices : numpy.ndarray
                Numpy arrays of outlier indices.
            nonoutlier_indices : numpy.ndarray
                Numpy arrays of non-outlier indices.
            test_report : list or None
                List of test report if generated.

        Raises
        ------
        ValueError
            If sample size < 5.

        Warnings
        --------
        UserWarning
            If sample size >= 5 but < 12. Applicable but result may not be reliable due to normality identification.

        Examples
        --------
        >>> outlier = ArrayOutlier()
        >>> outlier_ind, non_outlier_ind, report = outlier.modified_z([1, 2, 3, 99, 5, 6])
        """
        generate_report = self.generate_report
        threshold = self.modified_z_threshold

        # Convert data to a numpy array
        data_series = np.array(data_series).astype(self.numtype)

        # Sample size check
        n = data_series.shape[0]
        if n < 12:
            if n < 5:
                raise ValueError(f"Modified z test requires a minimum sample size of 5, got: {n}")
            warnings.warn(
                'Modified Z-score test requires minimum 12 samples for reliable results.\n\n', UserWarning, stacklevel=2
            )

        # Calculate the median and Median Absolute Deviation (MAD)
        median1 = np.median(data_series)
        mad = median_abs_deviation(data_series, scale='normal')

        # Avoid division by zero
        if mad == 0:
            mad = 1e-10  # Small value to prevent division by zero

        # Calculate the Modified Z-Score
        modified_z_scores = 0.6745 * (data_series - median1) / mad

        # Identify indices
        outlier_indices = np.where(np.abs(modified_z_scores) >= threshold)[0]
        nonoutlier_indices = np.where(np.abs(modified_z_scores) <= threshold)[0]

        if generate_report:
            test_report2 = []
            test_report2.append(['Sample size: ', n])
            test_report2.append(['Median: ', median1])
            test_report2.append(['Median absolute deviation: ', mad])
            test_report2.append(['Threshold: ', threshold])
            test_report2.append(['Modified z scores: ', list(modified_z_scores)])
            test_report2.append(['Outlier indices: ', list(outlier_indices)])
            if len(outlier_indices) > 0:
                test_report2.append(['Outliers: ', list(data_series[outlier_indices])])
            else:
                test_report2.append(['Outliers: ', 'NA'])

        if generate_report:
            return outlier_indices, nonoutlier_indices, test_report2
        else:
            return outlier_indices, nonoutlier_indices, None

    # Add non-normal method of SD and MADe
    # 2D array outlier calculation
    @simple_type_validator
    def replace(  # noqa: C901
        self,
        data: Union[Annotated[Any, arraylike_validator(ndim=2)], pd.DataFrame],
    ) -> np.ndarray:
        """
        Replace outliers in a 2D array or dataframe of 1D data series.

        Parameters
        ----------
        data : numpy.ndarray or pandas.DataFrame
            2D array or dataframe of 1D data.

        Returns
        -------
        numpy.ndarray
            Data with outlier replaced.

        Raises
        ------
        ValueError
            If input data is not 2D numpy array or pandas dataframe.
        ValueError
            Unknown absence of replace value in outlier replacement.

        Examples
        --------
        >>> outlier = ArrayOutlier()
        >>> outlier.replace([[1, 2, 3, 99, 5, 6], [2, 2, 4, 4, 6, 6]])
        """  # noqa: E501
        generate_report = self.generate_report
        test_method = self.test_method
        to = self.to
        axis = self.axis

        # Validate axis
        if (axis != 0) & (axis != 1):
            raise ValueError("Axis must be 0 or 1.")

        if generate_report:
            test_report = []

        # Dimension check
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D numpy array!")

        # Convert data to a numpy array and transpose if needed
        if axis == 0:
            data = np.array(data).astype(self.numtype)
        elif axis == 1:
            data = np.array(data).astype(self.numtype).T

        for i in range(data.shape[1]):

            data_series = np.array(data[:, i])

            if generate_report:
                test_report1 = []
                test_report1.append(['series number: ', i])

            if test_method == 'dixon':
                test_result = self.dixon(data_series)
            elif test_method == 'iqr':
                test_result = self.iqr(data_series)
            elif test_method == 'modified_z':
                test_result = self.modified_z(data_series)

            outlier_indices = test_result[0]
            nonoutlier_indices = test_result[1]

            if generate_report:
                test_report1 = test_report1 + test_result[2]

            # Determine if outlier exist
            if len(outlier_indices) > 0:

                # Replacement
                rp: Union[RealNumber, list]
                if to == 'nan':
                    rp = np.nan
                    data_series[outlier_indices[0]] = rp
                elif to == 'mean':
                    rp = np.nanmean(data_series[nonoutlier_indices])
                    data_series[outlier_indices[0]] = rp
                elif to == 'median':
                    rp = np.nanmedian(data_series[nonoutlier_indices])
                    data_series[outlier_indices[0]] = rp
                elif to == 'neighbor':
                    indit = np.array(outlier_indices[0]).astype('int')
                    if indit.ndim == 0:
                        indit = np.array([indit])
                    if generate_report:
                        rp = []
                    for j in indit:
                        subind1 = j + min(abs(nonoutlier_indices - j))
                        subind2 = j - min(abs(nonoutlier_indices - j))
                        if (subind1 in nonoutlier_indices) & (subind2 in nonoutlier_indices):
                            sub = (data_series[subind1] + data_series[subind2]) / 2
                        elif (subind1 in nonoutlier_indices) & (subind2 not in nonoutlier_indices):
                            sub = data_series[subind1]
                        elif (subind1 not in nonoutlier_indices) & (subind2 in nonoutlier_indices):
                            sub = data_series[subind2]
                        else:
                            raise ValueError(
                                f"Absence of replace value in outlier replacement, \
                                    value index: ({i}, {j})\n\
                                    outlier_indices: {outlier_indices}, \
                                    nonoutlier_indices: {nonoutlier_indices}, \
                                    replace_index1: {subind1}, \
                                    replace_index2: {subind2}\n\n"
                            )
                        data_series[j] = sub
                        if generate_report:
                            assert isinstance(rp, list)
                            rp.append(sub)
                if generate_report:
                    if isinstance(rp, RealNumber) and np.array(rp).ndim == 0:
                        test_report1.append(['Replace value: ', [rp]])
                    elif isinstance(rp, list) and np.array(rp).ndim == 1:
                        test_report1.append(['Replace value: ', list(rp)])
                    else:
                        raise ValueError(f"Invalid replace value, got: {rp}, type: {type(rp)}")
            elif generate_report:
                test_report1.append(['Replace value: ', 'No outlier detected'])

            # Append replaced results
            replaced_result = np.array(data_series).reshape(-1, 1).astype(self.numtype)
            if i == 0:
                result: np.ndarray = replaced_result
            else:
                result = np.append(result, replaced_result, axis=1)

            if generate_report:
                test_report.append(test_report1)

        if axis == 0:
            result = np.array(result)
        elif axis == 1:
            result = np.array(result).T

        if generate_report:
            self.report.append(test_report)

        return result
