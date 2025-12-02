# -*- coding: utf-8 -*-
"""
Tests for 1D data series denoising - kernel smoothing - local polynomial smoothing

Copyright (c) 2025 Siwei Luo. MIT License.
"""
import numpy as np
import pytest
import sys

from specpipe.denoiser.utils import create_test_array
from specpipe.denoiser.outlier import ArrayOutlier
from specpipe.denoiser.local_polynomial import LocalPolynomial


# %% Test LocalPolynomial


class TestLocalPolynomial:
    """Test LocalPolynomial functionalities."""

    @staticmethod
    def test_savitzky_golay_filter() -> None:
        """Test savitzky_golay_filter functionality."""
        arr = create_test_array()

        # Test basic
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1)
        res = denoiser.savitzky_golay_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape

        # Test polynomial order
        denoiser = LocalPolynomial(window_size=5, polynomial_order=1, axis=1)
        res = denoiser.savitzky_golay_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape

        # Test axis
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=0)
        res = denoiser.savitzky_golay_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape

        # Test mode
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, mode='end')
        res = denoiser.savitzky_golay_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape

        # Test other padding approaches
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, padding='none')
        res = denoiser.savitzky_golay_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == (14, 8)
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, padding='constant_edge')
        res = denoiser.savitzky_golay_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, padding='extrapolation')
        res = denoiser.savitzky_golay_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape

    @staticmethod
    def test_savitzky_golay_filter_edge() -> None:
        """Test savitzky_golay_filter edge cases."""
        arr = create_test_array()

        # Window size
        with pytest.raises(ValueError, match="window_size must be an odd integer"):
            denoiser = LocalPolynomial(window_size=4, polynomial_order=2, axis=1)
            _ = denoiser.savitzky_golay_filter(arr)

        # Roll mode
        with pytest.raises(ValueError, match="Mode 'knn' is not applicable"):
            denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, mode='knn')
            _ = denoiser.savitzky_golay_filter(arr)

        # Paddimg approach
        with pytest.raises(ValueError, match="'nan' padding is not applicable"):
            denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, padding='nan')
            _ = denoiser.savitzky_golay_filter(arr)

    @staticmethod
    def test_simple_polynomial_filter() -> None:
        """Test simple_polynomial_filter functionality."""
        arr = create_test_array()

        # Test basic
        denoiser = LocalPolynomial(window_size=4, polynomial_order=2, axis=1)
        res = denoiser.simple_polynomial_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape

        # Test polynomial order
        denoiser = LocalPolynomial(window_size=5, polynomial_order=1, axis=1)
        res = denoiser.simple_polynomial_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        denoiser = LocalPolynomial(window_size=5, polynomial_order=3, axis=1)
        res = denoiser.simple_polynomial_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape

        # Test axis
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=0)
        res = denoiser.simple_polynomial_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape

    @staticmethod
    def test_lowess_filter() -> None:
        """Test lowess_filter functionality."""
        arr = create_test_array()

        # Test basic
        denoiser = LocalPolynomial(window_size=4, polynomial_order=2, axis=1)
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape

        # Test polynomial order
        denoiser = LocalPolynomial(window_size=5, polynomial_order=1, axis=1)
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        denoiser = LocalPolynomial(window_size=5, polynomial_order=3, axis=1)
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape

        # Test axis
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=0)
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape

    @staticmethod
    def test_lowess_filter_wfunc_x() -> None:
        """Test lowess_filter weight function at x axis."""
        arr = create_test_array()

        # Test x weight functions
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, wfunc_x='tricubic')
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, wfunc_x='triangular')
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, wfunc_x='cosine')
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, wfunc_x='gaussian')
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, wfunc_x='epanechnikov')
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, wfunc_x='exponential')
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, wfunc_x='uniform')
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape

    @staticmethod
    def test_lowess_filter_wfunc_y() -> None:
        """Test lowess_filter weight function at y axis."""
        arr = create_test_array()

        # Test y weight functions
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, wfunc_y='tricubic')
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, wfunc_y='triangular')
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, wfunc_y='cosine')
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, wfunc_y='gaussian')
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, wfunc_y='epanechnikov')
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, wfunc_y='exponential')
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, wfunc_y='uniform')
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape

        # Test ties (all equal smooth data)
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=0, wfunc_y='tricubic')
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape

    @staticmethod
    def test_local_polynomial_filters_with_outlier_removal() -> None:
        """Test local polynomial filters with window outlier removal."""
        arr = create_test_array()

        outlier_remover = ArrayOutlier('dixon', to='median', generate_report=False)
        denoiser = LocalPolynomial(window_size=5, polynomial_order=2, axis=1, outlier_replacer=outlier_remover)
        # Test savitzky_golay_filter
        res = denoiser.savitzky_golay_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        # Test simple_polynomial_filter
        res = denoiser.simple_polynomial_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        # Test lowess_filter
        res = denoiser.lowess_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape


# %% Test main

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
