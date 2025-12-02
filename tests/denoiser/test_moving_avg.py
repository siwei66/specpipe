# -*- coding: utf-8 -*-
"""
Tests for 1D data series denoising - kernel smoothing - moving average smoothing

Copyright (c) 2025 Siwei Luo. MIT License.
"""
import numpy as np

import pytest

import sys

from specpipe.denoiser.utils import create_test_array
from specpipe.denoiser.outlier import ArrayOutlier
from specpipe.denoiser.moving_avg import MovingAvg


# %% Test MovingAvg


class TestMovingAvg:
    """Test MovingAvg functionalities."""

    @staticmethod
    def test_simple_moving_average() -> None:
        """Test simple_moving_average functionality."""
        arr = create_test_array()
        denoiser = MovingAvg(window_size=5, axis=1)
        res = denoiser.simple_moving_average(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape

    @staticmethod
    def test_moving_median() -> None:
        """Test moving_median functionality."""
        arr = create_test_array()
        denoiser = MovingAvg(window_size=5, axis=1)
        res = denoiser.moving_median(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape

    @staticmethod
    def test_weighted_moving_average() -> None:
        """Test weighted_moving_average functionality."""
        arr = create_test_array()
        denoiser = MovingAvg(window_size=5, axis=1)
        res = denoiser.weighted_moving_average(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        # List weights
        denoiser1 = MovingAvg(window_size=5, axis=1, window_weights=[1, 2, 3, 2, 1])
        res1 = denoiser1.weighted_moving_average(arr)
        assert isinstance(res1, np.ndarray)
        assert res1.shape == arr.shape
        # Array weights
        denoiser2 = MovingAvg(window_size=5, axis=1, window_weights=np.array([1, 2, 3, 2, 1]))
        res2 = denoiser2.weighted_moving_average(arr)
        assert isinstance(res2, np.ndarray)
        assert res2.shape == arr.shape
        assert np.allclose(res1, res2)
        # String weights
        denoiser3 = MovingAvg(window_size=5, axis=1, window_weights='gaussian')
        res3 = denoiser3.weighted_moving_average(arr)
        assert isinstance(res3, np.ndarray)
        assert res3.shape == arr.shape
        with pytest.raises(ValueError, match="Invalid window_weights name"):
            _ = MovingAvg(window_size=5, axis=1, window_weights='siwei')

    @staticmethod
    def test_gaussian_filter() -> None:
        """Test gaussian_filter functionality."""
        arr = create_test_array()
        denoiser = MovingAvg(window_size=5, axis=1)
        res = denoiser.gaussian_filter(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape
        denoiser1 = MovingAvg(window_size=5, axis=1, gaussian_sigma=-0.5)
        res1 = denoiser1.gaussian_filter(arr)
        assert isinstance(res1, np.ndarray)
        assert res1.shape == arr.shape

    @staticmethod
    def test_with_outlier_removal() -> None:
        """Test moving average filter with outlier removal."""
        arr = create_test_array()
        outlier_remover = ArrayOutlier('dixon', to='median', generate_report=True)
        denoiser = MovingAvg(window_size=5, axis=1, outlier_replacer=outlier_remover)
        res1 = denoiser.simple_moving_average(arr)
        assert isinstance(res1, np.ndarray)
        assert res1.shape == arr.shape
        res2 = denoiser.moving_median(arr)
        assert isinstance(res2, np.ndarray)
        assert res2.shape == arr.shape
        res3 = denoiser.weighted_moving_average(arr)
        assert isinstance(res3, np.ndarray)
        assert res3.shape == arr.shape
        res4 = denoiser.gaussian_filter(arr)
        assert isinstance(res4, np.ndarray)
        assert res4.shape == arr.shape


# %% Test main

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
