# -*- coding: utf-8 -*-
"""
Tests for denoiser helpers / utilities

Copyright (c) 2025 Siwei Luo. MIT License.
"""
import numpy as np

import pytest
from typing import Annotated, Any

import sys

from specpipe.specio import simple_type_validator, arraylike_validator
from specpipe.denoiser.utils import create_test_array, RollWindow


# %% Working function for test


@simple_type_validator
def func_for_apply(arr: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
    return np.array(np.nanmean(arr, axis=0, keepdims=True))


@simple_type_validator
def func_for_apply1(arr: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
    return np.array(np.nanstd(arr, axis=0, keepdims=True))


# %% Test RollWindow


# Test RollWindow
class TestRollWindow:
    """Test RollWindow method functionalities"""

    @staticmethod
    def test_apply_axes() -> None:
        """Test different rolling axes."""
        arr = create_test_array(shape=(14, 12))
        # Axis 0
        res_roll_axis0 = RollWindow(5, axis=0)._apply(arr, func_for_apply)
        assert isinstance(res_roll_axis0, np.ndarray)
        assert res_roll_axis0.shape == (10, 12)
        # Axis 1
        res_roll_axis1 = RollWindow(5, axis=1)._apply(arr, func_for_apply)
        assert isinstance(res_roll_axis1, np.ndarray)
        assert res_roll_axis1.shape == (14, 8)

    @staticmethod
    def test_window_sizes() -> None:
        """Test different rolling axes."""
        arr = create_test_array(shape=(14, 12))
        res_roll_window1 = RollWindow(2, axis=0, roll_mode='center')._apply(arr, func_for_apply)
        assert isinstance(res_roll_window1, np.ndarray)
        assert res_roll_window1.shape == (13, 12)
        res_roll_window2 = RollWindow(3, axis=0, roll_mode='center')._apply(arr, func_for_apply)
        assert isinstance(res_roll_window2, np.ndarray)
        assert res_roll_window2.shape == (12, 12)
        res_roll_window3 = RollWindow(4, axis=0, roll_mode='center')._apply(arr, func_for_apply)
        assert isinstance(res_roll_window3, np.ndarray)
        assert res_roll_window3.shape == (11, 12)
        with pytest.raises(ValueError, match='window_size must be at least 2'):
            _ = RollWindow(1, roll_mode='center')._apply(arr, func_for_apply)

    @staticmethod
    def test_roll_modes() -> None:
        """Test different rolling modes."""
        arr = create_test_array(shape=(14, 12))
        res_roll_mode1 = RollWindow(5, axis=0, roll_mode='center')._apply(arr, func_for_apply)
        assert isinstance(res_roll_mode1, np.ndarray)
        assert res_roll_mode1.shape == (10, 12)
        res_roll_mode2 = RollWindow(5, axis=0, roll_mode='end')._apply(arr, func_for_apply)
        assert isinstance(res_roll_mode2, np.ndarray)
        assert res_roll_mode2.shape == (10, 12)
        res_roll_mode3 = RollWindow(5, axis=0, roll_mode='knn')._apply(arr, func_for_apply)
        assert isinstance(res_roll_mode3, np.ndarray)
        assert res_roll_mode3.shape == (14, 12)

    @staticmethod
    def test_padding_approaches() -> None:
        """Test different padding approaches."""
        arr = create_test_array(shape=(14, 12))
        res_roll_pad1 = RollWindow(5, axis=0, roll_mode='center', padding='none')._apply(arr, func_for_apply)
        assert isinstance(res_roll_pad1, np.ndarray)
        assert res_roll_pad1.shape == (10, 12)
        res_roll_pad2 = RollWindow(5, axis=0, roll_mode='center', padding='nan')._apply(arr, func_for_apply)
        assert isinstance(res_roll_pad2, np.ndarray)
        assert res_roll_pad2.shape == (14, 12)
        res_roll_pad3 = RollWindow(5, axis=0, roll_mode='center', padding='ext_edge')._apply(arr, func_for_apply)
        assert isinstance(res_roll_pad3, np.ndarray)
        assert res_roll_pad3.shape == (14, 12)
        res_roll_pad4 = RollWindow(5, axis=0, roll_mode='center', padding='constant_edge')._apply(arr, func_for_apply)
        assert isinstance(res_roll_pad4, np.ndarray)
        assert res_roll_pad4.shape == (14, 12)
        res_roll_pad5 = RollWindow(5, axis=0, roll_mode='center', padding='extrapolation')._apply(arr, func_for_apply)
        assert isinstance(res_roll_pad5, np.ndarray)
        assert res_roll_pad5.shape == (14, 12)

    @staticmethod
    def test_chain_apply() -> None:
        """Test chain_apply functionality."""
        arr = create_test_array(shape=(14, 12))
        res_roll = RollWindow(5, axis=0)._apply(arr, func_for_apply)
        res_roll_chain = RollWindow(5, axis=0)._chain_apply(arr, [func_for_apply])
        assert np.allclose(res_roll, res_roll_chain)
        res_roll_chain1 = RollWindow(5, axis=0)._chain_apply(arr, [func_for_apply, func_for_apply1])
        assert isinstance(res_roll_chain1, np.ndarray)
        assert res_roll_chain1.shape == (10, 12)
        window_roller = RollWindow(5, axis=0, padding='extrapolation')
        res_roll_chain2 = window_roller._chain_apply(arr, [func_for_apply, func_for_apply1])
        assert isinstance(res_roll_chain2, np.ndarray)
        assert res_roll_chain2.shape == (14, 12)


# %% Test main

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
