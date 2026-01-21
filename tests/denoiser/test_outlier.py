# -*- coding: utf-8 -*-
"""
Tests for small 1D data outlier replacing tool

Copyright (c) 2025 Siwei Luo. MIT License.
"""
import numpy as np
import pytest
import sys

from swectral.denoiser.utils import create_test_array
from swectral.denoiser.outlier import ArrayOutlier, replace_outlier


# %% TestArrayOutlier


class TestArrayOutlier:
    """Test ArrayOutlier and replace_outlier functionalities."""

    @staticmethod
    def test_generate_report() -> None:
        """Test generate_report"""
        shape = (14, 12)
        arr = create_test_array(shape=shape)

        # Function
        res_o2 = replace_outlier(arr, 'dixon', generate_report=True, axis=1, to='nan')
        assert isinstance(res_o2, tuple)
        assert len(res_o2) == 2
        assert res_o2[0].shape == shape
        assert len(res_o2[1]) == 14

        res_o2_no_rep = replace_outlier(arr, 'dixon', generate_report=False, axis=1, to='nan')
        assert isinstance(res_o2_no_rep, np.ndarray)
        assert res_o2_no_rep.shape == shape

    @staticmethod
    def test_computation_axis() -> None:
        """Test different computation axes."""
        shape = (14, 12)
        arr = create_test_array(shape=shape)

        res_o1 = replace_outlier(arr, 'dixon', generate_report=True, axis=0, to='nan')
        assert res_o1[0].shape == shape
        assert len(res_o1[1]) == 12

        res_o2 = replace_outlier(arr, 'dixon', generate_report=True, axis=1, to='nan')
        assert res_o2[0].shape == shape
        assert len(res_o2[1]) == 14

    @staticmethod
    def test_different_replacement_methods() -> None:
        """Test different replacement methods."""
        shape = (14, 12)
        arr = create_test_array(shape=shape)

        res_o3 = replace_outlier(arr, 'dixon', to='mean')
        assert isinstance(res_o3, np.ndarray)
        assert res_o3.shape == shape

        res_o4 = replace_outlier(arr, 'dixon', to='median')
        assert isinstance(res_o4, np.ndarray)
        assert res_o4.shape == shape

        res_o5 = replace_outlier(arr, 'dixon', to='neighbor')
        assert isinstance(res_o5, np.ndarray)
        assert res_o5.shape == shape

        res_o6 = replace_outlier(arr, 'dixon', to='nan')
        assert isinstance(res_o6, np.ndarray)
        assert res_o6.shape == shape

    @staticmethod
    def test_different_outlier_detection_methods() -> None:
        """Test different outlier detection methods."""
        shape = (14, 12)
        arr = create_test_array(shape=shape)

        res_o_rr1 = replace_outlier(arr, 'dixon')
        assert isinstance(res_o_rr1, np.ndarray)
        assert res_o_rr1.shape == shape

        res_o_rr2 = replace_outlier(arr, 'iqr')
        assert isinstance(res_o_rr2, np.ndarray)
        assert res_o_rr2.shape == shape

        res_o_rr3 = replace_outlier(arr, 'modified_z')
        assert isinstance(res_o_rr3, np.ndarray)
        assert res_o_rr3.shape == shape

    @staticmethod
    def test_function_class_method_consistency() -> None:
        """Test different outlier detection methods."""
        arr = create_test_array()
        outlier_replacer = ArrayOutlier('dixon', axis=1, to='median', generate_report=True)
        res_method = outlier_replacer.replace(arr)

        res_func = replace_outlier(arr, 'dixon', axis=1, to='median', generate_report=True)

        assert np.allclose(res_func[0], res_method)
        assert res_func[1] == outlier_replacer.report[0]

    @staticmethod
    def test_class_method_report_accumulation() -> None:
        arr = create_test_array(shape=(14, 12))
        outlier_replacer = ArrayOutlier('dixon', generate_report=True, axis=1, to='median')

        res_method = outlier_replacer.replace(arr)
        assert len(res_method) == 14
        assert len(outlier_replacer.report) == 1
        assert len(outlier_replacer.report[0]) == 14

        res_method1 = outlier_replacer.replace(arr)
        assert len(res_method1) == 14
        assert len(outlier_replacer.report) == 2
        assert len(outlier_replacer.report[0]) == 14
        assert len(outlier_replacer.report[1]) == 14

    @staticmethod
    def test_sample_size_edge_case() -> None:
        """Test sample size edge case."""
        arr = create_test_array()

        with pytest.raises(ValueError, match="requires a minimum sample size of 5"):
            _ = replace_outlier(arr[0:4, :], 'modified_z', axis=0)

        with pytest.warns(UserWarning, match="requires minimum 12 samples for reliable results"):
            _ = replace_outlier(arr[0:8, :], 'modified_z', axis=0)


# %% Test main

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
