# -*- coding: utf-8 -*-
"""
Tests for 1D data series denoising - decomposition-based filters for SpecPipe

Copyright (c) 2025 Siwei Luo. MIT License.
"""
import numpy as np
import pytest
import sys

from specpipe.denoiser.utils import create_test_array
from specpipe.denoiser.decomposition_filters import FourierFilter, WaveletFilter


# %% Test de


class TestDecomposition:
    """Test decomposition-based filter functionalities."""

    @staticmethod
    def test_fourier_filter() -> None:
        """Test fourier filter basic functionalities"""
        shape = (14, 12)
        arr = create_test_array(shape=shape)

        # Test axes
        ff = FourierFilter(cutoff=0.5, sampling_rate=10, axis=0)
        filtered_arr = ff.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape
        ff = FourierFilter(cutoff=0.5, sampling_rate=10, axis=1)
        filtered_arr = ff.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape

    @staticmethod
    def test_wavelet_filter_basic() -> None:
        """Test wavelet filter basic functionalities"""
        shape = (14, 12)
        arr = create_test_array(shape=shape)

        # Test axes
        wf = WaveletFilter(wavelet='haar', cutoff=0.5, axis=0)
        filtered_arr = wf.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape
        wf = WaveletFilter(wavelet='haar', cutoff=0.5, axis=1)
        filtered_arr = wf.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape

    @staticmethod
    def test_wavelet_filter_wavelet() -> None:
        """Test wavelet filter wavelet form"""
        shape = (200, 200)
        arr = create_test_array(shape=shape)

        # Test wavelet
        wf = WaveletFilter(wavelet='db4', cutoff=0.5, axis=0)
        filtered_arr = wf.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape
        wf = WaveletFilter(wavelet='db8', cutoff=0.5, axis=0)
        filtered_arr = wf.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape
        wf = WaveletFilter(wavelet='sym8', cutoff=0.5, axis=0)
        filtered_arr = wf.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape
        wf = WaveletFilter(wavelet='coif2', cutoff=0.5, axis=0)
        filtered_arr = wf.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape

    @staticmethod
    def test_wavelet_filter_threshold_mode() -> None:
        """Test wavelet filter threshold mode"""
        shape = (14, 12)
        arr = create_test_array(shape=shape)

        # Test threshold mode
        wf = WaveletFilter(wavelet='haar', cutoff=0.5, axis=0, threshold_mode='soft')
        filtered_arr = wf.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape
        wf = WaveletFilter(wavelet='haar', cutoff=0.5, axis=0, threshold_mode='hard')
        filtered_arr = wf.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape
        wf = WaveletFilter(wavelet='haar', cutoff=0.5, axis=0, threshold_mode='garrote')
        filtered_arr = wf.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape
        wf = WaveletFilter(wavelet='haar', cutoff=0.5, axis=0, threshold_mode='greater')
        filtered_arr = wf.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape
        wf = WaveletFilter(wavelet='haar', cutoff=0.5, axis=0, threshold_mode='less')
        filtered_arr = wf.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape

    @staticmethod
    def test_wavelet_filter_extension_mode() -> None:
        """Test wavelet filter extension mode"""
        shape = (14, 12)
        arr = create_test_array(shape=shape)

        # Test extension mode
        wf = WaveletFilter(wavelet='haar', cutoff=0.5, axis=0, extension_mode='zero')
        filtered_arr = wf.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape
        wf = WaveletFilter(wavelet='haar', cutoff=0.5, axis=0, extension_mode='constant')
        filtered_arr = wf.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape
        wf = WaveletFilter(wavelet='haar', cutoff=0.5, axis=0, extension_mode='symmetric')
        filtered_arr = wf.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape
        wf = WaveletFilter(wavelet='haar', cutoff=0.5, axis=0, extension_mode='reflect')
        filtered_arr = wf.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape
        wf = WaveletFilter(wavelet='haar', cutoff=0.5, axis=0, extension_mode='periodic')
        filtered_arr = wf.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape
        wf = WaveletFilter(wavelet='haar', cutoff=0.5, axis=0, extension_mode='smooth')
        filtered_arr = wf.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape
        wf = WaveletFilter(wavelet='haar', cutoff=0.5, axis=0, extension_mode='periodization')
        filtered_arr = wf.apply(arr)
        assert isinstance(filtered_arr, np.ndarray)
        assert filtered_arr.shape == shape


# %% Test main

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
