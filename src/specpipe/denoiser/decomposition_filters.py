# -*- coding: utf-8 -*-
"""
1D data series denoising - decomposition-based filters for SpecPipe

The following source code was created with AI assistance and has been human reviewed and edited.

Copyright (c) 2025 Siwei Luo. MIT License.
"""

import numpy as np
import pywt
from typing import Annotated, Any, Union

from ..specio import simple_type_validator, arraylike_validator


# %% Decomposition-based smoothing - Fourier Filter


class FourierFilter:
    """
    Fourier filter to 2D array-like of 1D series data.

    Attributes
    ----------
    sampling_rate : float, optional
        Sampling rate. The default uses the number of samples.
    cutoff : float, optional
        Percentage frequency cutoffs, must be a number between 0 and 1. The default is 0.1.
    axis : int, optional
        Axis of 1D signal. If 0, each row of 2D array represents an 1D signal.
        The default is 0.

    Methods
    -------
    apply
        Apply Fourier filter to 2D array-like of 1D series data.
    """

    @simple_type_validator
    def __init__(
        self,
        sampling_rate: Union[int, float] = -1.0,
        cutoff: float = 0.5,
        axis: int = 0,
    ) -> None:
        self.sampling_rate = sampling_rate
        if (axis != 0) & (axis != 1):
            raise ValueError(f'axis must be 0 or 1, got: {axis}')
        if cutoff >= 1 or cutoff <= 0:
            raise ValueError(f'cutoff must be between 0 and 1, got: {cutoff}')
        self.cutoff = cutoff
        self.axis = axis

    @simple_type_validator
    def apply(
        self,
        data_array: Union[Annotated[Any, arraylike_validator(ndim=2)], Annotated[Any, arraylike_validator(ndim=1)]],
    ) -> np.ndarray:
        """
        Apply Fourier filter to 2D array-like of 1D series data.

        Parameters
        ----------
        data_array : 1D array-like or 2D array-like
            1D data array or 2D data array of 1D series data.

        Returns
        -------
        numpy.ndarray
            Array of filtered signals.

        Examples
        --------
        >>> ff = FourierFilter()
        >>> ff.apply([1, 2, 3, 4, 5, 6, 77, 88, 9, 10])
        >>> ff.apply([[1, 2, 3, 4, 5, 6, 77, 88, 9, 10], [1, 22, 33, 4, 5, 6, 7, 8, 9, 10]])

        Add to prepared ``SpecPipe`` instance ``pipe`` for ROI pixel spectrum processing::

            >>> pipe.add_process(6, 6, 0, FourierFilter().apply)

        Add to prepared ``SpecPipe`` instance ``pipe`` for the processing of 1D sample data::

            >>> pipe.add_process(7, 7, 0, FourierFilter().apply)
        """
        axis = self.axis
        sampling_rate = self.sampling_rate
        perc_cutoff = self.cutoff

        _is_1d: bool = False
        data_array = np.asarray(data_array)
        if axis == 1 and data_array.ndim == 2:
            data_array = data_array.T
        elif axis == 1 and data_array.ndim == 1:
            data_array = data_array.reshape(1, -1)
            _is_1d = True

        # Calculate the Nyquist frequency
        nyquist_freq = 0.5 * sampling_rate

        # Validate percentage cutoff
        if perc_cutoff >= 1 or perc_cutoff <= 0:
            raise ValueError(f'cutoff must be between 0 and 1, got: {perc_cutoff}')

        # Perform the Fourier Transform
        freq_signals = np.fft.fft(data_array, axis=1)

        # Create a frequency array
        num_samples = data_array.shape[1]
        freq = np.fft.fftfreq(num_samples, d=1 / sampling_rate)

        # Create a low-pass filter mask
        filter_mask = (np.abs(freq) / nyquist_freq) <= perc_cutoff

        # Apply the low-pass filter
        filtered_freq_signals = freq_signals * filter_mask

        # Inverse Fourier Transform to get the filtered signals
        filtered_signals = np.fft.ifft(filtered_freq_signals, axis=1)

        # Return the real part of the filtered signals
        out = np.real(filtered_signals)

        if axis == 1:
            out = out.T

        if _is_1d:
            out = np.asarray(out).reshape(-1)

        assert isinstance(out, np.ndarray)
        return out


# %% Decomposition-based smoothing - Wavelet Filter


class WaveletFilter:
    """
    Wavelet filter for 2D array-like of 1D series data.

    This class provides wavelet denoising functionality using PyWavelets as the underlying implementation.

    Attributes
    ----------
    wavelet : str, optional
        Wavelet form. The default is "haar".

        See ``PyWavelets`` documentation for available options.

    cutoff : float
        Percentage frequency cutoffs, must be a number between 0 and 1.

        The default is 0.1.

    threshold_mode : str
        Thresholding modes for wavelet coefficient processing.

        The default is "soft".

        See ``PyWavelets`` documentation for available options.

    extension_mode : str
        Signal extension mode. The default is "symmetric".

        See ``PyWavelets`` documentation for available options.

    axis : int, optional
        Axis of 1D signal. If 0, each row of 2D array represents an 1D signal.

        The default is 0.

    Methods
    -------
    apply
        Apply wavelet filter to 2D array-like of 1D series data.

    See Also
    --------
    PyWavelets
    """

    @simple_type_validator
    def __init__(
        self,
        wavelet: str = 'haar',
        cutoff: float = 0.5,
        threshold_mode: str = 'soft',
        extension_mode: str = 'symmetric',
        axis: int = 0,
    ) -> None:
        self.wavelet = wavelet
        self.cutoff = cutoff
        self.threshold_mode = threshold_mode
        self.extension_mode = extension_mode

        # Validate axis
        if (axis != 0) and (axis != 1):
            raise ValueError(f'axis must be 0 or 1, got: {axis}')
        self.axis = axis

    @staticmethod
    @simple_type_validator
    def _cutoff_to_threshold(pywt_coeffs: list[np.ndarray], perc_cutoff: float) -> float:
        """Convert percentage frequency cutoff to wavelet filter threshold."""
        # Flatten coefficients
        all_coeffs = np.concatenate(list(pywt_coeffs[1:]))

        # Calculate threshold of the percentage cutoff
        if perc_cutoff == 1.0:
            return 0.0
        sorted_coeffs = np.sort(np.abs(all_coeffs))[::-1]  # Sorted descending

        # Find threshold of perc_cutoff
        n_keep = int(perc_cutoff * len(sorted_coeffs))
        if n_keep == 0:
            threshold = sorted_coeffs[0] + 1  # Threshold above maximum
        else:
            threshold = sorted_coeffs[n_keep - 1]

        return float(threshold)

    @staticmethod
    @simple_type_validator
    def _safe_threshold(
        coeffs_i: Annotated[np.ndarray, arraylike_validator(ndim=1)],
        threshold: float,
        threshold_mode: str,
    ) -> np.ndarray:
        """Threshold with safe division"""
        if threshold_mode in ['soft', 'garrote']:
            magnitude = np.abs(coeffs_i)
            epsilon = 1e-15
            safe_magnitude = magnitude + epsilon
            if threshold_mode == 'soft':
                thresholded = coeffs_i * np.maximum(0, 1 - threshold / safe_magnitude)
            elif threshold_mode == 'garrote':
                thresholded = coeffs_i * np.maximum(0, 1 - threshold**2 / safe_magnitude**2)
            return np.array(thresholded)
        elif threshold_mode in ['hard', 'greater', 'less']:
            thresholded = pywt.threshold(coeffs_i, threshold, mode=threshold_mode)
            return np.array(thresholded)
        else:
            raise ValueError(
                f"threshold_mode must be one of 'soft' / 'hard' / 'garrote' / 'greater' / 'less', \
                    but got: {threshold_mode}"
            )

    @simple_type_validator
    def apply(
        self,
        data_array: Union[Annotated[Any, arraylike_validator(ndim=2)], Annotated[Any, arraylike_validator(ndim=1)]],
    ) -> np.ndarray:
        """
        Apply Fourier filter to 2D array-like of 1D series data.

        Parameters
        ----------
        data_array : 1D array-like or 2D array-like
            1D data array or 2D data array of 1D series data.

        Returns
        -------
        numpy.ndarray
            Array of filtered signals.

        Examples
        --------
        >>> wf = WaveletFilter()
        >>> wf.apply([1, 2, 3, 4, 5, 6, 77, 88, 9, 10])
        >>> wf.apply([[1, 2, 3, 4, 5, 6, 77, 88, 9, 10], [1, 22, 33, 4, 5, 6, 7, 8, 9, 10]])

        Add to prepared ``SpecPipe`` instance ``pipe`` for ROI pixel spectrum processing::

            >>> pipe.add_process(6, 6, 0, WaveletFilter().apply)

        Add to prepared ``SpecPipe`` instance ``pipe`` for the processing of 1D sample data::

            >>> pipe.add_process(7, 7, 0, WaveletFilter().apply)
        """
        wavelet = self.wavelet
        cutoff = self.cutoff
        axis = self.axis

        _is_1d: bool = False
        data_array = np.asarray(data_array)
        if axis == 1 and data_array.ndim == 2:
            data_array = data_array.T
        elif axis == 1 and data_array.ndim == 1:
            data_array = data_array.reshape(1, -1)
            _is_1d = True

        # Initialize output
        out = np.zeros_like(data_array)

        # Perform wavelet decomposition for all signals
        for i in range(data_array.shape[0]):
            # Get single signal
            signal = data_array[i, :]

            # Perform wavelet decomposition
            coeffs = pywt.wavedec(signal, wavelet, mode=self.extension_mode)
            threshold = self._cutoff_to_threshold(pywt_coeffs=coeffs, perc_cutoff=cutoff)

            # Apply thresholding to detail coefficients
            coeffs_thresholded = [
                self._safe_threshold(coeffs_i, threshold, threshold_mode=self.threshold_mode) for coeffs_i in coeffs
            ]

            # Reconstruct the smoothed signal
            reconstructed_signal = pywt.waverec(coeffs_thresholded, wavelet, mode=self.extension_mode)

            # Pad reconstructed signal to keep shape
            input_signal_len = len(signal)
            reconstructed_signal_len = len(reconstructed_signal)
            if reconstructed_signal_len > input_signal_len:
                reconstructed_signal = reconstructed_signal[:input_signal_len]
            elif reconstructed_signal_len < input_signal_len:
                pad_len = input_signal_len - reconstructed_signal_len
                reconstructed_signal = np.pad(reconstructed_signal, (pad_len // 2, pad_len // 2 + 1), mode='edge')

            # Store the filtered signal in the output array
            out[i] = reconstructed_signal

        if axis == 1:
            out = out.T

        if _is_1d:
            out = np.asarray(out).reshape(-1)

        assert isinstance(out, np.ndarray)
        return out
