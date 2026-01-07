# -*- coding: utf-8 -*-
"""
SpecPipe denoiser submodule
"""
from .outlier import ArrayOutlier, replace_outlier
from .moving_avg import MovingAvg
from .local_polynomial import LocalPolynomial
from .decomposition_filters import FourierFilter, WaveletFilter

__all__ = [
    "replace_outlier",
    "ArrayOutlier",
    "MovingAvg",
    "LocalPolynomial",
    "FourierFilter",
    "WaveletFilter",
]
