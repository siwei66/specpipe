# -*- coding: utf-8 -*-
"""
SpecPipe - Denoising module
"""
from .outlier import ArrayOutlier, replace_outlier
from .utils import RollWindow
from .moving_avg import MovingAvg
from .local_polynomial import LocalPolynomial
from .decomposition_filters import FourierFilter, WaveletFilter

__all__ = [
    "replace_outlier",
    "ArrayOutlier",
    "RollWindow",
    "MovingAvg",
    "LocalPolynomial",
    "FourierFilter",
    "WaveletFilter",
]
