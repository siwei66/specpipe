# -*- coding: utf-8 -*-
"""
SpecPipe - Model combiners
"""
from .model_connector import (
    combine_transformer_classifier,
    combine_transformer_regressor,
    TransClassifier,
    TransRegressor,
)

__all__ = ["combine_transformer_classifier", "combine_transformer_regressor", "TransClassifier", "TransRegressor"]
