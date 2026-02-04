# -*- coding: utf-8 -*-
"""
Swectral - Model combiners
"""

# Basic voting and bagging
from swectral.modelcombiners.bagging import BaggingEnsembler, create_bagging_model
from swectral.modelcombiners.regressor_to_classifier import RegressorToClassifier, regressor_to_classifier

__all__ = [
    "BaggingEnsembler",
    "create_bagging_model",
    "RegressorToClassifier",
    "regressor_to_classifier",
]

# TODO: Boosting
# TODO: Stacking
# TODO: Torch wraper / constructor section - Input nn structure and wrapped into sklearn-style
