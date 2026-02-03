# -*- coding: utf-8 -*-
"""
Swectral - Model combiners
"""

# Basic voting and bagging
from swectral.modelcombiners.bagging import BaggingEnsembler, create_bagging_model

__all__ = [
    "BaggingEnsembler",
    "create_bagging_model",
]

# TODO: Boosting
# TODO: Stacking
# TODO: Torch wraper / constructor section - Input nn structure and wrapped into sklearn-style
