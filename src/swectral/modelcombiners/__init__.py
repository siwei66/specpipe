# -*- coding: utf-8 -*-
"""
Swectral - Model combiners
"""

# Basic voting and bagging
from swectral.modelcombiners.bagging import BaggingEnsembler

__all__ = ["BaggingEnsembler"]

# TODO: Boosting
# TODO: Stacking
# TODO: Torch wraper / constructor section - Input nn structure and wrapped into sklearn-style
