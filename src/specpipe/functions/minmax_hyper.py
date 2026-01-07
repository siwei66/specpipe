# -*- coding: utf-8 -*-
"""
SpecPipe - process functions - MinMax (MinMax Normalization) for hyperspectral image pixel application

Copyright (c) 2025 Siwei Luo. MIT License.
"""

import torch

from typing import Annotated, Any
from ..specio import arraylike_validator, simple_type_validator


# %% MinMax


@simple_type_validator
def minmax_hyper(data: Annotated[Any, arraylike_validator(ndim=3)]) -> torch.Tensor:
    """
    MinMax (MinMax Normalization) function for image pixel spectrum correction in SpecPipe pipelines.
    This function is for optimized GPU-accelerated application.

        Process input data level: 4 - 'pixel_hyperspecs_tensor'

        Process output data level: 4 - 'pixel_hyperspecs_tensor'

    Parameters
    ----------
    data : 3D array-like (n_samples, n_bands, 1)
        Three-dimensional array containing 2D spectral data to be processed.
        The size of the third dimension must be exactly 1.

    Returns
    -------
    torch.Tensor
        SNV transformed spectral data.

    Examples
    --------
    >>> minmax_hyper([[[1], [2], [3], [4], [5], [6]], [[2], [2], [4], [4], [6], [6]]])

    Incorporation into pipeline for image processing, for SpecPipe instance ``pipe``:

        >>> pipe.add_process(4, 4, 0, minmax_hyper)
    """
    import numpy as np  # noqa: W291
    import torch  # noqa: W291

    if isinstance(data, torch.Tensor):
        tensor_data = data
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor_data = torch.from_numpy(np.asarray(data)).to(device)

    vmin = torch.min(tensor_data, dim=1, keepdim=True).values
    vmax = torch.max(tensor_data, dim=1, keepdim=True).values
    minmax_values = (tensor_data - vmin) / (vmax - vmin + 1e-15)

    assert isinstance(minmax_values, torch.Tensor)
    return minmax_values
