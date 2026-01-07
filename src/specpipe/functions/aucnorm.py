# -*- coding: utf-8 -*-
"""
SpecPipe - process functions - AUC (Area Under Curve) normalization

Copyright (c) 2025 Siwei Luo. MIT License.
"""

import numpy as np

from typing import Annotated, Any
from ..specio import arraylike_validator, simple_type_validator


# %% AUC normalization


@simple_type_validator
def aucnorm(data: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
    """
    AUC (Area Under Curve) normalization function.

    For image pixel spectrum correction in SpecPipe pipelines:

        Set process input data level: 2 / 'pixel_specs_array'

        Set process output data level: 2 / 'pixel_specs_array'

    For ROI spectrum normalization:

        Set process input data level: 6 / 'roispecs'

        Set process output data level: 6 / 'roispecs'

    For sample spectrum normalization:

        Set process input data level: 7 / 'spec1d'

        Set process output data level: 7 / 'spec1d'

    Parameters
    ----------
    data : 2D array-like (n_samples, n_bands)
        Two-dimensional array-like spectral data to be processed.

    Returns
    -------
    numpy.ndarray
        AUC normalization transformed spectral data.

    Examples
    --------
    >>> aucnorm([[1, 2, 3, 4, 5, 6], [2, 2, 4, 4, 6, 6]])

    Incorporation into pipeline for image processing, for SpecPipe instance ``pipe``:

        >>> pipe.add_process(2, 2, 0, aucnorm)

    Incorporation into pipeline for ROI spectra processing, for SpecPipe instance ``pipe``:

        >>> pipe.add_process(6, 6, 0, aucnorm)

    Incorporation into pipeline for 1D spectra processing, for SpecPipe instance ``pipe``:

        >>> pipe.add_process(7, 7, 0, aucnorm)
    """
    import numpy as np  # noqa: W291

    areas = np.sum(np.abs(data), axis=1, keepdims=True)
    auc_normalized = data / (areas + 1e-15)

    return np.asarray(auc_normalized)
