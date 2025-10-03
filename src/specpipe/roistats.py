# -*- coding: utf-8 -*-
"""
Basic ROI descriptive statistics functions

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Basics
# Calculation
import math

# For local test - delete after use
import warnings

# Typing
from typing import Annotated, Any, Callable, Optional, Protocol, Union, overload, runtime_checkable

import numpy as np
import pandas as pd

# Raster operation
import rasterio
import torch
from rasterio.mask import mask
from shapely.geometry import Polygon, box

# Self
from .specio import arraylike_validator, simple_type_validator

# %% Image and ROI functions


# ROI spectral data extractor
class ROISpec:
    """
    Extract spectra from spectral raster image in a multipolygon ROI defined with vertex coordinate pairs.

    Parameters
    ----------
    as_type : Union[type,str], optional
        Numeric data type for the output spectral values, support NumPy numeric dtypes. The default is 'float32'.
    """

    @simple_type_validator
    def __init__(self, as_type: Union[type, str] = "float32") -> None:
        self.as_type = as_type

    @simple_type_validator
    def roispec(
        self,
        image_path: str,
        roi_coordinates: list[list[tuple[Union[int, float], Union[int, float]]]],
    ) -> Annotated[Any, arraylike_validator(ndim=2)]:
        """
        Extract spectra from spectral raster image in a multipolygon ROI defined with vertex coordinate pairs.

        Parameters
        ----------
        image_path : str
            Spectral raster image path.

        roi_coordinates : list[list[tuple[Union[int, float],Union[int, float]]]]
            Lists of vertex coordinate pairs of the polygons of a region of interest (ROI).

        Returns
        -------
        pix_arr : ndarray
            Array of pixel spectra in the specified ROI, each row represents the spectrum values of a pixel.
        """
        as_type = self.as_type
        # Silencing NotGeoreferencedWarning
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

        # Validate roi_coordinates
        # Number of valid polygons
        npoly = 0
        if len(roi_coordinates) > 0:
            for roi_coords in roi_coordinates:
                if len(roi_coords) >= 3:
                    npoly = npoly + 1
        if npoly == 0:
            raise ValueError(f"No valid polygon found in the given roi_coordinates: {roi_coordinates}")

        # For each polygon
        for i, polygon_coords in enumerate(roi_coordinates):
            # Create a Polygon object from the coordinates
            polygon = Polygon(polygon_coords)

            with rasterio.open(image_path) as src:
                # Get raster bounds
                raster_bbox = box(*src.bounds)
                # Validate Polygon
                if raster_bbox.intersects(polygon):
                    # Apply the polygon mask to raster
                    out_image, out_transform = mask(src, [polygon], crop=True, nodata=src.nodata)
                    # Flatten the bands
                    out_image = out_image.reshape(out_image.shape[0], -1)
                    # Get valid pixel values
                    out_image = out_image[:, np.sum(out_image, axis=0) > 0]
                    out_image = out_image.astype(as_type)
                else:
                    raise ValueError(
                        f"\nROI located out of raster image bounds, \ngot ROI bounds: {polygon.bounds} , \
                            \nimage bounds: {src.bounds}"
                    )

            # Concatenate results if ROI is multipolygon
            if i == 0:
                out_image_total = out_image
            else:
                out_image_total = np.concatenate((out_image_total, out_image), axis=1)

        # Type conversion
        out_image_total = out_image_total.astype(as_type)

        # Return data default to row as sample
        out_image_total = out_image_total.T

        # Recover NotGeoreferencedWarning
        warnings.filterwarnings("default", category=rasterio.errors.NotGeoreferencedWarning)

        return out_image_total


# Fast call of ROISpec.roispec
@simple_type_validator
def roispec(
    image_path: str, roi_coordinates: list[list[tuple[Union[int, float], Union[int, float]]]]
) -> Annotated[Any, arraylike_validator(ndim=2)]:
    """
    Extract spectra from spectral raster image in a multipolygon ROI defined with vertex coordinate pairs.
    The data type of output values are fixed to float32. Use ROISpec(as_type).roispec to customize data types in the automation pipeline.

    Parameters
    ----------
    image_path : str
        Spectral raster image path.

    roi_coordinates : list[list[tuple[Union[int, float],Union[int, float]]]]
        Lists of vertex coordinate pairs of the polygons of a region of interest (ROI).

    Returns
    -------
    pix_arr : ndarray
        Array of pixel spectra in the specified ROI, each row represents the spectrum values of a pixel.
    """  # noqa: E501
    return ROISpec().roispec(image_path, roi_coordinates)


# ROI pixel counter
@simple_type_validator
def pixcount(
    image_path: str,
    roi_coordinates: list[list[tuple[Union[int, float], Union[int, float]]]],
    band: int = 1,
    threshold: Optional[tuple[Union[int, float], Union[int, float]]] = None,
) -> int:
    """
    Count valid pixel number within a ROI larger than a threshold at specific band.

    Parameters
    ----------
    image_path : str
        Spectral raster image path.

    roi_coordinates : list[list[tuple[Union[int, float],Union[int, float]]]]
        Lists of vertex coordinate pairs of the polygons of a region of interest (ROI).

    band : int, optional
        Reference band, if threshold is given, values at this band is compared with threshold.

    threshold : tuple[Union[int,float],Union[int,float]], optional
        The band value is compared with threshold, only pixel within the threshold are counted.

    Returns
    -------
    total_count : int
        Valid pixel number within a ROI larger than a threshold at specific band.
    """
    # Silencing NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    # For each polygon
    total_count = 0
    for polygon_coords in roi_coordinates:
        # Create a Polygon object from the coordinates
        polygon = Polygon(polygon_coords)

        with rasterio.open(image_path) as src:
            # Get raster bounds
            raster_bbox = box(*src.bounds)
            # Validate Polygon
            if raster_bbox.intersects(polygon):
                # Count
                masked, _ = mask(src, [polygon], crop=True, nodata=src.nodata, indexes=[band], all_touched=True)
                if threshold is None:
                    count: int = int(np.sum(masked[0] != src.nodata))
                else:
                    if threshold[0] >= threshold[1]:
                        raise ValueError(f"Invalid threshold range: {threshold}")
                    count = int(
                        np.sum((masked[0] != src.nodata) & (masked[0] >= threshold[0]) & (masked[0] <= threshold[1]))
                    )
            else:
                count = 0

        total_count = total_count + count

    # Recover NotGeoreferencedWarning
    warnings.filterwarnings("default", category=rasterio.errors.NotGeoreferencedWarning)

    return total_count


# Find min sampling bbox from a ROI with min valid pixels
@simple_type_validator
def minbbox(
    image_path: str,
    roi_coordinates: list[list[tuple[Union[int, float], Union[int, float]]]],
    minimum_valid_pixel: int,
    band: int = 1,
    valid_threshold: Optional[tuple[Union[int, float], Union[int, float]]] = None,
) -> list[list[tuple[float, float]]]:
    """
    Get minimum bounding box of a ROI that has at least some number of valid pixels with band value within a threshold.
    """
    # Get ROI bounding box
    carr = np.array(roi_coordinates[0])
    xmin, xmax = min(carr[:, 0]), max(carr[:, 0])
    ymin, ymax = min(carr[:, 1]), max(carr[:, 1])
    cdbounds = [[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]]

    # Count total number
    total_count = pixcount(image_path, roi_coordinates, band, valid_threshold)
    incx, incy = (xmax - xmin), (ymax - ymin)
    if total_count < 4 * minimum_valid_pixel:
        return cdbounds
    while True:
        incx, incy = incx / 2, incy / 2
        for i in (0, 1):
            for j in (0, 1):
                xmin, xmax = (xmin + incx * i), (xmin + incx * (i + 1))
                ymin, ymax = (ymin + incx * j), (ymin + incx * (j + 1))
                mbounds = [[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]]
                total_count = pixcount(image_path, mbounds, band, valid_threshold)
                if total_count > 4 * minimum_valid_pixel:
                    break
                else:
                    return mbounds
            if total_count > 4 * minimum_valid_pixel:
                break


# %% Stats metrics of extracted 2D array data


# Derivative calculator for 2d-array
@simple_type_validator
def nderiv(
    data: Annotated[Any, arraylike_validator(ndim=2)],
    n: int,
    axis: int = 1,
    edge: Union[int, float, None] = None,
) -> Annotated[Any, arraylike_validator(ndim=2)]:
    """
    Compute arbitrary n-th derivatives of 1D data series in 2D data array.

    Parameters
    ----------
    data : np.ndarray
        2D arraylike data.

    n : int
        Derivative order.

    axis : int, optional
        Axis along which to perform the calcuation.
        - 0: Calculate row-wise (treat each row as a variable, while each column as a sample)
        - 1: Calculate column-wise (treat each column as a variable, while each row as a sample)
        The default is 1.

    edge : optional
        Unavailable edge values of dervatives vectors. The default is nan.

    Returns
    -------
    derivative : np.ndarray
        The n-th derivative of the input data along given axis in the same size.

    Raises
    ------
    ValueError
        If data length insufficient to compute nth derivative.

    ValueError
        If given edge is not int nor float.

    """
    data = np.asarray(data).astype("float32")

    if axis == 0:
        data = data.T

    # Validate length of data series.
    dlen = data.shape[1]
    if dlen < 2 * n:
        raise ValueError(
            f"Insufficient data length. \
                The input data must have at least {2 * n} elements to compute the {n}th derivative."
        )

    # Compute nth derivative
    if n == 0:
        derivative = data
    else:
        for od in range(1, n + 1):
            # Create an array to hold the n-th derivative
            derivative = np.zeros_like(data)
            for i in range(od, dlen - od):
                derivative[:, i] = (data[:, i + 1] - data[:, i - 1]) / 2
            data = derivative

        # Edge value
        if edge is None:
            ed = np.nan
        elif (type(edge) is int) or (type(edge) is float):
            ed = edge
        else:
            raise TypeError(f"edge value must be int or float, got: {type(edge)}")
        derivative[:, 0:od] = ed
        derivative[:, dlen - od : dlen] = ed

    if axis == 0:
        derivative = derivative.T

    return derivative


# Computing axis converter
@simple_type_validator
def axisconv(
    arr2d: Annotated[Any, arraylike_validator(ndim=2, as_type=float)],
    axis: int,
    astype: type,
) -> np.ndarray:
    """
    Transpose array to make computing axis 0.
    """
    arr: np.ndarray
    if axis == 0:
        arr = np.array(arr2d).astype(astype)
    elif axis == 1:
        arr = np.array(arr2d).astype(astype).T
    else:
        raise ValueError(f"Axis must be 0 or 1, got: {axis}")
    return arr


# Statistical moment
@simple_type_validator
def moment2d(  # noqa: C901
    data_array_2d: Annotated[Any, arraylike_validator(ndim=2, as_type=float)],
    n: int,
    standardized: bool = False,
    reference: Optional[tuple[Union[int, float]]] = None,
    axis: int = 0,
    zero_division: str = "add",
) -> tuple[float]:
    """
    Arbitrary statistical moment calculator for 2D array.

    Parameters
    ----------
    data_array_2d : ndarray-like
        2D array-like data. Nan value is omited.
    n : int
        Order of moment.
    standardized : bool, optional
        Whether standardization is applied. If True, the moment is scaled with standard deviation. The default is True.
    reference : tuple[Union[int,float]], optional
        Reference point. If not given, the reference point is 0 for first moment and mean for higher-order moments.
    axis : int, optional
        Axis along which the moment is computed, axis=0 treats each row as a sample. The default is 0.
    zero_division : str
        Choose between:
         - "add" : add a small number (1e-30) to denominator to avoid zero.
         - "replace" : replace zero with a small number (1e-30) in the denominator.
         - "nan" : return nan for zero divisions.
         - "numpy" : use default approach of numpy, i.e. return nan for 0 / 0, and +/-inf for non-zero values.
         - "raise" : raise Error for zero divisions.
        The default is "add".

    Returns
    -------
    mom : tuple
        Target moment.
    """
    # Standardize axis
    arr = axisconv(data_array_2d, axis, float)

    # Validate sample size
    nsample = arr.shape[0]
    if nsample < n:
        raise ValueError(f"Sample size must be at least {n} to compute {n}th moment, got: {nsample}")

    # Number of variables
    nvar = arr.shape[1]

    if (n == 1) & (reference is None):
        mom = np.nanmean(arr, axis=0)
    else:
        # Reference point
        if reference is None:
            ref = np.nanmean(arr, axis=0, keepdims=True)
        elif len(reference) == nvar:
            ref = np.array(reference).reshape(1, -1)
        else:
            raise ValueError(
                f"Reference and data_array_2d must have same number of variables, \
                    but given reference has {len(reference)}, while given data has {nvar}"
            )

        # Moment
        dev = arr - ref
        mom = np.nanmean(dev**n, axis=0)

        # Standardization
        if standardized:
            std = np.nanstd(arr, axis=0)
            denom = std**n
            # Zero division treatment
            if zero_division.lower() == "add":
                denom = denom + 1e-30
            elif zero_division.lower() == "replace":
                denom[denom == 0] = 1e-30
            elif zero_division.lower() == "nan":
                denom[denom == 0] = np.nan
            elif zero_division.lower() == "numpy":
                pass
            elif zero_division.lower() == "raise":
                if np.any(denom == 0):
                    raise ValueError(f"\nZero value found in moment denominator: \n\n{denom}\n")
            else:
                raise ValueError(
                    f"Invalid zero_division policy '{zero_division}', \
                        zero_division must be 'add'/'replace'/'nan'/'numpy' or 'raise'."
                )
            mom = mom / denom
    return tuple(mom)


# Calculate band value histogram values - band value distribution
@simple_type_validator
def bandhist(
    spec_array_2d: Annotated[Any, arraylike_validator(ndim=2, as_type=float)],
    band: int,
    bins: Union[int, list[float]],
    axis: int = 0,
) -> tuple[float, ...]:
    """
    Band value histogram, compute quantile values of a band or bands at given qunatiles of a 2D spectra data array.
    The quantiles can be given as number of bins or list of quantiles.

    Parameters
    ----------
    spec_array_2d : ndarray-like
        2D spectra data array-like. If each column is treat as a sample, set axis=1.

    band : int
        band index or list of band indices for computation.

    bins : Union[int, list[float]]
        DESCRIPTION. The default is 0.

    axis : int, optional
        Axis along which the moment is computed, axis=0 treats each row as a sample. The default is 0.

    Returns
    -------
    values : tuple[float, ...]
        Quantile values at given bins.
    """
    # Standardize axis
    arr = axisconv(spec_array_2d, axis, float)

    # Number of variables
    nvar = arr.shape[1]

    # Validate band
    if (band >= nvar) or (band < -nvar):
        raise ValueError(f"band index exceeds the number of bands, got: {band}, number of bands: {nvar}")

    # Bins to quantiles
    if type(bins) is int:
        if bins < 2:
            raise ValueError(f"number of bins must be greater than 1, got number of bins: {bins}")
        bs = [float(i / (bins - 1)) for i in range(bins)]
    elif type(bins) is list:
        if (max(bins) <= 1) & (min(bins) >= 0):
            bs = bins
        else:
            raise ValueError(f"Bin value must be in range of [0, 1], got range: [{min(bins)}, {max(bins)}]")
        if len(bins) < 2:
            raise ValueError(f"number of bins must be greater than 1, got number of bins: {len(bins)}")

    # Calculate quantile values
    values: tuple[float, ...] = ()
    for q in bs:
        v = float(np.nanquantile(arr[:, band], q, axis=0))
        values = values + (v,)

    return values


# %% Common and custom statistical measures for 2D array-like


# Helper: stats measure opter
@simple_type_validator
def smopt(measure: str) -> Callable:  # noqa: C901
    """
    Return common statistical measure function by measure name.
    """

    # Skewness and kurtosis modified for zero std
    @simple_type_validator
    def stskew(v: Annotated[Any, arraylike_validator(ndim=2, as_type=float)], axis: int) -> tuple[float]:
        skewness: tuple[float] = moment2d(v, 3, standardized=True, axis=axis)
        return skewness

    @simple_type_validator
    def stkurt(v: Annotated[Any, arraylike_validator(ndim=2, as_type=float)], axis: int) -> tuple[float]:
        kurtosis: tuple[float] = moment2d(v, 4, standardized=True, axis=axis)
        return kurtosis

    # Single common measure function
    if type(measure) is str:
        if str(measure).lower() == "mean":
            mfunc = np.nanmean
        elif str(measure).lower() == "median":
            mfunc = np.nanmedian  # type: ignore[assignment]
            # overloaded expression function, follows assignment error the same reason
        elif str(measure).lower() == "min":
            mfunc = np.nanmin  # type: ignore[assignment]
        elif str(measure).lower() == "max":
            mfunc = np.nanmax  # type: ignore[assignment]
        elif str(measure).lower() == "var":
            mfunc = np.nanvar  # type: ignore[assignment]
        elif (str(measure).lower() == "std") | (str(measure).lower() == "stdev") | (str(measure).lower() == "sd"):
            mfunc = np.nanstd  # type: ignore[assignment]
        elif (str(measure).lower() == "skewness") | (str(measure).lower() == "skew"):
            mfunc = stskew
        elif (str(measure).lower() == "kurtosis") | (str(measure).lower() == "kurt"):
            mfunc = stkurt
        else:
            raise ValueError("Undefined measure name, please provide the measure function instead")

    def msfunc(arr: Annotated[Any, arraylike_validator(ndim=2, as_type=float)]) -> np.ndarray:
        result: np.ndarray = np.array(mfunc(arr, axis=0))
        return result

    return msfunc


# Helper: custom measure function validator
@simple_type_validator
def cmval(arr: Annotated[Any, arraylike_validator(ndim=2)], cfunc: Callable) -> Callable:
    """
    Custom measure function validator
    """
    cfunc_name = cfunc.__name__
    test_arr = np.array(arr).astype(float)[0:5, :]
    try:
        testv = cfunc(test_arr)
    except Exception as e:
        raise ValueError(
            f"Error in the testing given custom measure \n'{cfunc_name}' on the provided data: \n{e}"
        ) from e
    testv = np.array(testv)
    if np.issubdtype(testv.dtype, np.number):
        if testv.ndim > 1:
            raise ValueError(
                f"Given custom measure function \n'{cfunc_name}' \nreturned result shape {testv.shape}, expected (1,)"
            )
    else:
        raise TypeError(
            f"Given custom measure function \n'{cfunc_name}' \nmust return numbers, got datatype: {testv.dtype}"
        )
    return cfunc


# %% Common statistical measures for 2D array-like


# Common statistical measures for 2D array-like
class Stats2d:
    """
    Statistical measure calculator for a 2D data array with each row or column as a sample.

    For paralell processing, please use the following static methods:
        - mean / std / skew / kurt / minimal / median / maximal
    Or use functionn moment2d for arbitrary moments instead.

    Attributes
    ----------
    axis : int, optional
        Axis along which the moment is computed, axis=0 treats each row as a sample. The default is 0.
    measure : str or tuple of name and Callable or list of them, optional
        - Common measure can be specified by name, choose between "mean", "std", "skewness", "kurtosis", "min", "median" and "max".
        - Custom measure function should accept 2D array as its only argment, where each row represents a sample.
        - Multiple measures can be specified in a list or tuple.
        If not given, all aforementioned common measures are computed by default.
    """  # noqa: E501

    @simple_type_validator
    def __init__(self, measure: Union[str, Callable, list[Union[str, Callable]], None] = None, axis: int = 0) -> None:
        self.measure = measure
        self.axis = axis

    # Simplified measures
    @staticmethod
    @simple_type_validator
    def mean(data_array_2d: Annotated[Any, arraylike_validator(ndim=2)], axis: int = 0) -> np.ndarray:
        """
        Mean values of a 2D data array with each row or column as a sample. See stats2d.
        """
        result: np.ndarray = np.array(np.nanmean(data_array_2d, axis=axis))
        return result

    @staticmethod
    @simple_type_validator
    def std(data_array_2d: Annotated[Any, arraylike_validator(ndim=2)], axis: int = 0) -> np.ndarray:
        """
        Standard deviation values of a 2D data array with each row or column as a sample. See stats2d.
        """
        result: np.ndarray = np.array(np.nanstd(data_array_2d, axis=axis))
        return result

    @staticmethod
    @simple_type_validator
    def var(data_array_2d: Annotated[Any, arraylike_validator(ndim=2)], axis: int = 0) -> np.ndarray:
        """
        Variance values of a 2D data array with each row or column as a sample. See stats2d.
        """
        result: np.ndarray = np.array(np.nanvar(data_array_2d, axis=axis))
        return result

    @staticmethod
    @simple_type_validator
    def skew(data_array_2d: Annotated[Any, arraylike_validator(ndim=2)], axis: int = 0) -> np.ndarray:
        """
        Skewness values of a 2D data array with each row or column as a sample. See stats2d.
        """
        result: np.ndarray = np.array(moment2d(data_array_2d, 3, standardized=True, axis=axis))
        return result

    @staticmethod
    @simple_type_validator
    def kurt(data_array_2d: Annotated[Any, arraylike_validator(ndim=2)], axis: int = 0) -> np.ndarray:
        """
        Kurtosis values of a 2D data array with each row or column as a sample. See stats2d.
        """
        result: np.ndarray = np.array(moment2d(data_array_2d, 4, standardized=True, axis=axis))
        return result

    @staticmethod
    @simple_type_validator
    def minimal(data_array_2d: Annotated[Any, arraylike_validator(ndim=2)], axis: int = 0) -> np.ndarray:
        """
        Min values of a 2D data array with each row or column as a sample. See stats2d.
        """
        result: np.ndarray = np.array(np.nanmin(data_array_2d, axis=axis))
        return result

    @staticmethod
    @simple_type_validator
    def median(data_array_2d: Annotated[Any, arraylike_validator(ndim=2)], axis: int = 0) -> np.ndarray:
        """
        Median values of a 2D data array with each row or column as a sample. See stats2d.
        """
        result: np.ndarray = np.array(np.nanmedian(data_array_2d, axis=axis))
        return result

    @staticmethod
    @simple_type_validator
    def maximal(data_array_2d: Annotated[Any, arraylike_validator(ndim=2)], axis: int = 0) -> np.ndarray:
        """
        Max values of a 2D data array with each row or column as a sample. See stats2d.
        """
        result: np.ndarray = np.array(np.nanmax(data_array_2d, axis=axis))
        return result

    # Return stats values only, single measure only
    @simple_type_validator
    def _values(self, data_array_2d: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Measure values of a 2D data array with each row or column as a sample. See stats2d.
        This function accepts single measures only.
        """
        measure = self.measure
        if type(measure) is str:
            result: np.ndarray = np.array(Stats2d(measure).stats2d(data_array_2d)[measure])
            return result
        elif callable(measure):
            result = np.array(Stats2d(measure).stats2d(data_array_2d)[measure.__name__])
            return result
        elif measure is None:
            raise ValueError("'measure' must be specified for 'Stats2d' when calling this method, got None.")
        else:
            raise ValueError(f"'Stats2d.values' accepts single measure only, but got multiple measures: {measure}")

    # Return stats values with dynamic function name of the measure
    @property
    def values(self) -> Callable:
        def wrapper(*args, **kwargs) -> np.ndarray:  # type: ignore[no-untyped-def]
            result: np.ndarray = np.array(self._values(*args, **kwargs))
            return result

        # Set dynamic name
        if callable(self.measure):
            dname = f"{self.measure.__name__}_values"
        else:
            dname = str(self.measure) + "_values"
        wrapper.__name__ = dname
        wrapper.__qualname__ = f"{self.__class__.__name__}.{dname}"
        return wrapper

    # Stats function, able to return multiple measures at one time
    @simple_type_validator
    def stats2d(self, data_array_2d: Annotated[Any, arraylike_validator(ndim=2)]) -> dict[str, np.ndarray]:
        """
        Statistical measure calculator for a 2D data array with each row or column as a sample.

        Parameters
        ----------
        data_array_2d : ndarray-like
            2D array-like data. Nan value is omited.

        Returns
        -------
        mvalues : dict[str, np.ndarray]
            A dictionary of computed measure values, where:
                - **Keys** (str): Names of the measures, e.g. "mean"
                - **Values** (np.ndarray): Results for each measure
        """
        measure = self.measure
        axis = self.axis

        # Standardize axis
        arr = axisconv(data_array_2d, axis, float)

        # Validate sample size
        nsample = arr.shape[0]
        if nsample < 5:
            raise ValueError(f"Sample size must be at least 5, got: {nsample}")

        # Stats configuration
        if measure is not None:
            if (type(measure) is str) or callable(measure):
                measure = [measure]
            if (type(measure) is list) or (type(measure) is tuple):
                mnames = []
                mfuncs = []
                for ms in measure:
                    if type(ms) is str:
                        mnames.append(ms)
                        mfuncs.append(smopt(ms))
                    elif callable(ms):
                        mnames.append(ms.__name__)
                        # Test run
                        ms = cmval(arr, ms)
                        mfuncs.append(ms)
                    else:
                        raise TypeError(
                            f"Type of measure must be string or Callable, got type: {type(ms)} for measure {ms}"
                        )
            else:
                raise TypeError(
                    f"Type of measure must be string or Callable or list of them, got type: {type(measure)}"
                )
        else:
            mnames = ["mean", "std", "skewness", "kurtosis", "min", "median", "max"]
            mfuncs = [smopt(mn) for mn in mnames]

        # Computing measures
        mvalues: dict = {}
        for mn, mfunc in zip(mnames, mfuncs):
            mvalues[mn] = mfunc(arr)

        return mvalues


def roi_mean(image_path: str, roi_coordinates: list[list[tuple[Union[int, float], Union[int, float]]]]) -> np.ndarray:
    """
    Spectral mean of the given image within the given ROI.

    Parameters
    ----------
    image_path : str
        Spectral raster image path.

    roi_coordinates : list[list[tuple[Union[int, float],Union[int, float]]]]
        Lists of vertex coordinate pairs of the polygons of a region of interest (ROI).
    """
    result: np.ndarray = np.array(Stats2d().mean(ROISpec("float64").roispec(image_path, roi_coordinates)))
    return result


def roi_std(image_path: str, roi_coordinates: list[list[tuple[Union[int, float], Union[int, float]]]]) -> np.ndarray:
    """
    Standard deviation of the given image within the given ROI.

    Parameters
    ----------
    image_path : str
        Spectral raster image path.

    roi_coordinates : list[list[tuple[Union[int, float],Union[int, float]]]]
        Lists of vertex coordinate pairs of the polygons of a region of interest (ROI).
    """
    result: np.ndarray = np.array(Stats2d().std(ROISpec("float64").roispec(image_path, roi_coordinates)))
    return result


def roi_median(image_path: str, roi_coordinates: list[list[tuple[Union[int, float], Union[int, float]]]]) -> np.ndarray:
    """
    Spectral median of the given image within the given ROI.

    Parameters
    ----------
    image_path : str
        Spectral raster image path.

    roi_coordinates : list[list[tuple[Union[int, float],Union[int, float]]]]
        Lists of vertex coordinate pairs of the polygons of a region of interest (ROI).
    """
    result: np.ndarray = np.array(Stats2d().median(ROISpec("float64").roispec(image_path, roi_coordinates)))
    return result


# %% ROI spectral angle tools


# Spectral angle calculator
def spectral_angle(
    spec_1: Annotated[Any, arraylike_validator(ndim=1)],
    spec_2: Annotated[Any, arraylike_validator(ndim=1)],
    invalid_raise: bool = False,
) -> float:
    """
    Calculate spectral angle of two spectra in 1D-array-like.

    Parameters
    ----------
    spec_1 : list, series or 1-d numpy.array
        List of spectra.
    spec_2 : list, series or 1-d numpy.array
        List of spectra.
    invalid_raise : bool
        Whether error is raised if spectral angle is invalid for given spectra, if False, np.nan is returned.
        The default is False.

    Returns
    -------
    spectral_angle : float
        Spectral angle in radian.

    Raises
    ------
    ValueError
        If all values of a spectral vector equals zero.
    """
    # Convert lists to numpy arrays
    spec_1 = np.array(spec_1)
    spec_2 = np.array(spec_2)

    spec_angle: float

    # Validate spectral vectors
    if np.all(spec_1 == np.zeros(spec_1.shape)) or np.all(spec_2 == np.zeros(spec_2.shape)):
        if not invalid_raise:
            spec_angle = float(np.nan)
            return spec_angle
        else:
            raise ValueError(f"Undefined spectral angle for given spectral vector: \n{spec_1} \nand \n{spec_2}")

    # Calculate the dot product
    dot_product = np.dot(spec_1, spec_2)

    # Calculate the magnitudes
    magnitude_1 = np.linalg.norm(spec_1)
    magnitude_2 = np.linalg.norm(spec_2)

    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_1 * magnitude_2)

    # Calculate the spectral angle in radians
    spec_angle = float(np.arccos(cos_theta))

    return spec_angle


# %% Spectral angle of spectra in 2d array to a reference spectrum


# df spec angle
def arr_spectral_angles(
    spectra_array: Annotated[Any, arraylike_validator(ndim=2)],
    reference_spectrum: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
    invalid_raise: bool = False,
) -> np.ndarray:
    """
    Calculate spectra angles of a 2D-array-like of spectra to a reference spectrum.

    Parameters
    ----------
    spectra_array : 2-dimensional array like
        2d array like of 1d spectral data series.
    reference_spectrum : 1-dimensional array like
        Reference spectrum to calculate all spectral angles between the spectra and the reference.
    axis : int, optional
        If axis = 0, each row data represents a spectrum, if axis = 1, each column data represents a spectrum.
        The default is 0.
    invalid_raise : bool
        Whether error is raised if spectral angle is invalid for a given spectrum, if False, np.nan is returned.
        The default is False.

    Returns
    -------
    spec_angle : numpy.ndarray
        Spectral angles between given spectra and the reference spectrum in 1D array.

    Raises
    ------
    ValueError
        If input spectra_array is not 2d array like.
    ValueError
        If input spectra_array contains nan values.
    ValueError
        Axis can only be 0 or 1.
    ValueError
        If input spectra_array does not match with reference_spectrum.
    """
    # Data validation
    spectra_array = np.array(spectra_array)
    if spectra_array.ndim != 2:
        raise ValueError("input spectra_array should be 2d array like.")
    if np.isnan(spectra_array).any():
        raise ValueError("input spectra_array should not contain nan values.")
    if axis == 0:
        pass
    elif axis == 1:
        spectra_array = spectra_array.T
    else:
        raise ValueError("axis can only be 0 or 1.")
    reference_spectrum = np.array(reference_spectrum)
    if len(reference_spectrum) != spectra_array.shape[1]:
        raise ValueError("input spectra_array does not match with reference_spectrum.")

    # Calculating spec angles
    npix = spectra_array.shape[0]
    spec_angle: np.ndarray = np.zeros(npix)
    for i in range(npix):
        spec_angle[i] = spectral_angle(reference_spectrum, spectra_array[i, :], invalid_raise=invalid_raise)
    return spec_angle


# %% Round to significant numbers


class RealNumberMeta(type(Protocol)):  # type: ignore[misc]
    def __instancecheck__(cls, instance: Any) -> bool:
        # Exclude numpy arrays
        if hasattr(instance, '__len__'):
            return False
        # Include RealNumber
        return hasattr(instance, '__mul__') and hasattr(instance, '__lt__')


@runtime_checkable
class RealNumber(Protocol, metaclass=RealNumberMeta):
    def __mul__(self, v: Any) -> Any: ...
    def __lt__(self, v: Any) -> bool: ...


def num_sig_digit(v: RealNumber, sig_digit: int, mode: str = "round") -> float:
    """
    Round a value to given significant digits. Choose mode between 'round' / 'ceil' / 'floor'.
    """
    if v == 0:
        return 0.0
    factor = 10 ** (sig_digit - 1 - math.floor(math.log10(abs(v))))  # type: ignore[arg-type]
    # unrecognized user-defined RealNumber type
    if mode == "round":
        rdfunc = round
    elif mode == "ceil":
        rdfunc = math.ceil  # type: ignore[assignment]
        # overloaded expression function
    elif mode == "floor":
        rdfunc = math.floor  # type: ignore[assignment]
        # overloaded expression function
    else:
        raise ValueError(f"mode must one of 'round', 'ceil' and 'floor', but got: '{mode}'")
    return float(rdfunc(v * factor) / factor)


def np_sig_digit(arr_like: Annotated[Any, arraylike_validator()], sig_digit: int, mode: str = "round") -> np.ndarray:
    """
    Round values to given significant digits for numeric NumPy arrays. Choose mode between 'round' / 'ceil' / 'floor'.
    """
    arr = np.asarray(arr_like)
    with np.errstate(divide="ignore", invalid="ignore"):
        log10 = np.where(arr == 0, 0, np.floor(np.log10(np.abs(arr))))
    factor = 10 ** (sig_digit - 1 - log10)
    if mode == "round":
        rdfunc = np.round  # type: ignore[assignment]
        # overloaded expression function, following the same
    elif mode == "ceil":
        rdfunc = np.ceil  # type: ignore[assignment]
    elif mode == "floor":
        rdfunc = np.floor  # type: ignore[assignment]
    else:
        raise ValueError(f"mode must one of 'round', 'ceil' and 'floor', but got: '{mode}'")
    result: np.ndarray = np.array(rdfunc(arr_like * factor) / factor)
    return result


# For any scalar real number regardless of types - behavioral validation
@overload
def round_digit(value: RealNumber, sig_digit: int, mode: str) -> float: ...  # type: ignore[overload-cannot-match]


# Mypy failure on GitHub, the code works and passes local mypy validation, following the same


@overload
def round_digit(value: tuple, sig_digit: int, mode: str) -> tuple: ...  # type: ignore[overload-cannot-match]


@overload
def round_digit(value: list, sig_digit: int, mode: str) -> list: ...  # type: ignore[overload-cannot-match]


@overload
def round_digit(value: np.ndarray, sig_digit: int, mode: str) -> np.ndarray: ...  # type: ignore[overload-cannot-match]


@overload
def round_digit(  # type: ignore[overload-cannot-match]
    value: torch.Tensor, sig_digit: int, mode: str
) -> torch.Tensor: ...


@overload
def round_digit(  # type: ignore[overload-cannot-match]
    value: pd.DataFrame, sig_digit: int, mode: str
) -> pd.DataFrame: ...


@simple_type_validator
def round_digit(
    value: Union[RealNumber, Annotated[Any, arraylike_validator()]],
    sig_digit: int,
    mode: str = "round",
) -> Union[float, tuple, list, np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]:
    """
    Round values or values in arraylike to specified significant digits.

    Parameters
    ----------
    value : Union[float, list, tuple, np.ndarray, pd.DataFrame]
        Number or array-like values, e.g. Numpy ndarray or pd.DataFrame.
    sig_digit : int
        Significant digits.
    mode : str, optional
        Choose between 'round' / 'ceil' / 'floor'. The default is 'round'.

    Returns
    -------
    result : Union[float, np.ndarray]
        Rounded value or array of values.
    """
    if isinstance(value, (list, tuple, np.ndarray, pd.Series, pd.DataFrame, torch.Tensor)):
        value1 = np.array(value)
        result = np_sig_digit(value1, sig_digit, mode)
        if isinstance(value, (np.ndarray, pd.Series, pd.DataFrame)):
            value[:] = result[:]
        elif isinstance(value, list):
            value = result.tolist()
        elif isinstance(value, torch.Tensor):
            value = torch.tensor(result)
        elif value1.ndim == 1:
            value = tuple(result)
        else:
            value = tuple(tuple(map(tuple, result.tolist())))  # noqa: C414
            # The double tuples are necessary
        return value
    else:
        return num_sig_digit(value, sig_digit, mode)
