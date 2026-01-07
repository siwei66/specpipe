# -*- coding: utf-8 -*-
"""
Example data generators and downloader for SpecPipe

The docstrings were created with AI assistance and has been human reviewed and edited.

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# OS
import os

# Time
import time

# Typing
from typing import Optional, Union

# Basic data processing
import numpy as np
import pandas as pd

# Rasters
import rasterio

# Local
from .specexp import SpecExp
from .specio import roi_to_envi, silent, simple_type_validator

# Download
import urllib


# %% test helper functions


# Test helper functions : create_test_raster
@simple_type_validator
def create_test_raster(
    raster_path: str,
    width: int = 100,
    height: int = 50,
    bands: int = 4,
    incl_nodata: Union[None, int, float] = None,
    nodata_value: Union[int, float] = 0,
    dtype: Union[str, type] = "uint16",
    data: Optional[np.ndarray] = None,
) -> str:
    """
    Create a synthetic raster file for demonstration or testing purposes.

    Generates a GeoTIFF raster with specified dimensions, number of bands, and data type.
    Custom data can also be provided.
    Optional nodata values can be applied to simulate missing data.

    By default, it creates a 100x50 raster with 4 bands containing deterministic pseudo-random data.

    Parameters
    ----------
    raster_path : str
        Path to save the raster. Must end with ``.tif`` or ``.tiff``.
        The directory must exist.

    width : int, optional
        Raster width in pixels. Default is 100.

    height : int, optional
        Raster height in pixels. Default is 50.

    bands : int, optional
        Number of bands. Default is 4.

    incl_nodata : int, float, or None, optional
        Value to fill selected bands to simulate nodata regions. Default is ``None``.

    nodata_value : int or float, optional
        Metadata nodata value. Default is 0.

    dtype : str or type, optional
        Data type of raster array. Default is ``'uint16'``.

    data : numpy.ndarray, optional
        Custom raster data of shape ``(bands, height, width)``.

        If ``None``, synthetic data is generated. Default is `None`.

    Returns
    -------
    str
        Path of the created raster file.

    Notes
    -----
    - The raster uses a simple linear gradient with pseudo-random noise.
    - No real CRS is assigned; transform is based on pixel coordinates.
    - If `incl_nodata` is set, first and last bands (if >3) are filled with this value.

    Examples
    --------
    Generate a default test raster::

        >>> create_test_raster("test_raster.tif")

    Generate a raster with custom dimensions and 3 bands::

        >>> create_test_raster("custom_raster.tif", width=200, height=100, bands=3)

    Provide custom data for the raster::

        >>> import numpy as np
        >>> data = np.random.randint(0, 256, size=(4, 50, 100), dtype='uint16')
        >>> create_test_raster("custom_data_raster.tif", data=data)
    """

    # Validate raster path
    raster_path = raster_path.replace("\\", "/").replace("//", "/")
    if (raster_path[-4:] != ".tif") and (raster_path[-5:] != ".tiff"):
        raise ValueError(f"raster_path must have .tif or .tiff extension, got: {raster_path}")
    if not os.path.exists(os.path.dirname(raster_path)):
        raise ValueError(f"raster path directory does not exist: {os.path.dirname(raster_path)}")

    # Validate raster data
    if data is None:
        # Mock image data
        np.random.seed(42)
        data = np.array(
            [
                [
                    (
                        np.array([1080 / bands * j + 360 / bands * int(i / 5)] * width)
                        + (np.random.rand(width) * 120 / bands)
                    )
                    .astype(dtype)
                    .tolist()
                    for i in range(height)
                ]
                for j in range(bands)
            ]
        )
    elif type(data) is np.ndarray:
        data = np.array(data).astype(dtype)
        if data.shape != (bands, height, width):
            raise ValueError(
                f"Given raster data shape {data.shape} does not match specified raster dimensions (bands={bands}, \
                    height={height}, width={width})."
            )
    else:
        raise ValueError(f"Given raster data must be numpy.ndarray, got type: {type(data)}.")

    # Set some nodata values
    if incl_nodata is not None:
        data[0, :, :] = incl_nodata
        if bands > 3:
            data[-1, :, :] = incl_nodata

    # Define transform (georeferencing)
    transform = rasterio.transform.from_bounds(0, 0, width, height, width=width, height=height)

    # Create metadata
    meta = {
        "driver": "GTiff",
        "dtype": dtype,
        "nodata": nodata_value,
        "width": width,
        "height": height,
        "count": bands,
        "crs": None,
        "transform": transform,
    }

    # Write to file
    with rasterio.open(raster_path, "w", **meta) as dst:
        dst.write(data)

    return raster_path


# Alias
create_example_raster = create_test_raster


# Test helper functions : create_test_roi_xml
@simple_type_validator
def create_test_roi_xml(
    xml_path: str,
    raster_width: int = 10,
    raster_height: int = 50,
    roi_count: int = 10,
    roi_list: Optional[list[dict]] = None,
    return_path: bool = True,
    return_roi_list: bool = False,
) -> Union[str, list[dict], tuple[str, list[dict]], None]:
    """
    Create a synthetic ENVI ROI XML file for demonstration or testing purposes.

    Generates an ENVI-compatible ROI XML file containing a set of rectangular regions of interest (ROIs).
    Custom ROIs can be provided as a list of dictionaries.

    By default, it creates 10 ROIs evenly distributed along the raster height.

    Parameters
    ----------
    xml_path : str
        Path to save the ROI XML file.

    raster_width : int, optional
        Width of the raster used to define ROI coordinates. Default is 10.

    raster_height : int, optional
        Height of the raster used to define ROI coordinates. Default is 50.

    roi_count : int, optional
        Number of synthetic ROIs to generate if ``roi_list`` is not provided. Default is 10.

    roi_list : list of dict, optional
        Custom list of ROI dictionaries.

        Each dictionary must contain the following keys::

            {
                "name": str,
                "crs": str or CRS,
                "color": tuple of int or None,
                "type": str,
                "coordinates": list of list of tuple of float
            }

        If ``None``, synthetic ROIs are generated. Default is ``None``.

    return_path : bool, optional
        If True, return the path of the generated XML file. Default is True.

    return_roi_list : bool, optional
        If True, return the list of ROI dictionaries. Default is False.

    Returns
    -------
    str, list of dict, (str, list of dict), or None

        - If ``return_path`` is True and ``return_roi_list`` is False: returns the XML file path (str).
        - If ``return_path`` is False and ``return_roi_list`` is True: returns the ROI list (list of dict).
        - If both are True: returns a tuple ``(path, roi_list)``.
        - If both are False: returns None.

    Notes
    -----
    Coordinates are defined in raster pixel units, and CRS is set to ``"none"`` by default.
    This method is also available as ``create_example_roi_xml``.

    Examples
    --------
    Generate a default ROI XML file::

        >>> create_test_roi_xml("test_rois.xml")

    Generate a file with 5 ROIs for a raster of height 50::

        >>> create_test_roi_xml("small_rois.xml", raster_height=50, roi_count=5)

    Generate a file and also get the ROI list::

        >>> xml_path, rois = create_test_roi_xml("example_rois.xml", return_roi_list=True)

    Provide a custom ROI list::

        >>> custom_rois = [
            ...     {"name": "ROI_1", "type": "polygon", "coordinates": [[(0, 0), (5, 0), (0, 5), (0, 0)]]}
            ... ]
        >>> create_test_roi_xml("custom_rois.xml", roi_list=custom_rois)
    """

    # Create line of mock ROIs in the raster
    height_increment = int(raster_height / roi_count)
    if height_increment < 1:
        raise ValueError("raster_height / roi_count must be at least 1.")
    width_range = min(5, raster_width)

    if roi_list is None:
        roi_list = []
        # Test ROI data
        for i in range(roi_count):
            roi_dict = {
                "name": f"ROI_{i + 1}",
                "crs": "none",
                "color": None,
                "type": "polygon",
                "coordinates": [
                    [
                        (0.0, height_increment * i),
                        (width_range, height_increment * i),
                        (width_range, height_increment * (i + 1)),
                        (0.0, height_increment * (i + 1)),
                        (0.0, height_increment * i),
                    ]
                ],
            }
            roi_list.append(roi_dict)

    # Save ROI xml file
    path_out = roi_to_envi(file_path=xml_path, roi_list=roi_list)
    path_out = str(path_out)

    # Output
    result: Union[str, list[dict], tuple[str, list[dict]]]
    if return_path and not return_roi_list:
        result = path_out
    elif not return_path and return_roi_list:
        result = roi_list
    elif return_path and return_roi_list:
        result = path_out, roi_list
    else:
        return None
    return result


# Alias
create_example_roi_xml = create_test_roi_xml


# Test helper functions : create_test_spec_exp
@silent
@simple_type_validator
def create_test_spec_exp(dir_path: str, sample_n: int = 10, n_bands: int = 4, is_regression: bool = True) -> SpecExp:
    """
    Create a standard test `SpecExp` instance for spectral experiments.

    This function generates a `SpecExp` data manager pre-populated with:

    - A test group.
    - A synthetic multispectral image.
    - Synthetic ROIs.
    - Synthetic sample labels and targets.

    It can be used for testing spectral processing pipelines, demonstration or development.

    Parameters
    ----------
    dir_path : str
        Directory where the test data (images and ROI XML) will be stored.
        Will be created if it does not exist.

    sample_n : int, optional
        Number of samples / ROIs to generate. Default is 10.

    n_bands : int, optional
        Number of bands in the synthetic image. Default is 4.

    is_regression : bool, optional
        If True, target values are numeric (regression).
        If False, targets are categorical (classification).
        Default is True.

    Returns
    -------
    SpecExp
        A ``SpecExp`` instance populated with synthetic data, ready for spectral analysis.

    Notes
    -----
    - The synthetic image is a pseudo-random gradient with band-specific variations.
    - ROIs are rectangular and evenly distributed across the raster height.
    - Sample labels and target values are automatically generated based on ``sample_n``.

    Examples
    --------
    Create a default test ``SpecExp`` instance::

        >>> exp = create_test_spec_exp("test_spec_exp")

    Create a test instance with 20 samples and 6 spectral bands::

        >>> exp = create_test_spec_exp("test_spec_exp_dir", sample_n=20, n_bands=6)

    Create a classification-type ``SpecExp`` instance::

        >>> exp = create_test_spec_exp("class_spec_exp", is_regression=False)
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    exp1: SpecExp = SpecExp(dir_path, log_loading=False)

    # # Add test
    # group
    exp1.add_groups(["test_group"])

    # Mock image data
    np.random.seed(42)
    mock_img_data = np.array(
        [
            [
                (np.array([1080 / n_bands * j + 360 / n_bands * i] * 10) + (np.random.rand(10) * 120 / n_bands))
                .astype(np.uint16)
                .tolist()
                for i in range(5 * sample_n)
            ]
            for j in range(n_bands)
        ]
    )

    for i in range(5 * sample_n):
        mock_img_data[-int(n_bands / 3) :, i, :] = mock_img_data[-int(n_bands / 3) :, i, :] * (i // 5 % 2 + 0.5)

    # Mock image
    img_path = dir_path + "/test_img.tif"
    create_test_raster(
        raster_path=img_path,
        width=mock_img_data.shape[2],
        height=mock_img_data.shape[1],
        bands=n_bands,
        data=mock_img_data,
    )
    exp1.add_images_by_name(group="test_group", image_name=["test_img.tif"], image_directory=dir_path)

    # ROIs
    roi_path = dir_path + "/test_roi.xml"
    create_test_roi_xml(roi_path, roi_count=sample_n)
    exp1.add_rois_by_file(group="test_group", path=[roi_path], image_name="test_img.tif")

    # Samples
    dflb = exp1.ls_sample_labels()
    assert isinstance(dflb, pd.DataFrame)
    dflb.iloc[:, 1] = [f"sample_{str(i + 1)}" for i in range(len(dflb))]
    exp1.sample_labels = dflb  # type: ignore[assignment]
    # Auto-conversion in setter

    # Target values
    dft = exp1.ls_sample_targets()
    assert isinstance(dft, pd.DataFrame)
    if is_regression:
        dft["Target_value"] = list(range(len(dft)))
    else:
        dft["Target_value"] = [["a", "b"][int(i % 2)] for i in range(len(dft))]
    exp1.sample_targets_from_df(dft)

    return exp1


# Alias
create_example_spec_exp = create_test_spec_exp


# %% Demo data downloader


@simple_type_validator
def download_demo_data(
    data_dir: str = os.getcwd(),
    demo_dir_url: str = 'https://raw.githubusercontent.com/siwei66/specpipe/master/demo/demo_data/',
    files: tuple = ('demo_1.xml', 'demo_2.xml', 'demo_3.xml', 'demo_4.xml', 'demo_5.xml', 'demo.tiff'),
    retry_limit: int = 5,
) -> None:
    """
    Download real-world demo data files for demonstration or user testing purposes.

    Parameters
    ----------
    data_dir : str, optional
        Local directory where downloaded files will be saved.

        Default is the current working directory (``os.getcwd()``).

    demo_dir_url : str, optional
        Base URL of the remote demo data repository of ``SpecPipe`` package.

        Default is ``'https://raw.githubusercontent.com/siwei66/specpipe/master/demo/demo_data/'``.

    files : tuple of str, optional
        Names of files to download from the repository.

        Default includes ``('demo_1.xml', 'demo_2.xml', 'demo_3.xml', 'demo_4.xml', 'demo_5.xml', 'demo.tiff')``.
        The files must exist in the remote repository.

    retry_limit : int, optional
        Maximum number of download retries per file in case of failure.
        Default is 5.

    Returns
    -------
    None
        The files are downloaded to ``data_dir``. No value is returned.

    Examples
    --------
    Download default demo files to the current directory::

        >>> download_demo_data()

    Download demo files to a custom directory::

        >>> download_demo_data(data_dir="my_demo_data")

    Download only a subset of files::

        >>> download_demo_data(files=("demo_1.xml", "demo.tiff"))
    """

    # Validate retry limit
    retry_limit = min(3, retry_limit)
    # Setup download directory
    data_dir = (data_dir.replace('\\', '/') + '/').replace('//', '/')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Package demo dir path
    downloaded = []
    for file in files:
        url = demo_dir_url + file
        local_path = data_dir + file
        i = 0
        while i < retry_limit:
            try:
                urllib.request.urlretrieve(url, local_path)
                # Verify download
                if os.path.exists(local_path):
                    downloaded.append(file)
                    print(f"\nDownloaded: {local_path}")
                    break
            except Exception as e:
                i = i + 1
                print(f"\nDownload file '{file}' failed: {e}, \nRetry {i} in {2 ** i} seconds...")
                time.sleep(min(2**i, 16))
                if i >= retry_limit:
                    print(
                        f"\nDownload '{file}' failed, please download it manually from:\
                          https://github.com/siwei66/specpipe/tree/master/demo/demo_data/"
                    )
    if set(downloaded) != set(files):
        raise Exception(
            "Download is not completed, plaese download the files manually from: \
                https://github.com/siwei66/specpipe/tree/master/demo/demo_data/"
        )
    return None
