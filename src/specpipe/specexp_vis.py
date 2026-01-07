# -*- coding: utf-8 -*-
"""
SpecPipe - SpecExp data visualization

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# OS
import os

# Typing
from typing import Optional, Union

# Basic data
import numpy as np

# Local
from .specio import simple_type_validator, unc_path

# Raster
import rasterio

# Visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PltPolygon


# %% Image RGB display with ROI marks


@simple_type_validator
def raster_rgb_preview(  # noqa: C901
    raster_path: str,
    rgb_band_index: tuple[int, int, int],
    display_size: tuple[Union[int, float], Union[int, float]],
    roi_coords: Optional[list[list[tuple[Union[int, float], Union[int, float]]]]] = None,
    roi_color: Union[str, tuple[float, float, float]] = "red",
    roi_linewidth: Union[int, float] = 3,
    *,
    normalize: bool = True,
    output_path: Optional[str] = None,
    dpi: int = 150,
    show_image: bool = True,
) -> None:
    """
    Display a raster image in RGB plot with optional ROI frames.

    Parameters
    ----------
    raster_path : str
        Raster image path.
    rgb_band_index : tuple of 3 ints
        RGB band index in tuple.
    display_size : tuple of 2 numbers
        Size of RGB plot.
    roi_coords : list of lists of tuples of 2 numbers or None, optional
        ROI vertex coordinate pairs in tuples in a list.
    roi_color : str or tuple of 3 floats, optional
        ROI color in name string or RGB values.
    roi_linewidth : int, optional
        Line with of ROI polygon edges. The default is 3.
    normalize : bool, optional
        Whether normalize RGB band values to the range of 0~255. The default is True.
    output_path : str or None, optional
        Save RGB preview image to the path, if None, no file output. The default is None.
    dpi : int, optional
        DPI of the RGB preview image. The default is 150.

    Examples
    --------
    Generate RGB preview::

        >>> raster_rgb_preview("image.tif", rgb_band_index=(192, 124, 64))

    Generate preview and save preview image to file::

        >>> raster_rgb_preview("image.tif", rgb_band_index=(192, 124, 64), output_path="/image_rgb.png")

    Preview with ROI frames::

        >>> raster_rgb_preview(
        ... "image.tif",
        ... rgb_band_index=(192, 124, 64),
        ... roi_coords=[
        ...     [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        ...     [(30, 30), (30, 40), (40, 30), (30, 30)],
        ... ])
    """
    # Validate path
    if not os.path.exists(unc_path(raster_path)):
        raise ValueError(f"Invalid image path: {raster_path}")
    raster_dir, raster_name = os.path.split(raster_path)

    # Read image
    try:
        with rasterio.open(unc_path(raster_path)) as src:
            bands = src.count
            for bid in rgb_band_index:
                if bid > bands or bid < 1:
                    raise ValueError(f"Band index must be in the range of 1 ~ {bands}.")
            red = src.read(rgb_band_index[0])
            green = src.read(rgb_band_index[1])
            blue = src.read(rgb_band_index[2])
            print(f"\nImage dimensions: {red.shape}")
            print(f"\nData type: {red.dtype}")
    except Exception as e:
        raise ValueError(f"Error reading image: {e}") from e

    rgb_img = np.dstack((red, green, blue))

    # Normalize for display if needed
    if rgb_img.dtype != np.uint8:
        rgb_normalized = (rgb_img / rgb_img.max() * 255).astype(np.uint8)
    else:
        rgb_normalized = rgb_img

    # Display RGB image
    plt.figure(figsize=(display_size[0], display_size[1]))

    # Plot image
    plt.imshow(rgb_normalized)

    # Plot ROIs
    if roi_coords is not None:
        for poly_i in roi_coords:
            if len(poly_i) < 3:
                raise ValueError(f"The number of ROI polygon vertices must be at least 3, got: {len(poly_i)}")
            polygon = PltPolygon(
                poly_i,
                fill=False,
                edgecolor=roi_color,
                linewidth=roi_linewidth,
                linestyle='-',
            )
            plt.gca().add_patch(polygon)

    # Show plot title and axis
    # plt.title(f"RGB preview of {raster_name}")
    plt.axis('off')

    # Save image
    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(unc_path(output_dir)):
            raise ValueError(f"Output directory not found: {output_dir}")
        output_path = os.path.splitext(output_path)[0] + ".png"
        plt.savefig(unc_path(output_path), dpi=dpi, bbox_inches='tight', pad_inches=0)
        print(f"RGB image saved as: {output_path}")

    # Show plot and clear plt
    if show_image:
        plt.show()
    else:
        plt.close()
