# -*- coding: utf-8 -*-
"""
Tests for data management module of spectral experiment (SpecExp)

The following source code was created with AI assistance and has been human reviewed and edited.

Copyright (c) 2025 Siwei Luo. MIT License.
"""

import os
import shutil

import tempfile
import unittest

import pandas as pd
import pytest

from typing import Union

# Self
# Funcs to test
from specpipe.specexp import SpecExp

# Applied package functions for test
from specpipe.specio import silent

# %% Helper functions for testing


# Create mock files for testing
def create_test_directory_structure(base_dir: str) -> tuple[str, str]:
    """Create test directories and files for testing"""
    # Create test images directory
    images_dir = os.path.join(base_dir, "test_images")
    os.makedirs(images_dir, exist_ok=True)

    # Create test ROI files directory
    rois_dir = os.path.join(base_dir, "test_rois")
    os.makedirs(rois_dir, exist_ok=True)

    # Create some dummy image files
    image_files = ["image1.tif", "image2.tif", "image3.tif", "mask1.tif", "mask2.tif"]

    for img_file in image_files:
        with open(os.path.join(images_dir, img_file), "w") as f:
            f.write("dummy image content")

    # Create some dummy ROI files (simplified versions)
    roi_files = ["image1_rois.xml", "image2_rois.xml", "image1_mask.shp"]

    for roi_file in roi_files:
        with open(os.path.join(rois_dir, roi_file), "w") as f:
            if roi_file.endswith(".xml"):
                f.write(
                    """<?xml version="1.0" encoding="UTF-8"?>
<RegionsOfInterest version="1.1">
  <Region name="TestROI1" color="255,0,0">
    <GeometryDef>
      <CoordSysStr>none</CoordSysStr>
      <Polygon>
        <Exterior>
          <LinearRing>
            <Coordinates>100.0 200.0 300.0 400.0 500.0 600.0 100.0 200.0</Coordinates>
          </LinearRing>
        </Exterior>
      </Polygon>
    </GeometryDef>
  </Region>
  <Region name="TestROI2" color="0,255,0">
    <GeometryDef>
      <CoordSysStr>none</CoordSysStr>
      <Polygon>
        <Exterior>
          <LinearRing>
            <Coordinates>200.0 200.0 300.0 300.0 500.0 500.0 200.0 200.0</Coordinates>
          </LinearRing>
        </Exterior>
      </Polygon>
    </GeometryDef>
  </Region>
</RegionsOfInterest>"""
                )
            elif roi_file.endswith(".shp"):
                # Simplified shapefile content
                f.write("dummy shapefile content")

    return images_dir, rois_dir


# %% test functions : SpecExp
class TestSpecExp(unittest.TestCase):
    """Test class for SpecExp functionality"""

    def setUp(self) -> None:
        """Setup method that runs before each test"""
        # Create a temporary directory for testing
        self.test_dir = (str(tempfile.mkdtemp()).replace("\\", "/") + "/").replace("//", "/")
        self.report_dir = (str(os.path.join(self.test_dir, "reports")).replace("\\", "/") + "/").replace("//", "/")
        os.makedirs(self.report_dir, exist_ok=True)

        # Create test data directories
        self.images_dir, self.rois_dir = create_test_directory_structure(self.test_dir)

        # Initialize SpecExp instance
        self.spec_exp = SpecExp(report_directory=self.report_dir, log_loading=False)

    def tearDown(self) -> None:
        """Cleanup after each test method"""
        if hasattr(self, "test_dir") and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def spec_exp_init(self) -> None:
        """Initialize SpecExp"""
        self.spec_exp = SpecExp(report_directory=self.report_dir, log_loading=False)

    @silent
    def test_initialization(self) -> None:
        """Test SpecExp initialization"""
        # Initialize SpecExp
        self.spec_exp_init()

        assert self.spec_exp.report_directory == self.report_dir
        assert self.spec_exp.log_loading is False
        assert len(self.spec_exp.groups) == 0
        assert len(self.spec_exp.images) == 0
        assert len(self.spec_exp.rois) == 0

        # Test invalid report directory
        dir_path = "/invalid/path/that/does/not/exist"
        with pytest.raises(ValueError, match=dir_path):
            SpecExp(report_directory=dir_path)

    @silent
    def test_add_groups(self) -> None:
        """Test adding groups"""
        # Initialize SpecExp
        self.spec_exp_init()

        # Add single group
        self.spec_exp.add_groups(["group1"])
        assert "group1" in self.spec_exp.groups

        # Add multiple groups
        self.spec_exp.add_groups(["group2", "group3"])
        assert "group2" in self.spec_exp.groups
        assert "group3" in self.spec_exp.groups

        # Try to add duplicate group
        self.spec_exp.add_groups(["group1"])
        assert self.spec_exp.groups.count("group1") == 1  # Should not duplicate

        # Initialize SpecExp after testing
        self.spec_exp_init()

    @silent
    def test_rm_group(self) -> None:
        """Test removing groups"""
        # Initialize SpecExp
        self.spec_exp_init()

        # Setup: add groups and some data
        self.spec_exp.add_groups(["group1", "group2"])
        self.spec_exp.add_images("group1", image_name="image1.tif", image_directory=self.images_dir)

        # Remove group
        self.spec_exp.rm_group("group1")
        assert "group1" not in self.spec_exp.groups
        assert len(self.spec_exp.images) == 0  # Should remove associated images

        # Initialize SpecExp after testing
        self.spec_exp_init()

    @silent
    def test_add_images(self) -> None:
        """Test adding images"""
        # Initialize SpecExp
        self.spec_exp_init()

        self.spec_exp.add_groups(["test_group"])

        # Add images by name pattern
        self.spec_exp.add_images(group_name="test_group", image_name="image*.tif", image_directory=self.images_dir)

        # Check if images were added
        images = self.spec_exp.ls_images(return_dataframe=True)
        assert len(images) == 3  # image1.tif, image2.tif, image3.tif

        # Add images by full path
        image_path = os.path.join(self.images_dir, "mask1.tif")
        self.spec_exp.add_images(group_name="test_group", full_path=image_path, mask_of="image1.tif")

        # Check if mask was added
        masks = self.spec_exp.ls_images(mask_of="image1.tif", return_dataframe=True)
        assert len(masks) == 1

        # Test adding images with duplicated config of img_name and full_path
        with pytest.warns(UserWarning, match="Double definition of name pattern and paths"):
            self.spec_exp.add_images(
                group_name="test_group",
                image_name="image*.tif",
                full_path=image_path,
                image_directory=self.images_dir,
            )

        # Test adding images with invalid full path
        img_path = "/invalid/path/that/does/not/exist/image1.tif"
        with pytest.raises(ValueError, match=img_path):
            self.spec_exp.add_images("test_group", full_path=[img_path], image_directory=self.images_dir)

    @silent
    def test_ls_images(self) -> None:
        """Test listing images"""
        # Initialize SpecExp
        self.spec_exp_init()

        self.spec_exp.add_groups(["test_group"])
        self.spec_exp.add_images("test_group", image_name="image*.tif", image_directory=self.images_dir)

        # List all images
        all_images = self.spec_exp.ls_images(return_dataframe=True, print_result=False)

        assert len(all_images) == 3

        # List specific image
        specific_image = self.spec_exp.ls_images(image_name="image1.tif", return_dataframe=True, print_result=False)
        assert len(specific_image) == 1

    @silent
    def test_rm_images(self) -> None:
        """Test removing images"""
        # Initialize SpecExp
        self.spec_exp_init()

        self.spec_exp.add_groups(["test_group"])
        self.spec_exp.add_images("test_group", image_name="image*.tif", image_directory=self.images_dir)

        # Remove specific image
        self.spec_exp.rm_images(image_name="image1.tif")
        remaining_images = self.spec_exp.ls_images(return_dataframe=True, print_result=False, mask_of="")
        assert len(remaining_images) == 2

    @silent
    def test_add_rois_by_suffix(self) -> None:
        """Test adding ROIs by suffix"""
        # Initialize SpecExp
        self.spec_exp_init()

        self.spec_exp.add_groups(["test_group"])
        self.spec_exp.add_images("test_group", image_name="image*.tif", image_directory=self.images_dir)

        # Add ROIs by suffix
        self.spec_exp.add_rois_by_suffix("test_group", roi_filename_suffix="_rois.xml", search_directory=self.rois_dir)

        # Check if ROIs were added
        rois = self.spec_exp.ls_rois_sample(return_dataframe=True, print_result=False)
        assert len(rois) == 4

        # Test adding rois with invalid group
        group_name = "group_that_doex_not_exist"
        with pytest.raises(ValueError, match=group_name):
            self.spec_exp.add_rois_by_suffix(
                group_name, roi_filename_suffix="_rois.xml", search_directory=self.rois_dir
            )

        # Test adding rois with invalid image pattern
        self.spec_exp_init()
        self.spec_exp.add_groups(["test_group"])
        with pytest.raises(ValueError, match="No image added"):
            self.spec_exp.add_rois_by_suffix(
                "test_group", roi_filename_suffix="_rois.xml", search_directory=self.rois_dir
            )

        # Test adding rois with filter
        # Init
        self.spec_exp_init()
        self.spec_exp.add_groups(["test_group"])
        self.spec_exp.add_images("test_group", image_name="image1.tif", image_directory=self.images_dir)
        # Include adding
        self.spec_exp.add_rois_by_suffix(
            "test_group",
            roi_filename_suffix="_rois.xml",
            search_directory=self.rois_dir,
            include_roiname=["1"],
        )
        rois = self.spec_exp.ls_rois_sample(return_dataframe=True, print_result=False)
        assert len(rois) == 1

        # Init
        self.spec_exp_init()
        self.spec_exp.add_groups(["test_group"])
        self.spec_exp.add_images("test_group", image_name="image1.tif", image_directory=self.images_dir)
        # Exclude adding
        self.spec_exp.add_rois_by_suffix(
            "test_group",
            roi_filename_suffix="_rois.xml",
            search_directory=self.rois_dir,
            exclude_roiname=["1"],
        )
        rois = self.spec_exp.ls_rois_sample(return_dataframe=True, print_result=False)
        assert len(rois) == 1

    @silent
    def test_add_rois_by_file(self) -> None:
        """Test adding ROIs by file path"""
        # Initialize SpecExp
        self.spec_exp_init()

        self.spec_exp.add_groups(["test_group"])
        self.spec_exp.add_images("test_group", image_name="image1.tif", image_directory=self.images_dir)

        # Add ROIs by file path
        roi_file = os.path.join(self.rois_dir, "image1_rois.xml")
        self.spec_exp.add_rois_by_file("test_group", path_list=[roi_file], image_name="image1.tif")

        # Check if ROIs were added
        rois = self.spec_exp.ls_rois_from_file(return_dataframe=True, print_result=False)
        assert len(rois) == 2

        # Test adding rois with invalid group
        group_name = "group_that_doex_not_exist"
        with pytest.raises(ValueError, match=group_name):
            self.spec_exp.add_rois_by_file(group_name, path_list=[roi_file], image_name="image1.tif")

        # Test adding rois with invalid image pattern
        img_name = "image_name_that_doex_not_exist"
        with pytest.raises(ValueError, match=img_name):
            self.spec_exp.add_rois_by_file("test_group", path_list=[roi_file], image_name=img_name)

        # Test adding rois with filter
        # Init
        self.spec_exp_init()
        self.spec_exp.add_groups(["test_group"])
        self.spec_exp.add_images("test_group", image_name="image1.tif", image_directory=self.images_dir)
        # Include adding
        self.spec_exp.add_rois_by_file(
            "test_group", path_list=[roi_file], image_name="image1.tif", include_roiname=["1"]
        )
        rois = self.spec_exp.ls_rois_sample(return_dataframe=True, print_result=False)
        assert len(rois) == 1

        # Init
        self.spec_exp_init()
        self.spec_exp.add_groups(["test_group"])
        self.spec_exp.add_images("test_group", image_name="image1.tif", image_directory=self.images_dir)
        # Exclude adding
        self.spec_exp.add_rois_by_file(
            "test_group", path_list=[roi_file], image_name="image1.tif", exclude_roiname=["1"]
        )
        rois = self.spec_exp.ls_rois_sample(return_dataframe=True, print_result=False)
        assert len(rois) == 1

    @silent
    def test_add_roi_by_coords(self) -> None:
        """Test adding ROIs by coordinates"""
        # Initialize SpecExp
        self.spec_exp_init()

        self.spec_exp.add_groups(["test_group"])
        self.spec_exp.add_images("test_group", image_name="image1.tif", image_directory=self.images_dir)

        # Add ROI by coordinates
        coordinates: list[list[tuple[Union[int, float], Union[int, float]]]] = [
            [(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)]
        ]
        self.spec_exp.add_roi_by_coords("test_roi", "test_group", "image1.tif", coordinates)

        # Check if ROI was added
        rois = self.spec_exp.ls_rois_from_coords(return_dataframe=True, print_result=False)
        assert len(rois) == 1

    @silent
    def test_ls_rois(self) -> None:
        """Test listing ROIs"""
        # Initialize SpecExp
        self.spec_exp_init()

        self.spec_exp.add_groups(["test_group"])
        self.spec_exp.add_images("test_group", image_name="image1.tif", image_directory=self.images_dir)

        # Add ROI by coordinates
        coordinates: list[list[tuple[Union[int, float], Union[int, float]]]] = [
            [(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)]
        ]
        self.spec_exp.add_roi_by_coords("test_roi", "test_group", "image1.tif", coordinates)

        # List all ROIs
        all_rois = self.spec_exp.ls_rois(return_dataframe=True)
        assert len(all_rois) == 1

        # List specific ROI
        specific_roi = self.spec_exp.ls_rois(roi_name_list=["test_roi"], return_dataframe=True)
        assert len(specific_roi) == 1

    @silent
    def test_rm_rois(self) -> None:
        """Test removing ROIs"""
        # Initialize SpecExp
        self.spec_exp_init()

        self.spec_exp.add_groups(["test_group"])
        self.spec_exp.add_images("test_group", image_name="image1.tif", image_directory=self.images_dir)

        # Add ROI by coordinates
        coordinates: list[list[tuple[Union[int, float], Union[int, float]]]] = [
            [(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)]
        ]
        self.spec_exp.add_roi_by_coords("test_roi", "test_group", "image1.tif", coordinates)

        # Remove ROI
        self.spec_exp.rm_rois(roi_name="test_roi")
        remaining_rois = self.spec_exp.ls_rois(return_dataframe=True)
        assert len(remaining_rois) == 0

    @silent
    def test_add_standalone_specs(self) -> None:
        """Test adding standalone spectra"""
        # Initialize SpecExp
        self.spec_exp_init()

        self.spec_exp.add_groups(["test_group"])

        # Create test spectral data
        spectral_data = [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]]

        # Add standalone spectra
        self.spec_exp.add_standalone_specs("test_group", spectral_data, sample_name_list=["spec1", "spec2"])

        # Check if spectra were added
        specs = self.spec_exp.ls_standalone_specs(return_dataframe=True)
        assert len(specs) == 2

    @silent
    def test_ls_standalone_specs(self) -> None:
        """Test listing standalone spectra"""
        # Initialize SpecExp
        self.spec_exp_init()

        self.spec_exp.add_groups(["test_group"])

        # Create test spectral data
        spectral_data = [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]]

        # Add standalone spectra
        self.spec_exp.add_standalone_specs("test_group", spectral_data, sample_name_list=["spec1", "spec2"])

        # List all spectra
        all_specs = self.spec_exp.ls_standalone_specs(return_dataframe=True)
        assert len(all_specs) == 2

        # List specific spectrum
        specific_spec = self.spec_exp.ls_standalone_specs(sample_name="spec1", return_dataframe=True)
        assert len(specific_spec) == 1

    @silent
    def test_rm_standalone_specs(self) -> None:
        """Test removing standalone spectra"""
        # Initialize SpecExp
        self.spec_exp_init()

        self.spec_exp.add_groups(["test_group"])

        # Create test spectral data
        spectral_data = [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]]

        # Add standalone spectra
        self.spec_exp.add_standalone_specs("test_group", spectral_data, sample_name_list=["spec1", "spec2"])

        # Remove specific spectrum
        self.spec_exp.rm_standalone_specs(sample_name="spec1")
        remaining_specs = self.spec_exp.ls_standalone_specs(return_dataframe=True)
        assert len(remaining_specs) == 1

    @silent
    def test_sample_labels_management(self) -> None:
        """Test sample labels management"""
        # Initialize SpecExp
        self.spec_exp_init()

        # First add some samples (ROIs or standalone spectra)
        self.spec_exp.add_groups(["test_group"])

        # Add ROI to create samples
        self.spec_exp.add_images("test_group", image_name="image1.tif", image_directory=self.images_dir)

        coordinates: list[list[tuple[Union[int, float], Union[int, float]]]] = [
            [(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)]
        ]
        self.spec_exp.add_roi_by_coords("test_roi", "test_group", "image1.tif", coordinates)

        # Test listing sample labels
        labels = self.spec_exp.ls_sample_labels(return_dataframe=True)
        assert len(labels) > 0

        # Test setting sample labels from dataframe
        new_labels = pd.DataFrame({"Sample_ID": labels["Sample_ID"], "Label": ["label1"] * len(labels)})
        self.spec_exp.sample_labels_from_df(new_labels)

        # Test exporting to CSV
        csv_path = os.path.join(self.test_dir, "test_labels.csv")
        self.spec_exp.sample_labels_to_csv(csv_path)
        assert os.path.exists(csv_path)

        # Test importing from CSV
        self.spec_exp.sample_labels_from_csv(csv_path)
        updated_labels = self.spec_exp.ls_sample_labels(return_dataframe=True)
        assert len(updated_labels) == len(labels)

    @silent
    def test_sample_targets_management(self) -> None:
        """Test sample targets management"""
        # Initialize SpecExp
        self.spec_exp_init()

        # First add some samples
        self.spec_exp.add_groups(["test_group"])

        # Add ROI to create samples
        self.spec_exp.add_images("test_group", image_name="image1.tif", image_directory=self.images_dir)

        coordinates: list[list[tuple[Union[int, float], Union[int, float]]]] = [
            [(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)]
        ]
        self.spec_exp.add_roi_by_coords("test_roi", "test_group", "image1.tif", coordinates)

        # Set sample labels
        labels = self.spec_exp.ls_sample_labels(return_dataframe=True)
        labels["Label"] = [str(i) for i in range(len(labels))]
        self.spec_exp.sample_labels_from_df(labels)

        # Test listing sample targets
        targets = self.spec_exp.ls_sample_targets(return_dataframe=True)
        assert len(targets) > 0

        # Test setting sample targets from dataframe
        new_targets = pd.DataFrame(
            {
                "Sample_ID": targets["Sample_ID"],
                "Sample_label": targets["Sample_label"],
                "Target_value": [1.0] * len(targets),
            }
        )
        self.spec_exp.sample_targets_from_df(new_targets)

        # Test exporting to CSV
        csv_path = os.path.join(self.test_dir, "test_targets.csv")
        self.spec_exp.sample_targets_to_csv(csv_path)
        assert os.path.exists(csv_path)

        # Test importing from CSV
        self.spec_exp.sample_targets_from_csv(csv_path)
        updated_targets = self.spec_exp.ls_sample_targets(return_dataframe=True)
        assert len(updated_targets) == len(targets)

    @silent
    def test_save_and_load_config(self) -> None:
        """Test saving and loading configuration"""
        # Initialize SpecExp
        self.spec_exp_init()

        # Setup some data
        self.spec_exp.add_groups(["test_group"])
        self.spec_exp.add_images("test_group", image_name="image1.tif", image_directory=self.images_dir)

        coordinates: list[list[tuple[Union[int, float], Union[int, float]]]] = [
            [(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)]
        ]
        self.spec_exp.add_roi_by_coords("test_roi", "test_group", "image1.tif", coordinates)

        # Save configuration
        self.spec_exp.save_data_config(copy=False)

        # Create a new instance and load configuration
        new_spec_exp = SpecExp(report_directory=self.report_dir, log_loading=False)
        new_spec_exp.load_data_config()

        # Check if data was loaded correctly
        assert len(new_spec_exp.groups) == 1
        assert len(new_spec_exp.images) == 1
        assert len(new_spec_exp.rois) == 1

    @silent
    def test_property_access(self) -> None:
        """Test property accessors"""
        # Initialize SpecExp
        self.spec_exp_init()

        # Test read-only properties
        with pytest.raises(ValueError):
            self.spec_exp.groups = ["new_group"]

        with pytest.raises(ValueError):
            self.spec_exp.images = []

        with pytest.raises(ValueError):
            self.spec_exp.rois = []

        # Test readable properties
        assert isinstance(self.spec_exp.groups, list)
        assert isinstance(self.spec_exp.images, list)
        assert isinstance(self.spec_exp.rois, list)

    def test_method_alias(self) -> None:
        """Test method alias"""
        # Initialize SpecExp
        self.spec_exp_init()
        spec1 = self.spec_exp

        # Validate consistency of identical method
        assert spec1.add_specs == spec1.add_standalone_specs
        assert spec1.load_specs == spec1.load_standalone_specs
        assert spec1.ls_specs == spec1.ls_standalone_specs
        assert spec1.labels_to_csv == spec1.sample_labels_to_csv
        assert spec1.ls_labels == spec1.ls_sample_labels
        assert spec1.labels_from_df == spec1.sample_labels_from_df
        assert spec1.labels_from_csv == spec1.sample_labels_from_csv
        assert spec1.targets_from_csv == spec1.sample_targets_from_csv
        assert spec1.targets_to_csv == spec1.sample_targets_to_csv
        assert spec1.ls_targets == spec1.ls_sample_targets
        assert spec1.targets_from_df == spec1.sample_targets_from_df
        assert spec1.save_config == spec1.save_data_config
        assert spec1.load_config == spec1.load_data_config


# %% Test - SpecExp


# test_specexp = TestSpecExp()
# test_specexp.setUp()

# test_specexp.test_initialization()

# test_specexp.test_add_groups()
# test_specexp.test_rm_group()

# test_specexp.test_add_images()
# test_specexp.test_ls_images()
# test_specexp.test_rm_images()

# test_specexp.test_add_rois_by_suffix()
# test_specexp.test_add_rois_by_file()
# test_specexp.test_add_roi_by_coords()
# test_specexp.test_ls_rois()
# test_specexp.test_rm_rois()

# test_specexp.test_add_standalone_specs()
# test_specexp.test_ls_standalone_specs()
# test_specexp.test_rm_standalone_specs()

# test_specexp.test_sample_labels_management()
# test_specexp.test_sample_targets_management()

# test_specexp.test_save_and_load_config()

# test_specexp.test_property_access()

# test_specexp.test_method_alias()

# test_specexp.tearDown()


# %% Test main

if __name__ == "__main__":
    unittest.main()
