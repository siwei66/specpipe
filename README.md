# SpecPipe

## A high-performance, file-based pipeline for batch processing and modeling of hyperspectral images.

SpecPipe streamlines the batch testing and optimization of hyperspectral analysis workflows. It provides a structured framework to apply various image processing techniques (calibration, baseline correction, denoising, feature engineering, etc.) in combination with various machine learning models. The pipeline employs a comprehensive full-factorial design to evaluate all method combinations and generates standard reports on performance metrics, residual analysis, influence anlaysis and visualizations.

## Core features:
- **Batch processing**: Automate numerous data processing and modeling workflows in a single batch operation.
- **File-based**: A resumable, file-based processing pipeline enabling result validation and break tolerance.
- **High-performance**: Optimized for hyperspectral images with minimal memory consumption and options of GPU acceleration and pipeline-level multiprocessing.
- **Simple extensible integration**: Intuitive data management and straightforward integration for custom processing functions and Scikit-learn-style models.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Installation <a name="installation"></a>

Follow these steps to install the project:

1.  **Prerequisites:** Ensure you have Python 3.9 or higher installed.

2.  **Install from PyPI (Recommended):**
    ```python
    pip install specpipe
    ```

3.  **Install from source (for development):**
    ```python
    git clone https://github.com/siwei66/SpecPipe.git
    cd SpecPipe
    pip install -e specpipe
    ```


## Usage <a name="usage"></a>

### 1.  Prepare mock spectral experiment data

- Create a directory for mock experiment data (The example uses a temporary directory):
    ```python
    import tempfile
    data_dir = tempfile.mkdtemp(prefix="specpipe_example_")
    ```

- Create mock data:
    ```python
    from specpipe import create_test_raster, create_test_roi_xml
    create_test_raster(f"{data_dir}/example.tif")
    create_test_roi_xml(f"{data_dir}/example_roi.xml")
    ```

### 2. Configure your experiment data

#### 2.1 Create a spectral experiment instance

- Here we use the same directory as report directory:
    ```python
    report_dir = data_dir
    ```

- Create a SpecExp instance:
    ```python
    from specpipe import SpecExp
    exp = SpecExp(report_dir)
    ```
The instance stores and organizes the data loading configurations of an experiment, which faciliates lazy-loading.

- Check report directory:
    ```python
    exp.report_directory
    ```

#### 2.2. Experiment group management

- Add experiment groups:
    ```python
    exp.add_groups(['group_1', 'group_2'])
    ```

- Check groups:
    ```python
    exp.ls_groups()
    ```

- Remove a group:
    ```python
    exp.rm_group('group_2')
    ```


#### 2.3. Raster image management

- Add raster images:
    ```python
    exp.add_images(
        group_name = 'group_1', 
        image_name = 'example', 
        image_directory = data_dir
        )
    ```

- Check added images:
    ```python
    exp.ls_images()
    ```


#### 2.4. Region of interest (ROI) management

- Load image ROIs using suffix to image names:
    ```python
    exp.add_rois_by_suffix('group_1', '_roi.xml', data_dir)
    ```

- Remove ROIs by name:
    ```python
    exp.rm_rois(roi_name='ROI_10')
    ```

- Load ROIs to a image using ROI files by paths:
    ```python
    exp.add_rois_by_file('group_1', [f"{data_dir}/example_roi.xml"], image_name='example.tif')
    ```

- Check added ROIs:
    ```python
    exp.ls_rois()
    ```
The example xml ROI file contains 10 ROIs.

- Check sample ROIs:
    ```python
    exp.ls_rois_sample()
    ```


#### 2.5. Sample labels and target values

##### 2.5.1 Set sample labels

- Get current sample label dataframe:
    ```python
    labels = exp.ls_labels()
    ```

- Set new sample labels in the dataframe:
Here we use sample_1, sample_2 ... sample_10
    ```python
    labels.iloc[:,1] = [f'sample_{str(i+1)}' for i in range(len(labels))]
    ```

- Update sample labels using the updated dataframe:
    ```python
    exp.sample_labels = labels
    ```

- Check sample labels:
    ```python
    exp.ls_labels()
    ```

##### 2.5.2 Set target values

- List target value dataframe:
    ```python
    targets = exp.ls_sample_targets()
    ```

- Create mock target values for regression and update target dataframe:
Here we use 1 to 10
    ```python
    targets['Target_value'] = [i for i in range(len(targets))]
    ```

- Load target values from updated target dataframe:
    ```python
    exp.sample_targets_from_df(targets)
    ```

- Check target values:
    ```python
    exp.ls_targets()
    ```


### 3. Design testing pipeline

- SpecPipe follows a structured data processing workflow with these sequential data levels:
    ```text
    Raster image data -> ROI spectra -> ROI statistics -> Traits to model
    ```

- The technical data levels in SpecPipe includes:
    ```text
    Images: 
        0 - "image", input image path and output processed image path.
    
    Image pixel spectra: 
        1 - "pixel_spec", 1D spectrum of image pixel (simple)
    
        2 - "pixel_specs_array", 2D spectra array of image pixels (fast)
    
        3 - "pixel_specs_tensor", 2D spectra tensor of image pixels (fast)
    
        4 - "pixel_hyperspecs_tensor", 2D hyperspectra tensor of image pixels (fastest)
    
        (See "rasterop.pixel_apply" - apply processing functions to spectra of image pixels)
    
    Image ROIs:
        5 - "image_roi", raster with sample ROIs, for spectrum extraction
    
        6 - "roispecs", 2D array of ROI spectra
    
        7 - "spec1d", arbitrary 1D data of samples, e.g. 1D spectra, spectra statistics
    
    Models:
        8 - "model", model evaluation with standard report output as files
    ```

- The corresponding data processing workflow is:
    ```text
    Raster image data:    0 ~ 4
        ↓
    Extract ROI spectra:  5 - "image_roi"
        ↓
    ROI spectra:          6 - "roispecs"
        ↓
    ROI statistics:       7 - "spec1d"
        ↓
    Model evaluation:     8 - "model"
    ```
The processing functions are wrapped in the pipeline according to the specified "data levels".
Parallel processes can be added with identical "data level" and "application sequence", and they are arranged using full-factorial approach in the pipeline.


#### 3.1 Create processing pipeline
- Create processing pipeline from SpecExp instance configured above:
    ```python
    from specpipe import SpecPipe
    
    pipe = SpecPipe(exp)
    ```

#### 3.2 Image processing

- Create some image processing functions, such as: 

- Standard normal variate: 
    ```python
    import numpy as np
    def snv(v):
        vmean = np.mean(v, axis=1, keepdims=True)
        vstd = np.std(v, axis=1, keepdims=True)
        snv = (v - vmean) / vstd
        return snv
    ```

- Raw data for performance comparison:
    ```python
    def raw(v):
        return v
    ```

- Add these processing functions to the pipeline:
    ```python
    pipe.add_process(
        input_data_level = 'pixel_specs_array', 
        output_data_level = 'pixel_specs_array', 
        application_sequence = 0, 
        method = snv
        )
    ```

- Or we can specify the data level using the corresponding number:
    ```python
    pipe.add_process(2, 2, 0, raw)
    ```


#### 3.3 ROI statistics

- Import some ROI spectral statistic metrics:
    ```python
    from specpipe import roi_mean
    from specpipe import roi_median
    ```

- Add these process to the pipeline:
    ```python
    pipe.add_process(
        input_data_level = 'image_roi', 
        output_data_level = 'spec1d', 
        application_sequence = 0, 
        method = roi_mean
        )
    ```

- Or specify data level using number:
    ```python
    pipe.add_process(5, 7, 0, roi_median)
    ```


#### 3.4 Sample data wrangling

- Create a function to remove nan and inf values:
    ```python
    def replace_nan(v: np.ndarray) -> np.ndarray:
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    ```

- Add the process to the pipeline:
    ```python
    pipe.add_process('spec1d', 'spec1d', 0, replace_nan)
    ```

- Check all added processes:
    ```python
    pipe.ls_process()
    ```

- To remove added processes from the pipeline:
    ```python
    pipe.rm_process(method='raw')
    ```
Processes can be removed by various criteria, the example removes the function 'raw' by its name.


#### 3.5 Add models to the pipeline

- Create some models:
    ```python
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    
    linear_regressor = LinearRegression()
    rf_regressor = RandomForestRegressor(n_estimators=10)
    knn_regressor = KNeighborsRegressor(n_neighbors=3)
    svr = SVR()
    ```

- Add model using "add_process":
    ```python
    pipe.add_process('spec1d', 'model', 1, linear_regressor, validation_method = '10-fold')
    pipe.add_process(7, 8, 1, rf_regressor, validation_method = '10-fold')
    ```

- Add model using add_model:
    ```python
    pipe.add_model(knn_regressor, validation_method = '10-fold')
    pipe.add_model(svr, validation_method = '10-fold')
    ```

- Check added models:
    ```python
    pipe.ls_model()
    ```

- Check all processes including models:
    ```python
    pipe.ls_process()
    ```


### 4 Run pipeline

- Check processing chains of the pipeline:
    ```python
    pipe.ls_chains()
    ```

- Manually test all added processes:
    ```python
    pipe.test_run()
    ```
The added processes are also automatically tested before formal run.

- Run pipeline:
    ```python
    pipe.run()
    ```

- Set resume True to enable resuming:
    ```python
    pipe.run(resume=True)
    ```
If the implementation is interrupted or forcibly terminated, running the pipeline again will automatically resume from the last completed step.


### 5 Running results

- The pipeline produces following results for every processing chain, including:
• Final and intermediate preprocessing results
• Data configuration files
• Validation and application models
• Model evaluation reports
• Visualization plots

- **The resulting file structure is as follows:**

- For input data:
    ```text
    data_directory/
    ├── SpecExp_configuration/
    │    ├── Loading_history/
    │    │   ├── Loaded_images.csv
    │    │   └── Loaded_ROIs.csv
    │    └── SpecExp_data_configuration.dill
    ├── Your_rasters.tif
    └── Your_ROIs.xml
    ```

- For regression tasks, the pipeline generates:
    ```text
    report_directory/
    ├── Modeling/
    │    └── Model_Evaluation_Reports/
    │        ├── Data_chain_Preprocessing_#0_Model_(model label 0)/
    │        │   ├── Model_for_application/
    │        │   ├── Model_in_validation/
    │        │   ├── Validation_results.csv
    │        │   ├── Regression_performance.csv
    │        │   ├── Residual_analysis.csv
    │        │   ├── Influence_analysis.csv
    │        │   ├── Scatter_plot.png
    │        │   └── Residual_plot.png
    │        ├── Data_chain_Preprocessing_#0_Model_(model label 1)/
    │        ├── Data_chain_Preprocessing_#1_Model_(model label 0)/
    │        ├── Data_chain_Preprocessing_#1_Model_(model label 1)/
    │        ├──  Preprocessing_#0.txt
    │        └──  Preprocessing_#1.txt
    ├── Pre_execution_test_data/
    ├── Preprocessing/
    │    ├── Step_results/
    │    ├── PreprocessingChainResult_chain_0.csv
    │    └── PreprocessingChainResult_chain_1.csv
    └── test_run/
    ```

- For classification tasks, the pipeline generates:
    ```text
    report_directory/
    ├── Modeling/
    │    └── Model_Evaluation_Reports/
    │        ├── Data_chain_Preprocessing_#0_Model_(model label 0)/
    │        │   ├── Model_for_application/
    │        │   ├── Model_in_validation/
    │        │   ├── Validation_results.csv
    │        │   ├── Classification_performance.csv
    │        │   ├── Residual_analysis.csv
    │        │   ├── Influence_analysis.csv
    │        │   └── ROC_curve.png
    │        ├── Data_chain_Preprocessing_#0_Model_(model label 1)/
    │        ├── Data_chain_Preprocessing_#1_Model_(model label 0)/
    │        ├── Data_chain_Preprocessing_#1_Model_(model label 1)/
    │        ├──  Preprocessing_#0.txt
    │        └──  Preprocessing_#1.txt
    ├── Pre_execution_test_data/
    ├── Preprocessing/
    │    ├── Step_results/
    │    ├── PreprocessingChainResult_chain_0.csv
    │    └── PreprocessingChainResult_chain_1.csv
    └── test_run/
    ```


## Contributing <a name="contributing"></a>

This is an initial release of SpecPipe. Your experience applying this toolset in your specialized field is extremely valuable. Any feedback and contributions are highly welcomed!

- **Report bugs**: Found an issue? Please open a [GitHub issue](https://github.com/siwei66/specpipe/issues) with details
- **Share your domain expertise**: Tell us how SpecPipe works (or doesn't work) in your specific application area in [discussions](https://github.com/siwei66/specpipe/discussions)
- **Suggest features**: Have ideas for improvements? Use the [GitHub discussions](https://github.com/siwei66/specpipe/discussions) or issues tab
- **Submit pull requests**: Feel free to fork and submit PRs for bug fixes or small features
- **Test and provide feedback**: Try it out and let us know about your experience in [discussions](https://github.com/siwei66/specpipe/discussions)


## License <a name="license"></a>

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/MIT) file for details.
