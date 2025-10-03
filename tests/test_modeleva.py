# -*- coding: utf-8 -*-
"""
Tests for modeling and model evaluation module (ModelEva)

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Test
# OS Files
import os
import shutil

# For local test - delete after use
import tempfile
import unittest

# Testing third
import numpy as np
import pandas as pd
import pytest

# Modeling
from sklearn.linear_model import LinearRegression, LogisticRegression

# Self
# Funcs to test
from specpipe.modeleva import ModelEva

# %% Helper functions for ModelEva testing


def create_sample_data_regression(
    n_samples: int = 10,
) -> list[tuple[str, tuple[int], float, np.ndarray]]:
    """Create regression sample data."""
    np.random.seed(66)
    X = np.random.rand(n_samples, 5)  # noqa: N806
    y = np.random.rand(n_samples)
    return [(f"sample_{i}", (5,), float(y[i]), X[i]) for i in range(n_samples)]


def create_sample_data_classification(
    n_samples: int = 10,
) -> list[tuple[str, tuple[int], str, np.ndarray]]:
    """Create classification sample data."""
    np.random.seed(66)
    X = np.random.rand(n_samples, 5)  # noqa: N806
    y = np.random.choice(["A", "B"], size=n_samples)
    return [(f"sample_{i}", (5,), str(y[i]), X[i]) for i in range(n_samples)]


def modeleva_initialization_regression(temp_dir: str, data_split: str = "5-fold", n_samples: int = 10) -> ModelEva:
    """Test regression model initialization."""
    sample_list = create_sample_data_regression(n_samples)

    model = LinearRegression()

    report_dir = str(temp_dir).replace("\\", "/")
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    model_eva = ModelEva(
        sample_list=sample_list,
        model=model,
        validation_method=data_split,
        report_directory=report_dir,
        data_label="test_reg_data",
        is_regression=True,
        silent_all=True,
    )

    return model_eva


def modeleva_initialization_classification(temp_dir: str, data_split: str = "5-fold", n_samples: int = 10) -> ModelEva:
    """Test classification model initialization."""
    sample_list = create_sample_data_classification(n_samples)

    model = LogisticRegression()

    report_dir = str(temp_dir).replace("\\", "/")
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    model_eva = ModelEva(
        sample_list=sample_list,
        model=model,
        validation_method=data_split,
        report_directory=report_dir,
        data_label="test_cls_data",
        is_regression=False,
        silent_all=True,
    )

    return model_eva


# %% test functions : ModelEva


class TestModelEva:
    """Test class for ModelEva functionality."""

    @staticmethod
    def test_initialization_regression() -> None:
        """Test regression model initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_eva = modeleva_initialization_regression(temp_dir)

            assert os.path.exists(model_eva.report_directory)
            assert type(model_eva.model) is LinearRegression
            assert model_eva.model_label == LinearRegression.__name__
            assert model_eva.is_regression is True
            assert model_eva.X.shape == (10, 5)
            assert model_eva.y.shape == (10, 1)
            assert model_eva.validation_method == 5

        # Clear test report dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @staticmethod
    def test_initialization_classification() -> None:
        """Test classification model initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_eva = modeleva_initialization_classification(temp_dir)

            assert os.path.exists(model_eva.report_directory)
            assert type(model_eva.model) is LogisticRegression
            assert model_eva.model_label == LogisticRegression.__name__
            assert model_eva.is_regression is False
            assert model_eva.X.shape == (10, 5)
            assert model_eva.y.shape == (10, 1)
            assert model_eva.validation_method == 5

        # Clear test report dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @staticmethod
    def test_auto_choose_classification_regression() -> None:
        """Test automatic determination of regression and classification using target type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create report direcotry
            report_dir = str(temp_dir).replace("\\", "/")
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)

            # Auto regression
            sample_list_reg = create_sample_data_regression(10)
            model = LinearRegression()
            model_eva = ModelEva(
                sample_list=sample_list_reg,
                model=model,
                validation_method="10-fold",
                report_directory=report_dir,
                data_label="test_reg_data",
                silent_all=True,
            )
            assert model_eva.is_regression is True

            # Auto classification
            sample_list_cls = create_sample_data_classification(10)
            model = LogisticRegression()
            model_eva = ModelEva(
                sample_list=sample_list_cls,
                model=model,
                validation_method="10-fold",
                report_directory=report_dir,
                data_label="test_reg_data",
                silent_all=True,
            )
            assert model_eva.is_regression is False

        # Clear test report dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @staticmethod
    def test_auto_kfold_to_loocv() -> None:
        """Test automatic convert kfold to loo if k larger than sample size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create report direcotry
            report_dir = str(temp_dir).replace("\\", "/")
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)

            # Create model evaluator
            sample_list = create_sample_data_regression(8)
            model = LinearRegression()
            with pytest.warns(UserWarning, match="'loo' is applied instead"):
                model_eva = ModelEva(
                    sample_list=sample_list,
                    model=model,
                    validation_method="10-fold",
                    report_directory=report_dir,
                    data_label="test_reg_data",
                    is_regression=True,
                    silent_all=True,
                )
            assert len(model_eva.sid) == 8
            assert model_eva.validation_method == "loo"

        # Clear test report dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @staticmethod
    def test_property_access() -> None:
        """Test read-only property access."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_eva = modeleva_initialization_classification(temp_dir)
            with pytest.raises(ValueError, match="cannot be modified"):
                model_eva.create_time = "2025-01-01"
            with pytest.raises(ValueError, match="cannot be modified"):
                model_eva.model_time = "2025-01-01"
            with pytest.raises(ValueError, match="cannot be modified"):
                model_eva.unseen_threshold = 0.0
            with pytest.raises(ValueError, match="cannot be modified"):
                model_eva.sid = np.array([["1", "2", "3"]])
            with pytest.raises(ValueError, match="cannot be modified"):
                model_eva.X_original_shape = (2, 2)
            with pytest.raises(ValueError, match="cannot be modified"):
                model_eva.y = [1, 2, 3]
            with pytest.raises(ValueError, match="cannot be modified"):
                model_eva.X = [[0, 1], [1, 2], [1, 2]]
            with pytest.raises(ValueError, match="cannot be modified"):
                model_eva.dsp_inds = [("1", "2"), ("2", "1")]
            with pytest.raises(ValueError, match="cannot be modified"):
                model_eva.y_true_eva = [1, 2, 3]
            with pytest.raises(ValueError, match="cannot be modified"):
                model_eva.y_pred_eva = [1, 2, 3]
            with pytest.raises(ValueError, match="cannot be modified"):
                model_eva.ynames = ["1", "2", "3"]
            with pytest.raises(ValueError, match="cannot be modified"):
                model_eva.y_true_proba_eva = [[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]
            with pytest.raises(ValueError, match="cannot be modified"):
                model_eva.y_pred_proba_eva = [[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]
            with pytest.raises(ValueError, match="cannot be modified"):
                model_eva.sid_eva = ["1", "2", "3"]
            with pytest.raises(ValueError, match="cannot be modified"):
                model_eva.app_model = LinearRegression()
            with pytest.raises(ValueError, match="cannot be modified"):
                model_eva.app_model_create_time = "2025-01-01"

        # Clear test report dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @staticmethod
    def test_update_samples() -> None:
        """Test sample update functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Classification
            model_eva = modeleva_initialization_classification(temp_dir)
            new_samples_cls = create_sample_data_classification(11)
            model_eva.update_samples(new_samples_cls)

            assert model_eva.X.shape == (11, 5)
            assert model_eva.y.shape == (11, 1)

            # Regression
            model_eva = modeleva_initialization_regression(temp_dir)
            new_samples_reg = create_sample_data_regression(12)
            model_eva.update_samples(new_samples_reg)

            assert model_eva.X.shape == (12, 5)
            assert model_eva.y.shape == (12, 1)

        # Clear test report dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    # Classification model evaluation tests
    @staticmethod
    def test_classifier_data_split() -> None:
        """Test data split functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # K-fold
            model_eva = modeleva_initialization_classification(temp_dir, data_split="5-fold", n_samples=20)

            assert model_eva.dsp_inds == []
            model_eva._data_split()
            assert len(model_eva.dsp_inds) == 5
            assert len(model_eva.dsp_inds[0][0]) == 16
            assert len(model_eva.dsp_inds[0][1]) == 4

            # LOO
            model_eva = modeleva_initialization_classification(temp_dir, data_split="loo")

            assert model_eva.dsp_inds == []
            model_eva._data_split()
            assert len(model_eva.dsp_inds) == 10
            assert len(model_eva.dsp_inds[0][0]) == 9
            assert len(model_eva.dsp_inds[0][1]) == 1

            # Train-test-split
            model_eva = modeleva_initialization_classification(temp_dir, data_split="60-40-split")

            assert model_eva.dsp_inds == []
            model_eva._data_split()
            assert len(model_eva.dsp_inds) == 1
            assert len(model_eva.dsp_inds[0][0]) == 6
            assert len(model_eva.dsp_inds[0][1]) == 4

        # Clear test report dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @staticmethod
    def test_classifier_validation() -> None:
        """Test data split functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_eva = modeleva_initialization_classification(temp_dir, n_samples=20)
            model_eva._data_split()
            model_time = model_eva.model_time

            assert len(model_time) > 0
            assert model_eva.y_true_eva is None
            assert model_eva.y_pred_eva is None
            assert model_eva.y_true_proba_eva is None
            assert model_eva.y_pred_proba_eva is None
            assert model_eva.sid_eva == []

            model_eva._classifier_validation()

            assert len(model_time) > 0
            assert model_eva.y_true_eva is not None
            assert model_eva.y_pred_eva is not None
            assert model_eva.y is not None
            assert len(model_eva.y_true_eva) == len(model_eva.y)
            assert len(model_eva.y_pred_eva) == len(model_eva.y)
            assert model_eva.y_true_proba_eva is not None
            assert np.all(model_eva.y_true_proba_eva.shape == (len(model_eva.y), len(model_eva.ynames)))
            assert model_eva.y_pred_proba_eva is not None
            assert np.all(model_eva.y_pred_proba_eva.shape == (len(model_eva.y), len(model_eva.ynames)))
            assert len(model_eva.sid_eva) > 0

        # Clear test report dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @staticmethod
    def test_classifier_evaluation() -> None:
        """Test classifier evaluation workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_eva = modeleva_initialization_classification(temp_dir, data_split="10-fold", n_samples=20)
            model_eva.classifier_evaluation()
            metrics_filename = f"Classification_performance_{model_eva.model_label}.csv"
            roc_filename = f"ROC_curve_{model_eva.model_label}.png"
            residual_filename = f"Residual_analysis_{model_eva.model_label}.csv"
            influence_analysis_filename = f"Influence_analysis_{model_eva.model_label}.csv"
            output_path = (
                model_eva._report_directory
                + f"Model_evaluation_reports/Data_{model_eva.data_label}_Model_{model_eva.model_label}/"
            )

            # Test metrics report output
            metrics_path = output_path + metrics_filename
            assert os.path.exists(metrics_path)
            try:
                metrics = pd.read_csv(metrics_path)
                assert True
            except Exception as e:
                raise AssertionError(f"Failed to read CSV file '{metrics_filename}': {e}") from e
            assert list(metrics.columns) == [
                "Class",
                "TP",
                "TN",
                "FP",
                "FN",
                "Precision",
                "Recall",
                "F1_Score",
                "Accuracy",
                "AUC",
            ]
            assert "LogisticRegression" in metrics.iloc[-1, 0]
            assert metrics.shape == (5, 10)
            assert metrics.iloc[:-1, :].isna().sum().sum() == 0

            # Test ROC plot existence (manual check for plot image required)
            roc_path = output_path + roc_filename
            assert os.path.exists(roc_path)

            # Test residual report output
            residual_path = output_path + residual_filename
            assert os.path.exists(residual_path)
            try:
                residual = pd.read_csv(residual_path)
                assert True
            except Exception as e:
                raise AssertionError(f"Failed to read CSV file '{residual_filename}': {e}") from e
            assert residual.shape == (20, 15)
            assert residual.iloc[:, :].isna().sum().sum() == 0

            # Test Cook's distance report output
            influence_analysis_path = output_path + influence_analysis_filename
            assert os.path.exists(influence_analysis_path)
            try:
                influence = pd.read_csv(influence_analysis_path)
                assert True
            except Exception as e:
                raise AssertionError(f"Failed to read CSV file '{influence_analysis_filename}': {e}") from e
            assert list(influence.columns) == [
                "Sample_ID",
                "A_probability_Cooks_distance_like",
                "B_probability_Cooks_distance_like",
            ]
            assert influence.shape == (20, 3)
            assert influence.iloc[:, :].isna().sum().sum() == 0

        # Clear test report dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @staticmethod
    def test_classifier_application_model() -> None:
        """Test application model training and application and validation model file existence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test train_application_model
            model_eva = modeleva_initialization_classification(temp_dir, data_split="60-40-split")

            model_eva._data_split()
            model_eva._classifier_validation()
            assert model_eva.app_model is None

            model_eva.train_application_model()
            assert type(model_eva.app_model) is LogisticRegression

            # Test default app_model training in classifier_evaluation
            model_eva = modeleva_initialization_classification(temp_dir, data_split="60-40-split")
            assert model_eva.app_model is None
            model_eva.classifier_evaluation()
            assert type(model_eva.app_model) is LogisticRegression

            # Test saved models
            val_model_path = (
                model_eva._report_directory
                + "Model_evaluation_reports/"
                + f"Data_{model_eva.data_label}_Model_{model_eva.model_label}/Model_in_validation/"
                + f"val_model_fold-0_{model_eva._model_label}.dill"
            )
            assert os.path.exists(val_model_path)
            app_model_path = (
                model_eva._report_directory
                + "Model_evaluation_reports/"
                + f"Data_{model_eva.data_label}_Model_{model_eva.model_label}/Model_for_application/"
                + f"app_model_{model_eva._model_label}.dill"
            )
            assert os.path.exists(app_model_path)

        # Clear test report dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    # Regression model evaluation tests
    @staticmethod
    def test_regressor_data_split() -> None:
        """Test data split functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # K-fold
            model_eva = modeleva_initialization_regression(temp_dir, data_split="5-fold")

            assert model_eva.dsp_inds == []
            model_eva._data_split()
            assert len(model_eva.dsp_inds) == 5
            assert len(model_eva.dsp_inds[0][0]) == 8
            assert len(model_eva.dsp_inds[0][1]) == 2

            # LOO
            model_eva = modeleva_initialization_regression(temp_dir, data_split="loo")

            assert model_eva.dsp_inds == []
            model_eva._data_split()
            assert len(model_eva.dsp_inds) == 10
            assert len(model_eva.dsp_inds[0][0]) == 9
            assert len(model_eva.dsp_inds[0][1]) == 1

            # Train-test-split
            model_eva = modeleva_initialization_regression(temp_dir, data_split="60-40-split")

            assert model_eva.dsp_inds == []
            model_eva._data_split()
            assert len(model_eva.dsp_inds) == 1
            assert len(model_eva.dsp_inds[0][0]) == 6
            assert len(model_eva.dsp_inds[0][1]) == 4

        # Clear test report dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @staticmethod
    def test_regressor_validation() -> None:
        """Test data split functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_eva = modeleva_initialization_regression(temp_dir)
            model_eva._data_split()
            model_time = model_eva.model_time

            assert len(model_time) > 0
            assert model_eva.y_true_eva is None
            assert model_eva.y_pred_eva is None
            assert model_eva.y_true_proba_eva is None
            assert model_eva.y_pred_proba_eva is None
            assert model_eva.sid_eva == []

            model_eva._regressor_validation()

            assert len(model_time) > 0
            assert model_eva.y is not None
            assert model_eva.sid_eva is not None
            assert model_eva.y_true_eva is not None
            assert model_eva.y_pred_eva is not None
            assert model_eva.y_true_proba_eva is None
            assert model_eva.y_pred_proba_eva is None
            assert len(model_eva.y_true_eva) == len(model_eva.y)
            assert len(model_eva.y_pred_eva) == len(model_eva.y)
            assert len(model_eva.sid_eva) > 0

        # Clear test report dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @staticmethod
    def test_regressor_evaluation() -> None:
        """Test classifier evaluation workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_eva = modeleva_initialization_regression(temp_dir, data_split="5-fold")
            model_eva.regressor_evaluation()
            metrics_filename = f"Regression_performance_{model_eva.model_label}.csv"
            scatter_plot_filename = f"Scatter_plot_{model_eva.model_label}.png"
            residual_plot_filename = f"Residual_plot_{model_eva.model_label}.png"
            residual_filename = f"Residual_analysis_{model_eva.model_label}.csv"
            influence_analysis_filename = f"Influence_analysis_{model_eva.model_label}.csv"
            output_path = (
                model_eva._report_directory
                + f"Model_evaluation_reports/Data_{model_eva.data_label}_Model_{model_eva.model_label}/"
            )

            # Test metrics report output
            metrics_path = output_path + metrics_filename
            assert os.path.exists(metrics_path)
            try:
                metrics = pd.read_csv(metrics_path)
                assert True
            except Exception as e:
                raise AssertionError(f"Failed to read CSV file '{metrics_filename}': {e}") from e
            assert list(metrics.columns) == [
                "Mean_Error",
                "Standard_Deviation_of_Error",
                "Mean_Absolute_Error",
                "Normalized_MAE",
                "CV_MAE",
                "Mean_Squared_Error",
                "Root_Mean_Squared_Error",
                "Normalized_RMSE",
                "CV_RMSE",
                "Residual_Prediction_Deviation",
                "R2",
            ]
            assert metrics.shape == (1, 11)
            assert metrics.iloc[:, :].isna().sum().sum() == 0

            # Test scatter plot and residual plot existence (manual check for plot image required)
            scatter_plot_path = output_path + scatter_plot_filename
            assert os.path.exists(scatter_plot_path)
            residual_plot_path = output_path + residual_plot_filename
            assert os.path.exists(residual_plot_path)

            # Test residual report output
            residual_path = output_path + residual_filename
            assert os.path.exists(residual_path)
            try:
                residual = pd.read_csv(residual_path)
                assert True
            except Exception as e:
                raise AssertionError(f"Failed to read CSV file '{residual_filename}': {e}") from e
            assert residual.shape == (10, 5)
            assert residual.iloc[:, :].isna().sum().sum() == 0

            # Test Cook's distance report output
            influence_analysis_path = output_path + influence_analysis_filename
            assert os.path.exists(influence_analysis_path)
            try:
                influence = pd.read_csv(influence_analysis_path)
                assert True
            except Exception as e:
                raise AssertionError(f"Failed to read CSV file '{influence_analysis_filename}': {e}") from e
            assert list(influence.columns) == ["Sample_ID", "Cooks_distance_like"]
            assert influence.shape == (10, 2)
            assert influence.iloc[:, :].isna().sum().sum() == 0

        # Clear test report dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @staticmethod
    def test_regressor_application_model() -> None:
        """Test application model training and application and validation model file existence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test train_application_model
            model_eva = modeleva_initialization_regression(temp_dir, data_split="60-40-split")

            model_eva._data_split()
            model_eva._regressor_validation()
            assert model_eva.app_model is None

            model_eva.train_application_model()
            assert type(model_eva.app_model) is LinearRegression

            # Test default app_model training in regressor_evaluation
            model_eva = modeleva_initialization_regression(temp_dir, data_split="60-40-split")
            assert model_eva.app_model is None
            model_eva.regressor_evaluation()
            assert type(model_eva.app_model) is LinearRegression

            # Test saved models
            val_model_path = (
                model_eva._report_directory
                + "Model_evaluation_reports/"
                + f"Data_{model_eva.data_label}_Model_{model_eva.model_label}/Model_in_validation/"
                + f"val_model_fold-0_{model_eva._model_label}.dill"
            )
            assert os.path.exists(val_model_path)
            app_model_path = (
                model_eva._report_directory
                + "Model_evaluation_reports/"
                + f"Data_{model_eva.data_label}_Model_{model_eva.model_label}/Model_for_application/"
                + f"app_model_{model_eva._model_label}.dill"
            )
            assert os.path.exists(app_model_path)

        # Clear test report dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# %% Tests - ModelEva


# TestModelEva.test_initialization_regression()
# TestModelEva.test_initialization_classification()
# TestModelEva.test_auto_choose_classification_regression()
# TestModelEva.test_auto_kfold_to_loocv()

# TestModelEva.test_property_access()

# TestModelEva.test_update_samples()

# TestModelEva.test_classifier_data_split()
# TestModelEva.test_classifier_validation()
# TestModelEva.test_classifier_evaluation()
# TestModelEva.test_classifier_application_model()

# TestModelEva.test_regressor_data_split()
# TestModelEva.test_regressor_validation()
# TestModelEva.test_regressor_evaluation()
# TestModelEva.test_regressor_application_model()


# %% Test main

if __name__ == "__main__":
    unittest.main()
