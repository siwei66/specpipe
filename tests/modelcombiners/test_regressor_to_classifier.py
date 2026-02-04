# -*- coding: utf-8 -*-
"""
Tests for regressor_to_classifier tools

The following source code was created with AI assistance and has been human reviewed and edited.

Copyright (c) 2025 Siwei Luo. MIT License.
"""

import sys
import numpy as np
import pytest

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from swectral.modelcombiners import RegressorToClassifier, regressor_to_classifier


# %% TestRegressorToClassifier


class TestRegressorToClassifier:
    """Tests for RegressorToClassifier functionalities."""

    @staticmethod
    def test_single_output_fit_and_predict() -> None:
        """Test functionality for models that supports single output only."""
        # Simple single-output regressor
        X = np.array([[1], [5], [3], [4], [7], [8]])  # noqa: N806
        y = np.array(["0", "1", "0", "1", "2", "2"])
        reg = SVR()
        clf = RegressorToClassifier(regressor=reg)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        labels = clf.predict(X)

        assert proba.shape == (6, 3)
        assert labels.shape == (6,)
        assert np.allclose(proba.sum(axis=1), 1)
        assert clf.regressors_ is not None
        assert len(clf.regressors_) > 1

    @staticmethod
    def test_multi_output_fit_and_predict() -> None:
        """Test functionality for models that supports multi-output."""
        # Multi-output regressor
        X = np.array([[1], [5], [3], [4], [7], [8]])  # noqa: N806
        y = np.array(["0", "1", "0", "1", "2", "2"])
        reg = LinearRegression()
        clf = RegressorToClassifier(regressor=reg)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        labels = clf.predict(X)

        assert proba.shape == (6, 3)
        assert labels.shape == (6,)
        assert np.allclose(proba.sum(axis=1), 1)
        assert clf.regressors_ is not None
        assert len(clf.regressors_) == 1

    @staticmethod
    def test_one_class() -> None:
        """Test one class classification."""
        # One-class data
        X = np.array([[1], [2], [3]])  # noqa: N806
        y = np.array([0, 0, 0])
        reg = LinearRegression()
        clf = RegressorToClassifier(regressor=reg)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        labels = clf.predict(X)

        assert proba.shape == (3, 1)
        assert np.allclose(proba, 1)
        assert np.all(labels == str(0))

    @staticmethod
    def test_probability_functions() -> None:
        """Test different probability functions."""

        # Function names - default softmax, alt sigmoid
        X = np.array([[1], [2], [3], [4]])  # noqa: N806
        y = np.array([0, 1, 0, 1])
        reg = LinearRegression()
        clf = RegressorToClassifier(regressor=reg, proba_func="sigmoid")
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        labels = clf.predict(X)

        assert proba.shape == (4, 2)
        assert labels.shape == (4,)
        assert np.allclose(proba.sum(axis=1), 1)

        # Custom probability function (square and normalize)
        def square_normalize(x: np.ndarray) -> np.ndarray:
            x = np.square(x)
            row_sums = x.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            result = x / row_sums
            assert isinstance(result, np.ndarray)
            return result

        X = np.array([[1], [2], [3], [4]])  # noqa: N806
        y = np.array([0, 1, 0, 1])
        reg = LinearRegression()
        clf = RegressorToClassifier(regressor=reg, proba_func=square_normalize)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        labels = clf.predict(X)

        assert proba.shape == (4, 2)
        assert labels.shape == (4,)

    @staticmethod
    def test_unfitted_error() -> None:
        """Test unfitted error."""
        # Check error raised if predicting before fit
        X = np.array([[1], [2]])  # noqa: N806
        reg = LinearRegression()
        clf = RegressorToClassifier(regressor=reg)

        with pytest.raises(ValueError, match="not fitted"):
            clf.predict(X)

        with pytest.raises(ValueError, match="not fitted"):
            clf.predict_proba(X)

    @staticmethod
    def test_regressor_to_classifier_creator() -> None:
        # Default name
        X = np.array([[1], [2], [3], [4]])  # noqa: N806
        y = np.array([0, 1, 0, 1])
        reg = LinearRegression()
        clf = regressor_to_classifier(regressor=reg)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        labels = clf.predict(X)

        assert proba.shape == (4, 2)
        assert labels.shape == (4,)
        assert np.allclose(proba.sum(axis=1), 1)
        assert clf.__class__.__name__ == "LinearRegressionClassifier"

        # Custom name
        X = np.array([[1], [2], [3], [4]])  # noqa: N806
        y = np.array([0, 1, 0, 1])
        reg = LinearRegression()
        clf = regressor_to_classifier(regressor=reg, name="CustomClassName")
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        labels = clf.predict(X)

        assert proba.shape == (4, 2)
        assert labels.shape == (4,)
        assert np.allclose(proba.sum(axis=1), 1)
        assert clf.__class__.__name__ == "CustomClassName"


# %% Local test: TestRegressorToClassifier

# TestRegressorToClassifier.test_single_output_fit_and_predict()
# TestRegressorToClassifier.test_multi_output_fit_and_predict()
# TestRegressorToClassifier.test_one_class()
# TestRegressorToClassifier.test_probability_functions()
# TestRegressorToClassifier.test_unfitted_error()
# TestRegressorToClassifier.test_regressor_to_classifier_creator()


# %% Test main

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
