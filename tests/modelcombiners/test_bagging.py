# -*- coding: utf-8 -*-
"""
Tests for model bagging ensembler

The following source code was created with AI assistance and has been human reviewed and edited.

Copyright (c) 2025 Siwei Luo. MIT License.
"""

import sys
import numpy as np
import pytest

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_regression, make_classification
from sklearn.base import BaseEstimator, RegressorMixin

from swectral.modelcombiners import BaggingEnsembler


# %% Helper: ConstantRegressor


class ConstantRegressor(BaseEstimator, RegressorMixin):
    """Mock regressor with constant output."""

    def __init__(self, value: float) -> None:
        self.value = value

    def fit(self, X: np.ndarray, y: np.ndarray) -> object:  # noqa: N803
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        result = np.asarray(np.full(X.shape[0], self.value))
        assert isinstance(result, np.ndarray)
        return result


# %% test functions: BaggingEnsembler


class TestBaggingEnsembler:
    """Tests for BaggingEnsembler."""

    @staticmethod
    def test_regression_mean_aggregation() -> None:
        """Tests for mean aggregation of regressors."""
        X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=0)  # noqa: N806

        model = BaggingEnsembler(
            base_estimator=LinearRegression(),
            n_estimators=10,
            regressor_aggregate="mean",
            random_state=42,
        )
        model.fit(X, y)
        y_pred = model.predict(X)

        assert y_pred.shape == (X.shape[0],)
        assert np.isfinite(y_pred).all()

    @staticmethod
    def test_regression_median_aggregation() -> None:
        """Tests for median aggregation of regressors."""
        X, y = make_regression(n_samples=150, n_features=4, noise=1.0, random_state=1)  # noqa: N806

        model = BaggingEnsembler(
            base_estimator=LinearRegression(),
            n_estimators=15,
            regressor_aggregate="median",
            random_state=0,
        )
        model.fit(X, y)
        y_pred = model.predict(X)

        assert y_pred.shape == (X.shape[0],)
        assert np.isfinite(y_pred).all()

    @staticmethod
    def test_regression_trimmed_mean() -> None:
        """Tests for trimmed mean aggregation of regressors."""
        X, y = make_regression(n_samples=120, n_features=3, noise=5.0, random_state=2)  # noqa: N806

        model = BaggingEnsembler(
            base_estimator=LinearRegression(),
            n_estimators=20,
            regressor_aggregate=(0.2, 0.8),
            random_state=1,
        )
        model.fit(X, y)
        y_pred = model.predict(X)

        assert y_pred.shape == (X.shape[0],)
        assert not np.isnan(y_pred).any()

    @staticmethod
    def test_regression_trimmed_mean_fallback() -> None:
        """Tests fallback situation for trimmed mean aggregation of regressors."""
        X = np.zeros((3, 2))  # noqa: N806

        estimators = [
            ConstantRegressor(0.0),
            ConstantRegressor(10.0),
        ]

        model = BaggingEnsembler(
            base_estimator=ConstantRegressor(0.0),
            n_estimators=2,
            regressor_aggregate=(0.49, 0.51),
        )

        model.estimators_ = estimators
        model._is_classifier = False

        y_pred = model.predict(X)

        assert np.all(np.isin(y_pred, [0.0, 10.0]))

    @staticmethod
    def test_classifier_predict_and_predict_proba() -> None:
        """Tests basic functionalities for classifiers."""
        X, y = make_classification(  # noqa: N806
            n_samples=200,
            n_features=6,
            n_classes=2,
            random_state=0,
        )

        model = BaggingEnsembler(
            base_estimator=LogisticRegression(max_iter=1000),
            n_estimators=10,
            random_state=42,
        )
        model.fit(X, y)

        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        assert y_pred.shape == (X.shape[0],)
        assert y_proba.shape == (X.shape[0], len(model.classes_))
        assert np.allclose(y_proba.sum(axis=1), 1.0)

    @staticmethod
    def test_classifier_classes_attribute() -> None:
        """Tests for classes_ attribute for classifiers."""
        X, y = make_classification(  # noqa: N806
            n_samples=100,
            n_features=5,
            n_classes=3,
            n_informative=3,
            random_state=3,
        )

        model = BaggingEnsembler(
            base_estimator=LogisticRegression(max_iter=1000),
            n_estimators=5,
            random_state=0,
        )
        model.fit(X, y)

        assert hasattr(model, "classes_")
        assert np.array_equal(np.sort(model.classes_), np.unique(y))

    @staticmethod
    def test_limit_proba_clipping() -> None:
        """Tests for limit_proba functionality for classifiers."""
        X, y = make_classification(  # noqa: N806
            n_samples=150,
            n_features=4,
            n_classes=2,
            random_state=4,
        )

        model = BaggingEnsembler(
            base_estimator=LogisticRegression(max_iter=1000),
            n_estimators=10,
            limit_proba=(0.2, 0.8),
            random_state=1,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.min() >= 0.2
        assert proba.max() <= 0.8

    @staticmethod
    def test_predict_proba_raises_for_regressor() -> None:
        """Tests for predict_proba error raising for regressors."""
        X, y = make_regression(n_samples=100, n_features=4, random_state=5)  # noqa: N806

        model = BaggingEnsembler(
            base_estimator=LinearRegression(),
            n_estimators=5,
            random_state=0,
        )
        model.fit(X, y)

        with pytest.raises(AttributeError):
            model.predict_proba(X)

    @staticmethod
    def test_reproducibility_with_random_state() -> None:
        """Tests for reproducibility with random state."""
        X, y = make_regression(n_samples=120, n_features=3, random_state=6)  # noqa: N806

        model1 = BaggingEnsembler(
            base_estimator=LinearRegression(),
            n_estimators=10,
            random_state=123,
        )
        model2 = BaggingEnsembler(
            base_estimator=LinearRegression(),
            n_estimators=10,
            random_state=123,
        )

        model1.fit(X, y)
        model2.fit(X, y)

        y_pred1 = model1.predict(X)
        y_pred2 = model2.predict(X)

        assert np.allclose(y_pred1, y_pred2)


# %% Test: BaggingEnsembler

# TestBaggingEnsembler.test_classifier_classes_attribute()
# TestBaggingEnsembler.test_classifier_predict_and_predict_proba()
# TestBaggingEnsembler.test_limit_proba_clipping()
# TestBaggingEnsembler.test_predict_proba_raises_for_regressor()
# TestBaggingEnsembler.test_regression_mean_aggregation()
# TestBaggingEnsembler.test_regression_median_aggregation()
# TestBaggingEnsembler.test_regression_trimmed_mean()
# TestBaggingEnsembler.test_regression_trimmed_mean_fallback()
# TestBaggingEnsembler.test_reproducibility_with_random_state()


# %% Test main


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
