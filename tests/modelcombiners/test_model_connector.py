# -*- coding: utf-8 -*-
"""
Test model connectors - transformer-estimator connectors

Copyright (c) 2025 Siwei Luo. MIT License.
"""

import numpy as np
import pytest
import sys
from typing import Optional

from sklearn.decomposition import PCA  # Unsupervised transformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression  # Supervised transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from specpipe.specio import simple_type_validator
from specpipe.vegeind.demo_data import create_vegeind_demo_data

from specpipe.modelcombiners.model_connector import combine_transformer_classifier, combine_transformer_regressor


# %% Helper - create test data


@simple_type_validator
def create_model_test_data(
    is_classification: bool = True, nsample: int = 20, nfeature: int = 462, seed: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create classification test data."""
    incr_id = nsample // 3
    incr_idt = nsample // 6
    # X train
    X_train = np.array(create_vegeind_demo_data(nsample=nsample, nband=nfeature))  # noqa: N806
    X_train[0:incr_id] = X_train[0:incr_id] + 5000
    X_train[incr_id : (incr_id * 2)] = X_train[incr_id : (incr_id * 2)] + 10000

    # y train
    if is_classification:
        y_train = np.full((nsample,), 'c')
        y_train[0:incr_id] = 'a'
        y_train[incr_id : (incr_id * 2)] = 'b'
    else:
        y_train = np.zeros((nsample,))
        y_train[0:incr_id] = np.random.random((incr_id,)) + 0.5
        y_train[incr_id : (incr_id * 2)] = np.random.random((incr_id,)) + 1.0

    # Test data
    X_test = np.array(create_vegeind_demo_data(nsample=nsample // 2))  # noqa: N806
    X_test[0:incr_idt] = X_test[0:incr_idt] + 5000
    X_test[incr_idt : (incr_idt * 2)] = X_test[incr_idt : (incr_idt * 2)] + 10000

    # y test
    if is_classification:
        y_test = np.full((nsample // 2,), 'c')
        y_test[0:incr_idt] = 'a'
        y_test[incr_idt : (incr_idt * 2)] = 'b'
    else:
        y_test = np.zeros((nsample // 2,))
        y_test[0:incr_idt] = np.random.random((incr_idt,)) + 0.5
        y_test[incr_idt : (incr_idt * 2)] = np.random.random((incr_idt,)) + 1.0

    return X_train, y_train, X_test, y_test


# %% Tests


class TestCombineTransformerClassifier:
    """Test combine_transformer_classifier functionalities."""

    @staticmethod
    def test_supervised_transformer() -> None:
        """Test supervised transformer."""
        # Data
        test_data = create_model_test_data(nsample=100)
        X_train = test_data[0]  # noqa: N806
        y_train = test_data[1]
        X_test = test_data[2]  # noqa: N806
        y_test = test_data[3]

        # Create model
        test_transformer = SelectKBest(f_classif, k=5)
        test_classifier = RandomForestClassifier(n_estimators=10)
        combined_model = combine_transformer_classifier(test_transformer, test_classifier)

        # Assert modeling
        assert hasattr(combined_model, 'fit')
        assert hasattr(combined_model, 'transform')
        assert hasattr(combined_model, 'predict')
        assert hasattr(combined_model, 'predict_proba')
        assert hasattr(combined_model, 'score')
        combined_model.fit(X_train, y_train)
        X_trans = combined_model.transform(X_test)  # noqa: N806
        assert isinstance(X_trans, np.ndarray)
        assert X_trans.shape == (X_test.shape[0], 5)
        y_est = combined_model.predict(X_test)
        assert isinstance(y_est, np.ndarray)
        assert y_est.shape == y_test.shape
        y_est_proba = combined_model.predict_proba(X_test)
        assert isinstance(y_est_proba, np.ndarray)
        assert y_est_proba.shape == (y_test.shape[0], 3)
        test_score = combined_model.score(X_test, y_test)
        assert test_score > 0

        # Assert dynamic name
        assert combined_model.__class__.__name__ == "SelectKBest_RandomForestClassifier"

    @staticmethod
    def test_unsupervised_transformer() -> None:
        """Test unsupervised transformer."""
        # Data
        test_data = create_model_test_data(nsample=100)
        X_train = test_data[0]  # noqa: N806
        y_train = test_data[1]
        X_test = test_data[2]  # noqa: N806
        y_test = test_data[3]

        # Create model
        test_transformer = PCA(n_components=5)
        test_classifier = RandomForestClassifier(n_estimators=10)
        combined_model = combine_transformer_classifier(test_transformer, test_classifier)

        # Assert modeling
        assert hasattr(combined_model, 'fit')
        assert hasattr(combined_model, 'transform')
        assert hasattr(combined_model, 'predict')
        assert hasattr(combined_model, 'predict_proba')
        assert hasattr(combined_model, 'score')
        combined_model.fit(X_train, y_train)
        X_trans = combined_model.transform(X_test)  # noqa: N806
        assert isinstance(X_trans, np.ndarray)
        assert X_trans.shape == (X_test.shape[0], 5)
        y_est = combined_model.predict(X_test)
        assert isinstance(y_est, np.ndarray)
        assert y_est.shape == y_test.shape
        y_est_proba = combined_model.predict_proba(X_test)
        assert isinstance(y_est_proba, np.ndarray)
        assert y_est_proba.shape == (y_test.shape[0], 3)
        test_score = combined_model.score(X_test, y_test)
        assert test_score > 0

        # Assert dynamic name
        assert combined_model.__class__.__name__ == "PCA_RandomForestClassifier"


# %%


class TestCombineTransformerRegressor:
    """Test combine_transformer_regressor functionalities."""

    @staticmethod
    def test_supervised_transformer() -> None:
        """Test supervised transformer."""
        # Data
        test_data = create_model_test_data(is_classification=False, nsample=100)
        X_train = test_data[0]  # noqa: N806
        y_train = test_data[1]
        X_test = test_data[2]  # noqa: N806
        y_test = test_data[3]

        # Create model
        test_transformer = SelectKBest(f_regression, k=5)
        test_regressor = RandomForestRegressor(n_estimators=10)
        combined_model = combine_transformer_regressor(test_transformer, test_regressor)

        # Assert modeling
        assert hasattr(combined_model, 'fit')
        assert hasattr(combined_model, 'transform')
        assert hasattr(combined_model, 'predict')
        assert hasattr(combined_model, 'score')
        combined_model.fit(X_train, y_train)
        X_trans = combined_model.transform(X_test)  # noqa: N806
        assert isinstance(X_trans, np.ndarray)
        assert X_trans.shape == (X_test.shape[0], 5)
        y_est = combined_model.predict(X_test)
        assert isinstance(y_est, np.ndarray)
        assert y_est.shape == y_test.shape
        test_score = combined_model.score(X_test, y_test)
        assert test_score > 0

        # Assert dynamic name
        assert combined_model.__class__.__name__ == "SelectKBest_RandomForestRegressor"

    @staticmethod
    def test_unsupervised_transformer() -> None:
        """Test unsupervised transformer."""
        # Data
        test_data = create_model_test_data(is_classification=False, nsample=100)
        X_train = test_data[0]  # noqa: N806
        y_train = test_data[1]
        X_test = test_data[2]  # noqa: N806
        y_test = test_data[3]

        # Create model
        test_transformer = PCA(n_components=5)
        test_regressor = RandomForestRegressor(n_estimators=10)
        combined_model = combine_transformer_regressor(test_transformer, test_regressor)

        # Assert modeling
        assert hasattr(combined_model, 'fit')
        assert hasattr(combined_model, 'transform')
        assert hasattr(combined_model, 'predict')
        assert hasattr(combined_model, 'score')
        combined_model.fit(X_train, y_train)
        X_trans = combined_model.transform(X_test)  # noqa: N806
        assert isinstance(X_trans, np.ndarray)
        assert X_trans.shape == (X_test.shape[0], 5)
        y_est = combined_model.predict(X_test)
        assert isinstance(y_est, np.ndarray)
        assert y_est.shape == y_test.shape
        test_score = combined_model.score(X_test, y_test)
        assert test_score > 0

        # Assert dynamic name
        assert combined_model.__class__.__name__ == "PCA_RandomForestRegressor"


# %% Test main

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
