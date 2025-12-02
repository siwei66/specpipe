# -*- coding: utf-8 -*-
"""
SpecPipe model connectors - connect data transformer and estimator
For evaluation of dimension reduction, feature selection or other feature engineering models that requires fitting.

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# ruff: noqa: I001
# ruff: noqa: N806

# Basic
import numpy as np  # noqa: E402
from typing import Any, Annotated  # noqa: E402

# Model
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin  # noqa: E402
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted  # noqa: E402
from sklearn.metrics import accuracy_score, r2_score  # noqa: E402

# Local
from ..specio import simple_type_validator, arraylike_validator  # noqa: E402

# # For local test
# from specpipe.specio import simple_type_validator, arraylike_validator


# %% Model validators


def validate_transformer(data_transformer: object) -> None:
    """Scikit-learn style transformer validator"""
    if isinstance(data_transformer, type):
        raise TypeError(f"Expected a transformer instance, but got class {data_transformer.__name__}.")
    if callable(data_transformer):
        raise TypeError(f"Expected a transformer instance, but got function {data_transformer.__name__}.")  # type: ignore[attr-defined]
    native_data_type = (int, float, str, bool, list, dict, tuple, set, bytes, type(None))
    if type(data_transformer) in native_data_type:
        raise TypeError(f"Expected a transformer instance, but got '{type(data_transformer)}'.")
    if not hasattr(data_transformer, "fit") or not hasattr(data_transformer, "transform"):
        raise ValueError("Expected a transformer instance with 'fit' and 'transform' methods.")


def validate_classifier(classifier: object) -> None:
    """Scikit-learn style classifier validator"""
    if isinstance(classifier, type):
        raise TypeError(f"Expected a classifier instance, but got class {classifier.__name__}.")
    if callable(classifier):
        raise TypeError(f"Expected a classifier instance, but got function {classifier.__name__}.")  # type: ignore[attr-defined]
    native_data_type = (int, float, str, bool, list, dict, tuple, set, bytes, type(None))
    if type(classifier) in native_data_type:
        raise TypeError(f"Expected a classifier instance, but got '{type(classifier)}'.")
    if not hasattr(classifier, "fit") or not hasattr(classifier, "predict") or not hasattr(classifier, "predict_proba"):
        raise ValueError("Expected a classifier instance with 'fit', 'predict' and 'predict_proba' methods.")


def validate_regressor(regressor: object) -> None:
    """Scikit-learn style regressor validator"""
    if isinstance(regressor, type):
        raise TypeError(f"Expected a regressor instance, but got class {regressor.__name__}.")
    if callable(regressor):
        raise TypeError(f"Expected a regressor instance, but got function {regressor.__name__}.")  # type: ignore[attr-defined]
    native_data_type = (int, float, str, bool, list, dict, tuple, set, bytes, type(None))
    if type(regressor) in native_data_type:
        raise TypeError(f"Expected a regressor instance, but got '{type(regressor)}'.")
    if not hasattr(regressor, "fit") or not hasattr(regressor, "predict"):
        raise ValueError("Expected a regressor instance with 'fit' and 'predict' methods.")


# %% Model pipeline tools


# Constructor - combined classifier
def combine_transformer_classifier(data_transformer: object, classifier: object) -> object:
    """
    Combine a data transformation model with a classifier into a unified estimator that preserves component names.
    This wrapper functions similarly to scikit-learn's Pipeline but compatible with any transformer and classifier that follows scikit-learn's method conventions.

    Parameters:
    -----------
    data_transformer
        Data transformation model, any data transformation or feature selection model.
    classifier
        Classification model.

    Returns:
    --------
    model
        Combined classification model.
    """  # noqa: E501
    # Validate input models
    validate_transformer(data_transformer)
    validate_classifier(classifier)

    # Create combined name
    combined_name = f"{data_transformer.__class__.__name__}_{classifier.__class__.__name__}"

    # Create new model class to customize name
    class CombinedModel(TransClassifier):
        def __repr__(self) -> str:
            return (
                "TransClassifier"
                + f"(transformer={self.data_transformer.__class__.__name__}, "
                + f"classifier={self.classifier.__class__.__name__})"
            )

        def __str__(self) -> str:
            return combined_name

    # Customize name
    CombinedModel.__name__ = combined_name
    CombinedModel.__qualname__ = combined_name

    # Create model instance
    combined_model = CombinedModel(data_transformer=data_transformer, classifier=classifier)

    return combined_model


# Constructor - combined regressor
def combine_transformer_regressor(data_transformer: object, regressor: object) -> object:
    """
    Combine a data transformation model with a regressor into a unified estimator that preserves component names.
    This wrapper functions similarly to scikit-learn's Pipeline but compatible with any transformer and regressor that follows scikit-learn's method conventions.

    Parameters:
    -----------
    data_transformer
        Data transformation model, any data transformation or feature selection model.
    regressor
        Regression model.

    Returns:
    --------
    model
        Combined regression model.
    """  # noqa: E501
    # Validate input models
    validate_transformer(data_transformer)
    validate_regressor(regressor)

    # Create combined name
    combined_name = f"{data_transformer.__class__.__name__}_{regressor.__class__.__name__}"

    # Create new model class to customize name
    class CombinedModel(TransRegressor):
        def __repr__(self) -> str:
            return (
                "TransRegressor"
                + f"(transformer={self.data_transformer.__class__.__name__}, "
                + f"regressor={self.regressor.__class__.__name__})"
            )

        def __str__(self) -> str:
            return combined_name

    # Customize name
    CombinedModel.__name__ = combined_name
    CombinedModel.__qualname__ = combined_name

    # Create model instance
    combined_model = CombinedModel(data_transformer=data_transformer, regressor=regressor)

    return combined_model


# Combined classifier
class TransClassifier(BaseEstimator, ClassifierMixin):
    """
    Combine a data transformation model with a classifier into a unified estimator.

    This wrapper functions similarly to scikit-learn's Pipeline but compatible with any transformer and classifier that follows scikit-learn's method conventions.


    Attributes:
    -----------
    data_transformer
        Data transformation model, any data transformation or feature selection model.

    classifier
        Classification model.


    Methods:
    --------
    fit(X, y)
        Fit the transformer on X, then fit the classifier on transformed X.

    transform(X)
        Transform X using the fitted transformer.

    predict(X)
        Transform X using the fitted transformer, then predict using the fitted classifier.

    predict_proba(X)
        Predict the probability of X using the fitted transformer and classifier.

    score(X, y)
        Compute the accuracy score of the fitted models on the provided X and y.
    """  # noqa: E501

    def __init__(self, data_transformer: object, classifier: object) -> None:
        validate_transformer(data_transformer)
        self.data_transformer = data_transformer
        validate_classifier(classifier)
        self.classifier = classifier

    @simple_type_validator
    def fit(
        self,
        X: Annotated[Any, arraylike_validator(ndim=2)],  # noqa: N803
        y: Annotated[Any, arraylike_validator(ndim=1)],
    ) -> 'TransClassifier':
        """
        Fit the transformer on X, then fit the classifier on transformed X.

        Parameters
        ----------
        X : 2D array-like
            Training dataset.
        y : Annotated[Any, arraylike_validator(ndim, optional
            Training target values.

        Returns
        -------
        TransRegressor
            The fitted combined model.
        """
        # Validate inputs
        X, y = check_X_y(X, y)
        # Fit transformer and transform X_train
        assert hasattr(self.data_transformer, 'fit')
        try:
            self.data_transformer.fit(X)  # Try unsupervised
        except Exception:
            self.data_transformer.fit(X, y)  # Try supervised
        assert hasattr(self.data_transformer, 'transform')
        X_transformed = self.data_transformer.transform(X)
        # Fit classifier
        assert hasattr(self.classifier, 'fit')
        self.classifier.fit(X_transformed, y)
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, 'is_fitted_') and self.is_fitted_

    @simple_type_validator
    def transform(self, X: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:  # noqa: N803
        """
        Transform X using the fitted transformer.

        Parameters
        ----------
        X : 2D array-like
            Training dataset.

        Returns
        -------
        np.ndarray
            Transformed X.
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        assert hasattr(self.data_transformer, 'transform')
        X_transformed = self.data_transformer.transform(X)
        X_transformed = np.array(X_transformed)
        return X_transformed

    @simple_type_validator
    def predict(self, X: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:  # noqa: N803
        """
        Transform X using the fitted transformer, then predict targets using the fitted classifier.

        Parameters
        ----------
        X : 2D array-like
            Dataset to predict.

        Returns
        -------
        np.ndarray
            Predicted target values of X.
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        # Transform X_pred
        assert hasattr(self.data_transformer, 'transform')
        X_transformed = self.data_transformer.transform(X)
        # Predict using classifier
        assert hasattr(self.classifier, 'predict')
        y_pred = self.classifier.predict(X_transformed)
        y_pred = np.array(y_pred)
        return y_pred

    @simple_type_validator
    def predict_proba(self, X: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:  # noqa: N803
        """
        Transform X using the fitted transformer, then predict the probabilities of the targets using the fitted classifier.

        Parameters
        ----------
        X : 2D array-like
            Dataset to predict.

        Returns
        -------
        np.ndarray
            Predicted target values of X.
        """  # noqa: E501
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        assert hasattr(self.data_transformer, 'transform')
        X_transformed = self.data_transformer.transform(X)
        assert hasattr(self.classifier, 'predict_proba')
        y_pred_proba = self.classifier.predict_proba(X_transformed)
        y_pred_proba = np.array(y_pred_proba)
        return y_pred_proba

    @simple_type_validator
    def score(
        self,
        X: Annotated[Any, arraylike_validator(ndim=2)],  # noqa: N803
        y: Annotated[Any, arraylike_validator(ndim=1)],
    ) -> float:
        """
        Compute overall accuracy score of the fitted models on the provided X and y.

        Parameters
        ----------
        X : 2D array-like
            Test dataset.
        y : Annotated[Any, arraylike_validator(ndim, optional
            Test target values.

        Returns
        -------
        TransRegressor
            The fitted combined model.
        """
        check_is_fitted(self, 'is_fitted_')
        assert hasattr(self.data_transformer, 'transform')
        X_transformed = self.data_transformer.transform(X)
        if hasattr(self.classifier, 'score'):
            assert hasattr(self.classifier, 'score')
            overall_accuracy = self.classifier.score(X_transformed, y)
        else:
            assert hasattr(self.classifier, 'predict')
            y_pred = self.classifier.predict(X_transformed)
            overall_accuracy = accuracy_score(y, y_pred)
        overall_accuracy = float(overall_accuracy)
        return overall_accuracy


# Combined regressor
class TransRegressor(BaseEstimator, RegressorMixin):
    """
    Combine a data transformation model with a regressor into a unified estimator.

    This wrapper functions similarly to scikit-learn's Pipeline but compatible with any transformer and regressor that follows scikit-learn's method conventions.


    Attributes:
    -----------
    data_transformer
        Data transformation model, any data transformation or feature selection model.

    regressor
        Regression model.


    Methods:
    --------
    fit(X, y)
        Fit the transformer on X, then fit the regressor on transformed X.

    transform(X)
        Transform X using the fitted transformer.

    predict(X)
        Transform X using the fitted transformer, then predict using the fitted regressor.

    score(X, y)
        Compute the goodness of fit score of the fitted models on the provided X and y.
    """  # noqa: E501

    def __init__(self, data_transformer: object, regressor: object) -> None:
        validate_transformer(data_transformer)
        self.data_transformer = data_transformer
        validate_regressor(regressor)
        self.regressor = regressor

    @simple_type_validator
    def fit(
        self,
        X: Annotated[Any, arraylike_validator(ndim=2)],  # noqa: N803
        y: Annotated[Any, arraylike_validator(ndim=1)],
    ) -> 'TransRegressor':
        """
        Fit the transformer on X, then fit the regressor on transformed X.

        Parameters
        ----------
        X : 2D array-like
            Training dataset.
        y : Annotated[Any, arraylike_validator(ndim, optional
            Training target values.

        Returns
        -------
        TransRegressor
            The fitted combined model.
        """
        # Validate inputs
        X, y = check_X_y(X, y)
        # Fit transformer and transform X_train
        assert hasattr(self.data_transformer, 'fit')
        try:
            self.data_transformer.fit(X)  # Try unsupervised
        except Exception:
            self.data_transformer.fit(X, y)  # Try supervised
        assert hasattr(self.data_transformer, 'transform')
        X_transformed = self.data_transformer.transform(X)
        # Fit regressor
        assert hasattr(self.regressor, 'fit')
        self.regressor.fit(X_transformed, y)
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, 'is_fitted_') and self.is_fitted_

    @simple_type_validator
    def transform(self, X: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:  # noqa: N803
        """
        Transform X using the fitted transformer.

        Parameters
        ----------
        X : 2D array-like
            Training dataset.

        Returns
        -------
        np.ndarray
            Transformed X.
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        assert hasattr(self.data_transformer, 'transform')
        X_transformed = self.data_transformer.transform(X)
        X_transformed = np.array(X_transformed)
        return X_transformed

    @simple_type_validator
    def predict(self, X: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:  # noqa: N803
        """
        Transform X using the fitted transformer, then predict targets using the fitted regressor.

        Parameters
        ----------
        X : 2D array-like
            Dataset to predict.

        Returns
        -------
        np.ndarray
            Predicted target values of X.
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        # Transform X_pred
        assert hasattr(self.data_transformer, 'transform')
        X_transformed = self.data_transformer.transform(X)
        # Predict using regressor
        assert hasattr(self.regressor, 'predict')
        y_pred = self.regressor.predict(X_transformed)
        y_pred = np.array(y_pred)
        return y_pred

    @simple_type_validator
    def score(
        self,
        X: Annotated[Any, arraylike_validator(ndim=2)],  # noqa: N803
        y: Annotated[Any, arraylike_validator(ndim=1)],
    ) -> float:
        """
        Compute the goodness of fit score of the fitted models on the provided X and y.

        Parameters
        ----------
        X : 2D array-like
            Test dataset.
        y : Annotated[Any, arraylike_validator(ndim, optional
            Test target values.

        Returns
        -------
        TransRegressor
            The fitted combined model.
        """
        check_is_fitted(self, 'is_fitted_')
        assert hasattr(self.data_transformer, 'transform')
        X_transformed = self.data_transformer.transform(X)
        if hasattr(self.regressor, 'score'):
            assert hasattr(self.regressor, 'score')
            gof_score = self.regressor.score(X_transformed, y)
        else:
            assert hasattr(self.regressor, 'predict')
            y_pred = self.regressor.predict(X_transformed)
            gof_score = r2_score(y, y_pred)
        gof_score = float(gof_score)
        return gof_score
