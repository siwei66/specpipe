# -*- coding: utf-8 -*-
"""
Swectral - model combiners - Regressor to classifier wraper

The following source code was created with AI assistance and has been human reviewed and edited.

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Typing
from typing import Optional, Union, Annotated, Any, Callable

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax, expit
from sklearn.base import clone

# Local
from ..specio import simple_type_validator, arraylike_validator


# %% converted classifier creator


@simple_type_validator
def regressor_to_classifier(
    regressor: object, proba_func: Union[str, Callable] = 'softmax', name: Optional[str] = None
) -> object:
    """
    Create a classifier from a numpy-style regressor using one-hot encoding and probability regression.

    Parameters
    ----------
    regressor : object
        A regressor implementing ``fit(X, y)`` and ``predict(X)``.

    proba_func : str or Callable, optional
        Function to convert raw regressor outputs to probabilities. Choose between:

            - ``'softmax'`` : row-wise softmax for single-label, mutually exclusive classes
            - ``'sigmoid'`` : per-class sigmoid for multi-label problems
            - Callable : a custom probability function.

        Default is ``'softmax'``.

    name : str or None, optional
        Name of the created model class.
        If None, the class name is ``'<RegressorClassName>Classifier'``. Default is None.

    Returns
    -------
    object
        An numpy-style classifier instance.

    See Also
    --------
    RegressorToClassifier

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.array([[1], [2], [3], [4]])
    >>> y = np.array([0, 1, 0, 1])

    >>> reg = LinearRegression()

    >>> clf = regressor_to_classifier(regressor=reg, proba_func='softmax')
    >>> clf.fit(X, y)

    >>> clf.predict(X)

    >>> clf.predict_proba(X)
    """

    regressor_name = regressor.__class__.__name__
    if name is None:
        class_name = f"{regressor_name}Classifier"
    else:
        class_name = name

    # Dynamically create a subclass of BaggingEnsembler
    ToClassifierClass = type(class_name, (RegressorToClassifier,), {})  # noqa: N806

    # Return the created instance
    model: object = ToClassifierClass(regressor=regressor, proba_func=proba_func)
    return model


# %% RegressorToClassifier


class RegressorToClassifier:
    """
    Wrap a sklearn-style regressor into a sklearn-style classifier using one-hot encoding and probability regression.

    This class converts a regressor that predicts continuous outputs into a classifier by one-hot encoding the targets and applying a probability conversion function to the regressor outputs.

    Attributes
    ----------
    regressor : object
        A regressor implementing ``fit(X, y)`` and ``predict(X)``.

    proba_func : str or Callable, optional
        Function to convert raw regressor outputs to probabilities. Choose between:

            - ``'softmax'`` : row-wise softmax for single-label, mutually exclusive classes
            - ``'sigmoid'`` : per-class sigmoid for multi-label problems
            - Callable : a custom probability function.

        Default is ``'softmax'``.

    encoder : OneHotEncoder
        Encoder for transforming class labels to one-hot vectors.

    classes_ : numpy.ndarray of shape (n_classes,) or None
        Array of class labels seen during fitting.

    regressors_ : list of object or None
        Fitted regressors.
        Contains a single regressor if multi-output regression is supported, otherwise one regressor per class.

    Methods
    -------
    fit(X, y)
        Fit the regressor(s) on one-hot encoded class targets.
    predict_proba(X)
        Predict class probabilities for each sample.
    predict(X)
        Predict class labels for each sample.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.array([[1], [2], [3], [4]])
    >>> y = np.array([0, 1, 0, 1])

    >>> reg = LinearRegression()

    >>> clf = RegressorToClassifier(regressor=reg, proba_func='softmax')
    >>> clf.fit(X, y)

    >>> clf.predict(X)

    >>> clf.predict_proba(X)
    """  # noqa: E501

    @simple_type_validator
    def __init__(self, regressor: object, proba_func: Union[str, Callable] = 'softmax') -> None:
        self.regressor: object = regressor

        if isinstance(proba_func, str):
            if proba_func.lower() == 'softmax':
                self.proba_func: Callable = self._softmax
            elif proba_func.lower() == 'sigmoid':
                self.proba_func = self._sigmoid_normalize
            else:
                raise ValueError(f"proba_func must be 'softmax', 'sigmoid', or a callable function, got: {proba_func}.")
        elif callable(proba_func):
            self.proba_func = proba_func

        self.encoder: OneHotEncoder = OneHotEncoder(categories='auto', sparse_output=False)
        self.classes_: Optional[np.ndarray] = None
        self.regressors_: Optional[list[object]] = None

    @staticmethod
    def _softmax(X: np.ndarray) -> np.ndarray:  # noqa: N803
        """
        Apply row-wise softmax to convert raw outputs into probabilities.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_classes)
            Raw regressor outputs.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_classes)
            Probability predictions for each class, rows sum to 1.
        """
        result = softmax(X, axis=1)
        assert isinstance(result, np.ndarray)
        return result

    @staticmethod
    def _sigmoid_normalize(X: np.ndarray) -> np.ndarray:  # noqa: N803
        """
        Apply per-class sigmoid and row-normalize probabilities.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_classes)
            Raw regressor outputs.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_classes)
            Probability predictions for each class, rows sum to 1.
        """
        X = expit(X)  # noqa: N806
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        result = X / row_sums
        assert isinstance(result, np.ndarray)
        return result

    @simple_type_validator
    def fit(
        self,
        X: Annotated[Any, arraylike_validator(ndim=2)],  # noqa: N803
        y: Annotated[Any, arraylike_validator(ndim=1)],
    ) -> object:
        """
        Fit the regressor on one-hot encoded targets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input features.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        RegressorToClassifier
            Fitted instance.
        """
        y = np.array(y).reshape(-1, 1)
        self.encoder.fit(y)
        self.classes_ = self.encoder.categories_[0]
        y_onehot = self.encoder.transform(y)
        n_classes = y_onehot.shape[1]

        # Try fitting as multi-output
        try:
            reg = self.regressor
            assert hasattr(reg, "fit")
            reg.fit(X, y_onehot)
            self.regressors_ = [reg]
        except Exception:
            # Fallback: single-output regressor
            self.regressors_ = [clone(self.regressor) for _ in range(n_classes)]
            for i in range(n_classes):
                reg = self.regressors_[i]
                assert hasattr(reg, "fit")
                reg.fit(X, y_onehot[:, i])

        return self

    @simple_type_validator
    def predict_proba(self, X: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:  # noqa: N803
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        if self.classes_ is None or self.regressors_ is None:
            raise ValueError("This RegressorToClassifier instance is not fitted yet.")

        if len(self.regressors_) == 1:
            reg = self.regressors_[0]
            assert hasattr(reg, "predict")
            raw_pred = reg.predict(X)
        else:
            raw_pred_list = []
            for r in self.regressors_:
                assert hasattr(r, "predict")
                raw_pred_list.append(r.predict(X))
            raw_pred = np.column_stack(raw_pred_list)

        result = self.proba_func(raw_pred)
        assert isinstance(result, np.ndarray)
        return result

    @simple_type_validator
    def predict(self, X: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:  # noqa: N803
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        labels : numpy.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        assert self.classes_ is not None
        result = self.classes_[class_indices].astype(str)
        assert isinstance(result, np.ndarray)
        return result
