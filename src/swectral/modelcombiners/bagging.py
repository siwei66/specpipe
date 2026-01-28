# -*- coding: utf-8 -*-
"""
Swectral - model combiners - Bagging tool

Copyright (c) 2025 Siwei Luo. MIT License.
"""
# Typing
from typing import Optional, Union, Annotated, Any

# Basic computation
import numpy as np

# Modeling
from sklearn.utils import resample
from sklearn.base import clone
from scipy.stats import mode

# Local
from ..specio import simple_type_validator, arraylike_validator


# %% Bagging ensembler


class BaggingEnsembler:
    """
    Bagging ensemble or basic voting for regression and classification models.

    This class creates a bagging (bootstrap aggregating) model ensemble from any scikit-learn compatible base estimator.
    The ensemble supports both regression and classification.
    If the base estimator exposes a ``predict_proba`` method, the ensemble is treated as a classifier and probability averaging is used for prediction.

    Attributes
    ----------
    base_estimator : object
        Any estimator implementing ``fit`` and ``predict`` following the scikit-learn API.
        If the estimator implements ``predict_proba``, the ensemble will operate in classification mode.

    n_estimators : int, optional
        Number of base estimators to train in the ensemble.
        Default is 20.

    max_samples : float, optional
        Fraction of the training samples to draw for each base estimator.
        Must be in the interval ``(0, 1]``.
        Default is 1.

    replace : bool, optional
        Whether sampling is performed with replacement.
        If ``False``, sampling is performed without replacement.
        Default is True.

    random_state : int or None, optional
        Seed used by the random number generator for reproducible bootstrap sampling.
        Default is None.

    regressor_aggregate: str, optional
        Aggregate type for regressors. Choose between:

            - ``"mean"``: Use the average of base estimator predictions.
            - ``"median"``: Use the median of base estimator predictions.
            - tuple of two float: Use a trimmed mean, keeping only predictions within the given quantile range (e.g., (0.1, 0.9)).

        Default is "mean".

    limit_proba: None or tuple of two float, optional
        Limit probability in ensemble. Any probability from base models will be restricted to this range.
        If None, no limit of probability is applied.
        Default is ``None``.

    estimators_ : list of objects
        The collection of fitted base estimators.

    classes_ : ndarray of shape (n_classes,), optional
        Class labels known to the classifier. Only present if the base
        estimator supports ``predict_proba``.

    Methods
    -------
    fit(X, y)
        Fit the bagging ensemble on the training data.

    predict(X)
        Predict regression targets or class labels for ``X``.

    predict_proba(X)
        Predict class probabilities for ``X``. Only available if the base estimator supports ``predict_proba``.

    Examples
    --------
    Bagged regressor::

        from sklearn.cross_decomposition import PLSRegression

        model = BaggingEnsembler(
            base_estimator=PLSRegression(n_components=5),
            n_estimators=100
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    Bagged classifier::

        from sklearn.linear_model import LogisticRegression

        model = BaggingEnsembler(
            base_estimator=LogisticRegression(max_iter=1000),
            n_estimators=50
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

    Specify fraction of training sample and random state used for base estimators::

        model = BaggingEnsembler(
            base_estimator=PLSRegression(n_components=5),
            n_estimators=100,
            max_samples=0.8,
            random_state=42
        )

    Use without replacement::

        model = BaggingEnsembler(
            base_estimator=PLSRegression(n_components=5),
            n_estimators=100,
            replace=False
        )
    """  # noqa: E501

    @simple_type_validator
    def __init__(
        self,
        base_estimator: object,
        n_estimators: int = 50,
        max_samples: float = 1.0,
        replace: bool = True,
        random_state: Optional[int] = None,
        regressor_aggregate: Union[str, tuple[float, float]] = "mean",
        limit_proba: Optional[tuple[float, float]] = None,
    ) -> None:
        # Validate base_estimator
        valid: bool = True
        if not (hasattr(base_estimator, "fit") and hasattr(base_estimator, "predict")):
            valid = False
        elif not (callable(base_estimator.fit) and callable(base_estimator.predict)):
            valid = False
        if not valid:
            raise TypeError(
                "base_estimator must be a scikit-learn style estimator with callable 'fit' and 'predict' methods."
            )
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.replace = replace
        self.random_state = random_state
        # Validate regressor_aggregate
        if isinstance(regressor_aggregate, str):
            if regressor_aggregate.lower() not in ["mean", "median"]:
                raise ValueError(
                    "'regressor_aggregate' must be 'mean' or 'median' or tuple of float for trimmed mean,"
                    f"got: {regressor_aggregate}"
                )
        else:
            if (
                regressor_aggregate[0] >= regressor_aggregate[1]
                or regressor_aggregate[0] < 0
                or regressor_aggregate[1] > 1
            ):  # noqa: E501
                raise ValueError(f"Invalid quantile range: {regressor_aggregate}")
        self.regressor_aggregate = regressor_aggregate
        # Validate limit_proba
        if limit_proba is not None:
            if limit_proba[0] >= limit_proba[1] or limit_proba[0] < 0 or limit_proba[1] > 1:
                raise ValueError(f"Invalid probability range: {limit_proba}")
        self.limit_proba = limit_proba
        # Validate whether classifier
        self._is_classifier = self._detect_classifier(self.base_estimator)

    def _detect_classifier(self, estimator: object) -> bool:
        """Determine if the given estimator is a classifier."""
        if getattr(estimator, "_estimator_type", None) == "classifier":
            return True
        elif hasattr(estimator, "predict_proba"):
            return True
        elif hasattr(estimator, "classes_"):
            return True
        else:
            return False

    @simple_type_validator
    def _fit_single(self, X: np.ndarray, y: np.ndarray, seed: np.integer) -> object:  # noqa: N803
        """Fit single estimator."""
        X_res, y_res = resample(  # noqa: N806
            X, y, replace=self.replace, n_samples=int(self.max_samples * X.shape[0]), random_state=seed
        )
        est = clone(self.base_estimator)
        est.fit(X_res, y_res)
        return est

    @simple_type_validator
    def fit(
        self, X: Annotated[Any, arraylike_validator()], y: Annotated[Any, arraylike_validator()]  # noqa: N803
    ) -> object:
        """
        Fit the bagging ensemble on the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : BaggingEnsembler
            Fitted ensemble.
        """
        # Validate X and y
        X = np.asarray(X)  # noqa: N806
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples in X ({X.shape[0]}) and y ({y.shape[0]}) do not match.")

        rand_state = np.random.RandomState(self.random_state)
        seeds = rand_state.randint(0, 100000000, size=self.n_estimators)

        self.estimators_ = [self._fit_single(X, y, seed) for seed in seeds]

        if self._is_classifier:
            self.classes_ = self.estimators_[0].classes_

        return self

    @simple_type_validator
    def predict(self, X: Annotated[Any, arraylike_validator()]) -> np.ndarray:  # noqa: N803
        """
        Predict regression targets or class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray
            Predicted values or class labels.
        """
        # Validate model is fitted
        if not hasattr(self, "estimators_"):
            raise AttributeError("This ensemble model is not fitted yet. Please fit the model first.")
        # Validate X
        X = np.asarray(X)  # noqa: N806
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")
        preds = np.array([est.predict(X) for est in self.estimators_])

        if self._is_classifier:
            proba = self.predict_proba(X)
            return np.asarray(self.classes_[np.argmax(proba, axis=1)])

        # For classifier
        if self._is_classifier:
            # Soft voting
            if hasattr(self.estimators_[0], "predict_proba"):
                proba = self.predict_proba(X)
                return np.asarray(self.classes_[np.argmax(proba, axis=1)])
            else:
                # Hard voting
                return mode(preds, axis=0).mode.ravel()

        # For regressor
        if isinstance(self.regressor_aggregate, str):
            if self.regressor_aggregate.lower() == "median":
                return np.asarray(np.nanmedian(preds, axis=0))
            else:
                return np.asarray(np.nanmean(preds, axis=0))
        else:
            lower, upper = self.regressor_aggregate
            q_low = np.quantile(preds, lower, axis=0)
            q_high = np.quantile(preds, upper, axis=0)
            # Initialized trimmed preds with nan
            trimmed_preds = np.full_like(preds, np.nan, dtype=float)
            # Mask predictions within the quantile bounds
            mask = (preds >= q_low) & (preds <= q_high)
            trimmed_preds[mask] = preds[mask]
            # For samples where all preds are outside, pick closest to bounds
            for i in range(preds.shape[1]):  # iterate over samples
                if np.all(np.isnan(trimmed_preds[:, i])):
                    # Compute distances to bounds
                    diffs_low = np.abs(preds[:, i] - q_low[i])
                    diffs_high = np.abs(preds[:, i] - q_high[i])
                    # Choose prediction closest to either bound
                    closest_idx = np.argmin(np.minimum(diffs_low, diffs_high))
                    trimmed_preds[closest_idx, i] = preds[closest_idx, i]
            return np.asarray(np.nanmean(trimmed_preds, axis=0))

    @simple_type_validator
    def predict_proba(self, X: Annotated[Any, arraylike_validator()]) -> np.ndarray:  # noqa: N803
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Averaged class probabilities.

        Raises
        ------
        AttributeError
            If the base estimator does not support ``predict_proba``.
        """
        # Validate model is fitted
        if not hasattr(self, "estimators_"):
            raise AttributeError("This ensemble model is not fitted yet. Please fit the model first.")
        # Validate X
        X = np.asarray(X)  # noqa: N806
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")

        if not hasattr(self.base_estimator, "predict_proba"):
            raise AttributeError("Given base estimator does not support 'predict_proba'.")

        probas = np.array([est.predict_proba(X) for est in self.estimators_])
        avg_proba = np.nanmean(probas, axis=0)

        if self.limit_proba is not None:
            min_val, max_val = self.limit_proba
            avg_proba = np.clip(avg_proba, min_val, max_val)

        return np.asarray(avg_proba)
