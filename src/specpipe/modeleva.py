# -*- coding: utf-8 -*-
"""
Model evaluation module for SpecPipe - model wrapers and comprehensive model evaluation tools

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Basics
import copy
import math
import os

# For local test - delete after use
import warnings
from datetime import datetime
from pathlib import Path

# Typing
from typing import Annotated, Any, Literal, Optional, Union, overload

# Model to file
import dill

# Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import mahalanobis
from scipy.stats import entropy

# Model evaluation
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)

# Data
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, train_test_split

# Modeling
from sklearn.preprocessing import OneHotEncoder

# Self
from .roistats import round_digit
from .specio import arraylike_validator, simple_type_validator

# %% Validate data types - for regression and classification distinguishing


# Validate numeric types
def is_numeric(value: Any) -> bool:  # type: ignore[no-untyped-def]
    isnum = False
    try:
        value + 1
        btypes = [bool, np.bool_, pd.BooleanDtype]
        if type(value) not in btypes:
            isnum = True
    except Exception:
        return False
    return isnum


# Validate float types
def is_float(value: Any) -> bool:  # type: ignore[no-untyped-def]
    ftypes = [
        float,  # native
        np.float16,
        np.float32,
        np.float64,  # np and pd
        pd.Float32Dtype,
        pd.Float64Dtype,  # pd nullable
        torch.float16,
        torch.float32,
        torch.float64,  # torch
    ]
    if type(value) in ftypes:
        return True
    else:
        return False


# %% ModelEva module

# Modeling 1D input data without shape hint

# Model module accepts: 0 - list of samples with arbitrary data level, 1 - ScikitLearn-style model class
# Sample data item: (0 - Sample id, 1 - Original shape, 2 - Target value, 3 - Sample predictor value)
# Original shape is used for reading and reshape the X values in the provided model, not necessary,
# commonly not required for spectral model, and the models all use 1D data.
# Data shape can be provided with 1D data to enable modeling higher dimensional data within this frame.
# However, this shaped data is not supported in the current version.


class ModelEva:
    """
    Perform comprehensive model evaluation on SpecPipe sample data.
    The evaluation reports & plots are generated as files in the reporting directory.

    Evaluation includes:
    - model validation results
    - performance metrics
    - case analysis of residuals & Cook's distance
    - scatter plot of true and predicted target values
    - residual analysis report
    - residual plot


    Attributes:
    -----------
    sample_list : list[tuple[str, tuple[int, ...], Union[str,int,bool,float], np.ndarray]]
        SpecPipe sample data as list of tuples containing:
        (sample ID, original shape, target value, predictor array)

    model : scikit-learn-style model object
        Model to evaluate. Must implement:
        - fit()
        - predict()
        - predict_proba() (for classifiers)

    validation_method : str
        Validation method, Choose between:
        - "loo" :
            Leave-one-out cross-validation.
        - "k-fold" (e.g. "5-fold") :
            K-fold cross validation, k is the number of folds.
        - "m-n-split" (e.g. "70-30-split") :
            Train-test split. m% training, n% testing (only test set used for evaluation).

    report_directory : str
        Directory path for saving the evaluation reports and plots.

    model_label : str
        Custom label for the model. Defaults to model class name.

    data_label : str
        Label identifying the given sample_list.

    is_regression : Optional[bool]
        Whether regression model is applied.
        If None, it is automatically determined using the type of target values in the sample list. The default is None.

    unseen_threshold : float = 0.0
        For classification models trained on data with missing classes, a sample is assigned to the unknown class if its highest predicted probability among the known classes is below the unseen_threshold.

    result_backup : bool
        Whether copies of result files are saved, if True, copy files with modeling time are saved in addition.

    silent_all : bool
        Whether to silent all print and plotting. The default is False.


    Methods:
    --------
    update_samples (sample_list)
        Update current sample data using sample_list.

    classifier_evaluation (data_split_config, validation_config, metrics_config, roc_plot_config, residual_config, influence_analysis_config)
        Evaluate classification model performance.

    regressor_evaluation (data_split_config, validation_config, metrics_config, scatter_plot_config, residual_config, residual_plot, influence_analysis_config)
        Evaluate regression model performance.

    train_application_model(return_result, dump_result)
        Train model on entire dataset for application use.
    """  # noqa: E501

    @simple_type_validator
    def __init__(
        self,
        sample_list: list[
            tuple[
                str,
                tuple[int, ...],
                Union[str, int, bool, float],
                Annotated[Any, arraylike_validator(ndim=1)],
            ]
        ],
        model: object,
        validation_method: str,
        report_directory: str,
        model_label: str = "",
        data_label: str = "",
        is_regression: Optional[bool] = None,
        unseen_threshold: float = 0.0,
        result_backup: bool = False,
        silent_all: bool = False,
    ) -> None:
        # Report dir
        report_directory = (report_directory.replace("\\", "/") + "/").replace("//", "/")
        report_directory_path = Path(report_directory)
        if report_directory_path.is_dir() is False:
            raise ValueError(f"\nInvalid report_directory path: \n'{report_directory}'")
        self._report_directory: str = report_directory

        # Model name and dataset name
        if model_label == "":
            model_label = type(model).__name__
        else:
            model_label = str(model_label)
        self._model_label: str = model_label
        self._data_label: str = str(data_label)

        # Model types - regression or classification
        if is_regression is None:
            if is_numeric(sample_list[0][2]):
                is_regression = True
            else:
                is_regression = False
        self.is_regression: bool = is_regression

        # Set sample data features from sample_list (must be first) for model training and evaluation
        # Sample_list item: (0 - Sample id, 1 - Original shape, 2 - Target value, 3 - Sample predictor value)
        # Sample data validated: (0 - Sample ids, 1 - Original shapes, 2 - Target values, 3 - Sample predictor values)
        # (np.array(sid), tuple(xshp), np.array(y).reshape(-1,1), X)
        sample_data_structured = self._val_sample_list(sample_list)
        del sample_list
        self._sid: Annotated[Any, arraylike_validator(ndim=1)] = sample_data_structured[0]  # Sample ID
        self._X_original_shape: tuple[int, ...] = sample_data_structured[1]
        self._y: Annotated[Any, arraylike_validator(ndim=2)] = sample_data_structured[2]
        self._X: Annotated[Any, arraylike_validator(ndim=2)] = sample_data_structured[3]
        if sample_data_structured[2].dtype.kind in ("U", "S", "i", "b"):
            ynames = list(np.unique(self._y))
        elif sample_data_structured[2].dtype.kind in ("i", "f"):
            ynames = None
        self._ynames: Optional[list[Union[str, int, bool]]] = ynames

        # Set model
        self._model: object = self._val_model(model)

        # Set validation_method
        self._validation_method: Union[int, tuple[float, float], str] = self._val_validation_method(validation_method)

        # Data split indices
        self._dsp_inds: list[tuple[Any, Any]] = []

        # Validation results for model evaluation
        self._y_true_eva: Optional[Annotated[Any, arraylike_validator(ndim=2)]] = None
        self._y_pred_eva: Optional[Annotated[Any, arraylike_validator(ndim=2)]] = None
        self._y_true_proba_eva: Optional[Annotated[Any, arraylike_validator(ndim=2)]] = None
        self._y_pred_proba_eva: Optional[Annotated[Any, arraylike_validator(ndim=2)]] = None  # For classifier only
        self._sid_eva: list[str] = []

        # Model application mode
        self._app_model: Optional[object] = None
        self._app_model_create_time: Optional[str] = None

        # Unseen threshold
        self._unseen_threshold: float = 0.0

        # Creating time
        self._create_time: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Model validation time
        self._model_time: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Save backups for results and report
        self._result_backup: bool = result_backup

        # Silent all print and plotting
        self._silent_all: bool = silent_all

    ## Read only or internal properties
    @property
    def create_time(self) -> str:
        return self._create_time

    @create_time.setter
    def create_time(self, value: str) -> None:
        raise ValueError("create_time cannot be modified")

    @property
    def model_time(self) -> str:
        return self._model_time

    @model_time.setter
    def model_time(self, value: str) -> None:
        raise ValueError("model_time cannot be modified")

    @property
    def unseen_threshold(self) -> float:
        return self._unseen_threshold

    @unseen_threshold.setter
    def unseen_threshold(self, value: float) -> None:
        raise ValueError("unseen_threshold cannot be modified")

    @property
    def sid(self) -> Annotated[Any, arraylike_validator(ndim=1)]:
        return self._sid

    @sid.setter
    def sid(self, value: Annotated[Any, arraylike_validator(ndim=1)]) -> None:
        raise ValueError("Sample IDs (sid) cannot be modified directly, use method 'update_samples' to update sid")

    @property
    def X_original_shape(self) -> tuple[int, ...]:  # noqa: N802
        return self._X_original_shape

    @X_original_shape.setter
    def X_original_shape(self, value: tuple[int, ...]) -> None:  # noqa: N802
        raise ValueError(
            "X_original_shape cannot be modified directly, use method 'update_samples' to update X_original_shape"
        )

    @property
    def y(self) -> Annotated[Any, arraylike_validator(ndim=2)]:
        return self._y

    @y.setter
    def y(self, value: Annotated[Any, arraylike_validator(ndim=2)]) -> None:
        raise ValueError("Target value vector y cannot be modified directly, use method 'update_samples' to update y")

    @property
    def X(self) -> Annotated[Any, arraylike_validator(ndim=2)]:  # noqa: N802
        return self._X

    @X.setter
    def X(self, value: Annotated[Any, arraylike_validator(ndim=2)]) -> None:  # noqa: N802
        raise ValueError(
            "Predictor value matrix X cannot be modified directly, use method 'update_samples' to update X"
        )

    @property
    def dsp_inds(self) -> list[tuple[Any, Any]]:
        return self._dsp_inds

    @dsp_inds.setter
    def dsp_inds(self, value: list[tuple[Any, Any]]) -> None:
        raise ValueError("Data train-test split indices 'dsp_inds' cannot be modified")

    @property
    def y_true_eva(self) -> Optional[Annotated[Any, arraylike_validator(ndim=2)]]:
        return self._y_true_eva

    @y_true_eva.setter
    def y_true_eva(self, value: Optional[Annotated[Any, arraylike_validator(ndim=2)]]) -> None:
        raise ValueError("y_true_eva cannot be modified")

    @property
    def y_pred_eva(self) -> Optional[Annotated[Any, arraylike_validator(ndim=2)]]:
        return self._y_pred_eva

    @y_pred_eva.setter
    def y_pred_eva(self, value: Optional[Annotated[Any, arraylike_validator(ndim=2)]]) -> None:
        raise ValueError("y_pred_eva cannot be modified")

    @property
    def ynames(self) -> Optional[list[Union[str, int, bool]]]:
        return self._ynames

    @ynames.setter
    def ynames(self, value: Optional[list[Union[str, int, bool]]]) -> None:
        raise ValueError("ynames cannot be modified")

    @property
    def y_true_proba_eva(self) -> Optional[Annotated[Any, arraylike_validator(ndim=2)]]:
        return self._y_true_proba_eva

    @y_true_proba_eva.setter
    def y_true_proba_eva(self, value: Optional[Annotated[Any, arraylike_validator(ndim=2)]]) -> None:
        raise ValueError("y_true_proba_eva cannot be modified")

    @property
    def y_pred_proba_eva(self) -> Optional[Annotated[Any, arraylike_validator(ndim=2)]]:
        return self._y_pred_proba_eva

    @y_pred_proba_eva.setter
    def y_pred_proba_eva(self, value: Optional[Annotated[Any, arraylike_validator(ndim=2)]]) -> None:
        raise ValueError("y_pred_proba_eva cannot be modified")

    @property
    def sid_eva(self) -> list[str]:
        return self._sid_eva

    @sid_eva.setter
    def sid_eva(self, value: list[str]) -> None:
        raise ValueError("sid_eva cannot be modified")

    @property
    def app_model(self) -> object:
        return self._app_model

    @app_model.setter
    def app_model(self, value: object) -> None:
        raise ValueError("app_model cannot be modified")

    @property
    def app_model_create_time(self) -> Optional[str]:
        return self._app_model_create_time

    @app_model_create_time.setter
    def app_model_create_time(self, value: str) -> None:
        raise ValueError("app_model_create_time cannot be modified")

    ## Mutable properties
    @property
    def report_directory(self) -> str:
        return self._report_directory

    @report_directory.setter
    def report_directory(self, value: str) -> None:
        value = (value.replace("\\", "/") + "/").replace("//", "/")
        value_path = Path(value)
        if value_path.is_dir() is False:
            raise ValueError(f"\nreport_directory is invalid: \n'{value}'")
        self._report_directory = value

    @property
    def data_label(self) -> str:
        return self._data_label

    @data_label.setter
    def data_label(self, value: str) -> None:
        raise ValueError("data_label cannot be modified")

    @property
    def model_label(self) -> str:
        return self._model_label

    @model_label.setter
    def model_label(self, value: str) -> None:
        raise ValueError("model_label cannot be modified")

    @property
    def model(self) -> object:
        return self._model

    @model.setter
    def model(self, model: object) -> None:
        self._model = self._val_model(model)

    @property
    def validation_method(self) -> Union[int, tuple[float, float], str]:
        return self._validation_method

    @validation_method.setter
    def validation_method(self, validation_method: Union[int, tuple[float, float], str]) -> None:
        self._validation_method = self._val_validation_method(validation_method)

    @property
    def result_backup(self) -> bool:
        return self._result_backup

    @result_backup.setter
    def result_backup(self, value: bool) -> None:
        if type(value) is bool:
            self._result_backup = value
        else:
            raise TypeError(f"Result_backup must be bool, got : {value}, type: {type(value)}.")

    @property
    def silent_all(self) -> bool:
        return self._silent_all

    @silent_all.setter
    def silent_all(self, value: bool) -> None:
        if type(value) is bool:
            self._silent_all = value
        else:
            raise TypeError(f"silent_all must be bool, got : {value}, type: {type(value)}.")

    # Update sample_list data
    @simple_type_validator
    def update_samples(
        self,
        sample_list: list[
            tuple[
                str,
                tuple[int, ...],
                Union[str, int, bool, float],
                Annotated[Any, arraylike_validator(ndim=1)],
            ]
        ],
    ) -> None:
        # Update sample_list (must be first)
        sample_data_structured = self._val_sample_list(sample_list)
        self._sid = sample_data_structured[0]
        self._X_original_shape = sample_data_structured[1]
        self._y = sample_data_structured[2]
        self._X = sample_data_structured[3]
        # Update ynames
        if sample_data_structured[2].dtype.kind in ("U", "S", "i", "b"):
            ynames = list(np.unique(self._y))
        elif sample_data_structured[2].dtype.kind in ("i", "f"):
            ynames = None
        self._ynames = ynames

    # Validate  and transform sample data from SpecPipe
    # Sample data item: (0 - Sample id, 1 - Original shape, 2 - Target value, 3 - Sample predictor value)
    @simple_type_validator
    def _val_sample_list(  # noqa: C901
        self,
        sample_list: list[
            tuple[
                str,
                tuple[int, ...],
                Union[str, int, bool, float],
                Annotated[Any, arraylike_validator(ndim=1)],
            ]
        ],
    ) -> tuple[
        Annotated[np.ndarray, arraylike_validator(ndim=1)],
        tuple[int, ...],
        Annotated[np.ndarray, arraylike_validator(ndim=2)],
        Annotated[np.ndarray, arraylike_validator(ndim=2)],
    ]:
        """
        Validate and get sample properties and data from SpecPile sample_list.
        """
        # Construct input samples
        y = []
        for i, st in enumerate(sample_list):
            # Get predictors
            if np.array(st[3]).ndim == 1:
                Xi = np.array([st[3]])  # noqa: N806
            else:
                raise ValueError(
                    f"Expected 1D sample data, \
                        but got {np.array(st).ndim}D array at index {i} in the given sample_list."
                )
            if i == 0:
                X = Xi  # noqa: N806
            else:
                X = np.concatenate((X, Xi), axis=0)  # noqa: N806

            # Get targets
            if i == 0:
                y0 = st[2]
            if self.is_regression & (not is_numeric(st[2])):
                raise TypeError(
                    f"Expected numeric target variable dtypes for regression models, \
                        but got type '{type(st[2])}' at index {i}."
                )
            elif not self.is_regression:
                if is_float(st[2]):
                    raise TypeError(
                        f"Target variable dtype cannot be float, but got type '{type(st[2])}' at index {i}."
                    )
                elif type(st[2]) is not type(y0):
                    raise TypeError(
                        f"Inconsistent target value dtypes, expected type: '{type(y0)}', \
                            got type '{type(st[2])}' at index {i}."
                    )
            y.append(st[2])

            # Get original shape
            if i == 0:
                xshp = st[1]
            else:
                if st[1] != xshp:
                    raise TypeError(
                        f"Inconsistent predictor shape, expected shape: '{xshp}', \
                            got shape '{st[1]}' at index {i} in the give sample_list."
                    )

            # Get sample ID
            if i == 0:
                sid = [st[0]]
            else:
                sid.append(st[0])

        # Convert and return result in tuple of np.ndarrays
        return (np.array(sid), tuple(xshp), np.array(y).reshape(-1, 1), X)

    # Validate specified model
    @simple_type_validator
    def _val_model(self, model: object) -> object:
        """
        Validate model
        """
        model_methods = [
            method
            for method in dir(model)
            if hasattr(model, method) and callable(getattr(model, method)) and not method.startswith("_")
        ]

        if ("fit" in model_methods) and ("predict" in model_methods):
            if (not self.is_regression) & ("predict_proba" in model_methods):
                pass

        else:
            raise ValueError(
                f"model must be defined in ScikitLearn style with methods 'fit' and 'predict', \
                    and includes method 'predict_proba' if it is a classifier, \
                    \nbut got model methods: {model_methods}"
            )

        return model

    # Validate model validation method / data train-test split method
    @simple_type_validator
    def _val_validation_method(self, validation_method: str) -> Union[int, tuple[float, float], str]:  # noqa: C901
        """
        Validate validation_method
        Choose between: "loo" / "k-fold" (e.g. "5-fold") / "m-n-split" (e.g. "70-30-split")
        """
        # Validate k-fold
        if "fold" in validation_method:
            fsp = validation_method.split("-")
            if (len(fsp) != 2) or (fsp[-1] != "fold"):
                raise ValueError(
                    f"Invalid k-fold cross validation method, \
                        expected format: 'k-fold' (k is the number of folds), \
                        but got: '{validation_method}'"
                )
            else:
                try:
                    k = int(fsp[0])
                except Exception as e:
                    raise ValueError(f"{e}\nk must be a number in k-fold cross validation, got: {fsp[0]}") from e
                if k < 2:
                    raise ValueError(f"k must be at least 2 for k-fold cross validation, got: {k}")
                elif k > self._X.shape[0]:
                    warnings.warn(
                        f"Specified k = {k} is larger than sample size {self._X.shape[0]}, 'loo' is applied instead.",
                        UserWarning,
                        stacklevel=3,
                    )
                    return "loo"
            return k

        # Validate train-test-split
        elif "split" in validation_method:
            ttsp = validation_method.split("-")
            if (len(ttsp) != 3) or (ttsp[-1] != "split"):
                raise ValueError(
                    f"Invalid train-test split, \
                        expected format: 'm-n-split' ('m' is train size, 'n' is test size), \
                        got: '{validation_method}'"
                )
            else:
                try:
                    m, n = float(ttsp[0]), float(ttsp[1])
                except Exception as e:
                    raise ValueError(
                        f"{e}\nInvalid train-test split values. \
                            Expected numbers for 'm' (train size) and 'n' (test size), but got: {fsp[0]}"
                    ) from e
                if (m <= 0) | (n <= 0):
                    raise ValueError(
                        f"m (train size) and n (test size) must be positive numbers for train-test-split, \
                            got train size: {m}, test size: {n}"
                    )
            return (m / (m + n), n / (m + n))

        # Validate LOOCV
        elif (validation_method.lower() == "loo") or (validation_method.lower() == "loocv"):
            return "loo"

        else:
            raise ValueError(
                "Unsupported validation method, \
                    validation_method must be one of: \
                    'loo' / 'k-fold' (e.g. '5-fold') / 'm-n-split' (e.g. '70-30-split')"
            )

    # Get indices for data train-test split
    @overload
    def _data_split(
        self,
        random_state: Optional[int] = None,
        validation_method: Optional[str] = None,
        update_ids: bool = True,
        return_ids: Literal[False] = False,
    ) -> None: ...

    @overload
    def _data_split(
        self,
        random_state: Optional[int] = None,
        validation_method: Optional[str] = None,
        update_ids: bool = True,
        return_ids: Literal[True] = True,
    ) -> list[tuple[Any, Any]]: ...

    # Get indices for data train-test split
    @simple_type_validator
    def _data_split(  # noqa: C901
        self,
        random_state: Optional[int] = None,
        validation_method: Optional[str] = None,
        update_ids: bool = True,
        return_ids: bool = False,
    ) -> Union[None, list[tuple[Any, Any]]]:
        """
        Split of train and test data for model validation.
        """
        # Set random state
        if random_state is None:
            random_state = np.random.randint(0, np.iinfo(np.int32).max)

        # Validation method
        val_method: Union[int, tuple[float, float], str]
        if validation_method is None:
            val_method = self._validation_method
        else:
            val_method = self._val_validation_method(validation_method)

        # Get sample indices
        indices = np.arange(len(self._X))

        # Train-test split
        dsp_inds: list = []
        if type(val_method) is tuple:
            train_idx, test_idx = train_test_split(
                indices, test_size=val_method[1], random_state=random_state, shuffle=True
            )
            dsp_inds = [(train_idx, test_idx)]

        # k-Fold
        elif type(val_method) is int:
            if self.is_regression:
                kf = KFold(n_splits=val_method, shuffle=True, random_state=random_state)
                for train_idx, val_idx in kf.split(indices):
                    dsp_inds.append((train_idx, val_idx))
            else:
                kf = StratifiedKFold(n_splits=val_method, shuffle=True, random_state=random_state)
                for train_idx, val_idx in kf.split(self.X, self.y):
                    dsp_inds.append((train_idx, val_idx))

        # LOOCV
        elif val_method == "loo":
            loo = LeaveOneOut()
            for train_idx, val_idx in loo.split(indices):
                dsp_inds.append((train_idx, val_idx))

        # Invalid validation_method
        else:
            raise ValueError(f"Unknown validation method: {self._validation_method}")

        # Output result
        if update_ids:
            self._dsp_inds = dsp_inds
        if return_ids:
            return dsp_inds
        else:
            return None

    # Train training folds and predict target fold (Classfier)
    @simple_type_validator
    def _classifier_validation(  # noqa: C901
        self,
        unseen_threshold: Optional[float] = None,
        use_original_shape: bool = False,
        save_fold_model: bool = True,
        save_fold_data: bool = True,
    ) -> None:
        """
        Training and validating classification model performance on the specified data.
        By running this function, sample validation results are updated in the 'ModelEva' object.
        """
        if not self.silent_all:
            print(f"\nModel validation, Dataset: '{self.data_label}', Model: '{self.model_label}'")

        # Model validation time
        self._model_time = datetime.now().strftime("created_at_%Y-%m-%d_%H-%M-%S")

        # Store unseen threshold
        if unseen_threshold is None:
            unseen_threshold = self._unseen_threshold

        # Get and validate data
        sid = self._sid
        X = self._X  # noqa: N806
        y = self._y
        if y.dtype.kind == "f":
            raise TypeError(f"Expected discontinuous y dtype, got: '{y.dtype}'")
        ynames = self._ynames

        # Reshape X sample data if required (e.g. for wrapped torch models)
        if use_original_shape:
            for sdim in self._X_original_shape:
                if sdim <= 0:
                    raise ValueError(
                        f"Dimensions of X_original_shape must be positive, but got shape: {self._X_original_shape}"
                    )
            if len(self._X_original_shape) == 1:
                pass
            elif len(self._X_original_shape) > 1:
                X = X.reshape(X.shape[0], *self._X_original_shape)  # noqa: N806
            else:
                raise ValueError(f"Invalid X_original_shape: {self._X_original_shape}")

        # For OneHotEncoding
        encoder = OneHotEncoder(sparse_output=False)

        # Fold indices
        dsps = self._dsp_inds

        # Training
        itr = 0
        sid_eva = []

        for train_ind, test_ind in dsps:
            if type(self._validation_method) is tuple:
                if not self.silent_all:
                    print("Model training ...")
            else:
                if not self.silent_all:
                    print(f"\rTraining fold: {itr + 1}/{len(dsps)}", end="", flush=True)
            # Model
            model = copy.deepcopy(self._model)  # type: ignore[attr-defined]
            # unrecognized dynamic custom model, independent runtime validated, following the same

            # Sample ids of target values in validation
            sid_test = sid[test_ind]
            sid_train = sid[train_ind]

            # Fold data for training and testing
            X_train, X_test = X[train_ind], X[test_ind]  # noqa: N806
            y_train, y_test = y[train_ind], y[test_ind]

            # Fit
            model.fit(X_train, y_train.flatten())  # type: ignore[attr-defined]

            # Predict
            y_test_pred = model.predict(X_test).reshape(-1, 1)  # type: ignore[attr-defined]
            y_test_pred_proba = model.predict_proba(X_test)  # type: ignore[attr-defined]
            y_test_pred_proba_df = pd.DataFrame(
                y_test_pred_proba, columns=model.classes_, index=sid_test  # type: ignore[attr-defined]
            )

            # Store results
            if itr == 0:
                y_true = y_test
                y_pred = y_test_pred
                y_pred_proba = y_test_pred_proba_df.reindex(columns=ynames)
                # Old: y_pred_proba = pd.concat([pd.DataFrame(columns=ynames), y_test_pred_proba_df], ignore_index=True)
                sid_eva = list(sid_test)
            else:
                y_true = np.concatenate((y_true, y_test), axis=0)
                y_pred = np.concatenate((y_pred, y_test_pred), axis=0)
                y_pred_proba = pd.concat([y_pred_proba, y_test_pred_proba_df], ignore_index=True)
                sid_eva = sid_eva + list(sid_test)

            # Save fold results to files
            if save_fold_model:
                self._dump_val_model(
                    fold_i=itr,
                    X_train=X_train,
                    y_train=y_train,
                    sid_train=sid_train,
                    X_test=X_test,
                    y_test=y_test,
                    sid_test=sid_test,
                    y_true_all=y,
                    fold_model=model,
                    y_test_proba=np.array(y_test_pred_proba_df),
                    dump_associated_data=save_fold_data,
                    attach_proba=True,
                )
            itr = itr + 1

        if not self.silent_all:
            print("")

        # Generate y_true_proba
        y_true_proba = pd.DataFrame(encoder.fit_transform(y_true), columns=encoder.categories_[0])
        y_true_proba = y_true_proba.reindex(columns=ynames)
        # Old: y_true_proba = pd.concat([pd.DataFrame(columns=ynames), y_true_proba], ignore_index=True)

        # Replace pandas nan
        if hasattr(pd.DataFrame, 'map'):
            # pandas >= 2.1.0
            y_pred_proba = y_pred_proba.map(lambda x: np.nan if pd.isna(x) else x)
        else:
            # pandas < 2.1.0
            y_pred_proba = y_pred_proba.applymap(lambda x: np.nan if pd.isna(x) else x)
        # Old: y_pred_proba = y_pred_proba.applymap(lambda x: np.nan if pd.isna(x) else x)

        # Convert back to array
        y_true_proba = np.array(y_true_proba)
        y_pred_proba = np.array(y_pred_proba)

        # Unseen class
        for ri, pp_row in enumerate(y_pred_proba):
            na_num: int = int(np.sum(np.isnan(pp_row)))
            if na_num > 1:
                pp_row[np.isnan(pp_row)] = 0
                y_pred_proba[ri] = pp_row
            elif na_num == 1:
                if np.max(pp_row) <= unseen_threshold:
                    pp_row[np.isnan(pp_row)] = 0.5
                    pp_row = pp_row / np.sum(pp_row)
                else:
                    pp_row[np.isnan(pp_row)] = 0
                y_pred_proba[ri] = pp_row

        # Update model validation results
        self._y_true_eva = y_true
        self._y_pred_eva = y_pred
        self._y_true_proba_eva = y_true_proba
        self._y_pred_proba_eva = y_pred_proba
        self._sid_eva = list(sid_eva)

        # Report directory
        dout = self._report_directory + f"Model_evaluation_reports/Data_{self.data_label}_Model_{self.model_label}/"
        if not os.path.exists(dout):
            os.makedirs(dout)

        # Result dataframe for write to file
        assert ynames is not None
        coln_val = ["Sample_ID", "y_true", "y_predicted"] + [f"Proba_{str(yn)}" for yn in list(ynames)]
        df_val = pd.DataFrame(np.zeros((len(y_true), len(coln_val))), columns=coln_val)
        df_val["Sample_ID"] = sid_eva
        df_val["y_true"] = y_true
        df_val["y_predicted"] = y_pred
        df_val.iloc[:, 3:] = y_pred_proba

        # Write result to file
        task_time = self._model_time
        df_val.to_csv(dout + f"Validation_results_{self.model_label}.csv", index=False)
        if self.result_backup:
            df_val.to_csv(dout + f"Validation_results_{self.model_label}_{task_time}.csv", index=False)

    # Classifier metrics
    @simple_type_validator
    def _classifier_metrics(self) -> None:  # noqa: C901
        """
        Calculate and save classifier performance evaluation metrics based on true and predicted target values.
        """
        if not self.silent_all:
            print("Calculating performance metrics ...")

        # Get data
        y_true = self._y_true_eva
        y_pred = self._y_pred_eva
        y_pred_proba = self._y_pred_proba_eva

        # Validate data
        y_true = arraylike_validator(ndim=2)(y_true)
        y_pred = arraylike_validator(ndim=2)(y_pred)
        y_pred_proba = arraylike_validator(ndim=2)(y_pred_proba)

        # Validate data settings
        if (y_true is None) & (y_pred is None) & (y_pred_proba is None):
            pass
        elif (y_true is not None) & (y_pred is not None) & (y_pred_proba is not None):
            pass
        else:
            warnings.warn(
                "y_true, y_pred and y_pred_proba is partially specified, \
                    the specified values must match the unspecified default values",
                UserWarning,
                stacklevel=3,
            )

        # Validate type of target variables for classifier
        if y_true.dtype.kind == "f":
            raise TypeError(f"Expected discontinuous y dtype, got: '{y_true.dtype}'")
        if y_pred.dtype.kind == "f":
            raise TypeError(f"Expected discontinuous y dtype, got: '{y_pred.dtype}'")
        if y_pred_proba.dtype.kind != "f":
            raise TypeError(f"Expected type of target class probabilities is float, got: '{y_pred_proba.dtype}'")

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        TP = np.diag(cm)  # True Positives for each class  # noqa: N806
        FP = cm.sum(axis=0) - TP  # False Positives  # noqa: N806
        FN = cm.sum(axis=1) - TP  # False Negatives  # noqa: N806
        TN = cm.sum() - (FP + FN + TP)  # True Negatives  # noqa: N806

        # Class metrics summary
        ynames = self._ynames
        metrics_data = []
        if ynames is None:
            raise ValueError("No class label found")
        for i, class_name in enumerate(ynames):
            precision = TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else np.nan
            recall = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else np.nan
            if ((precision + recall) > 0) & (precision is not np.nan) & (recall is not np.nan):
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = np.nan
            # Binarize the output
            y_binary = (y_true == ynames[i]).astype("int")
            y_predi = (np.array(y_pred) == ynames[i]).astype("int")
            fpr, tpr, _ = roc_curve(y_binary, [score[i] for score in y_pred_proba])
            roc_auc = auc(fpr, tpr)
            accuracyi = accuracy_score(y_binary, y_predi)
            metrics_data.append(
                {
                    "Class": class_name,
                    "TP": TP[i],
                    "TN": TN[i],
                    "FP": FP[i],
                    "FN": FN[i],
                    "Precision": precision,
                    "Recall": recall,
                    "F1_Score": f1,
                    "Accuracy": accuracyi,
                    "AUC": roc_auc,
                }
            )

        # Create a DataFrame for metrics
        metrics_df = pd.DataFrame(metrics_data)

        # Calculate overall accuracy, micro- and macro-average metrics
        accuracy_micro = accuracy_score(y_true, y_pred)
        micro_precision = precision_score(y_true, y_pred, average="micro", zero_division=np.nan)
        micro_recall = recall_score(y_true, y_pred, average="micro", zero_division=np.nan)
        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=np.nan)
        micro_metrics = pd.DataFrame(
            {
                "Class": ["Micro_avg"],
                "TP": [TP.sum()],
                "TN": [TN.sum()],
                "FP": [FP.sum()],
                "FN": [FN.sum()],
                "Precision": [micro_precision],
                "Recall": [micro_recall],
                "F1_Score": [micro_f1],
                "Accuracy": [accuracy_micro],
                "AUC": ["-"],
            }
        )
        accuracy_macro = metrics_df["Accuracy"].mean()
        auc_macro = metrics_df["AUC"].mean()
        macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=np.nan)
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=np.nan)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=np.nan)
        macro_metrics = pd.DataFrame(
            {
                "Class": ["Macro_avg"],
                "TP": ["-"],
                "TN": ["-"],
                "FP": ["-"],
                "FN": ["-"],
                "Precision": [macro_precision],
                "Recall": [macro_recall],
                "F1_Score": [macro_f1],
                "Accuracy": [accuracy_macro],
                "AUC": [auc_macro],
            }
        )
        metrics_df = pd.concat([metrics_df, micro_metrics], ignore_index=True)
        metrics_df = pd.concat([metrics_df, macro_metrics], ignore_index=True)
        rptitle = pd.DataFrame(
            {
                "Class": [f"{self.model_label} model classification performance report"],
                "TP": [""],
                "TN": [""],
                "FP": [""],
                "FN": [""],
                "Precision": [""],
                "Recall": [""],
                "F1_Score": [""],
                "Accuracy": [""],
                "AUC": [""],
            }
        )
        metrics_df = pd.concat([metrics_df, rptitle], ignore_index=True)

        # Report directory
        dout = self._report_directory + f"Model_evaluation_reports/Data_{self.data_label}_Model_{self.model_label}/"
        if not os.path.exists(dout):
            os.makedirs(dout)

        # Save metrics df to CSV
        task_time = self._model_time
        metrics_df.to_csv(dout + f"Classification_performance_{self.model_label}.csv", index=False)
        if self.result_backup:
            metrics_df.to_csv(dout + f"Classification_performance_{self.model_label}_{task_time}.csv", index=False)

    # Class ROC plots of classification performance
    @simple_type_validator
    def _classifier_roc_curve(  # noqa: C901
        self,
        plot_title: str = "ROC Curve",
        title_size: Union[int, float] = 26,
        title_pad: Union[int, float, None] = None,
        figure_size: tuple[Union[int, float], Union[int, float]] = (8, 8),
        plot_margin: tuple[float, float, float, float] = (
            0.15,
            0.95,
            0.9,
            0.13,
        ),  # (left,right,top,bottom)
        plot_line_with: Union[int, float] = 3,
        plot_line_alpha: float = 0.8,
        diagnoline_width: Union[int, float] = 3,
        x_axis_limit: Optional[tuple[Union[int, float], Union[int, float]]] = None,
        x_axis_label: str = "False Positive Rate",
        x_axis_label_size: Union[int, float] = 26,
        x_tick_size: Union[int, float] = 24,
        x_tick_number: int = 6,
        y_axis_limit: Optional[tuple[Union[int, float], Union[int, float]]] = None,
        y_axis_label: str = "True Positive Rate",
        y_axis_label_size: Union[int, float] = 26,
        y_tick_size: Union[int, float] = 24,
        y_tick_number: int = 6,
        axis_line_size_left: Union[int, float, None] = 1.5,
        axis_line_size_right: Union[int, float, None] = 1.5,
        axis_line_size_top: Union[int, float, None] = 1.5,
        axis_line_size_bottom: Union[int, float, None] = 1.5,
        legend: bool = True,
        legend_location: str = "lower right",
        legend_fontsize: Union[int, float] = 20,
        legend_title: str = "",
        legend_title_fontsize: Union[int, float] = 24,
        background_grid: bool = False,
        show_plot: bool = False,
    ) -> None:
        """
        Plot classifier Receiver Operating Characteristic (ROC) curves on the validation data.
        """
        plt.ioff()
        if not self.silent_all:
            print("Generating ROC plot ...")
        # Get classes
        ynames = self._ynames
        # Get data
        y_true = self._y_true_eva
        y_pred_proba = self._y_pred_proba_eva
        # Validate value existence
        assert ynames is not None
        assert y_true is not None
        assert y_pred_proba is not None

        # ROC curves
        plt.figure(figsize=figure_size)  # For ROC curve
        for i in range(len(ynames)):
            # Binarize the output
            y_binary = np.array((y_true == ynames[i])).astype("int")
            fpr, tpr, _ = roc_curve(y_binary, [score[i] for score in y_pred_proba])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                label=f"{ynames[i]} (AUC = {roc_auc:.2f})",
                linewidth=plot_line_with,
                alpha=plot_line_alpha,
            )

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], "k--", linewidth=diagnoline_width)  # Diagonal line

        # Plot title
        if len(plot_title) > 0:
            if title_pad is None:
                title_pad = int(title_size / 2) + 1
            plt.title(plot_title, fontsize=title_size, pad=title_pad)

        # Plot axis
        if x_axis_limit is not None:
            if x_axis_limit[0] < x_axis_limit[1]:
                plt.xlim(x_axis_limit[0], x_axis_limit[1])
            else:
                raise ValueError(f"Invalid x_axis_limit range: {x_axis_limit}")
        if y_axis_limit is not None:
            if y_axis_limit[0] < y_axis_limit[1]:
                plt.ylim(y_axis_limit[0], y_axis_limit[1])
            else:
                raise ValueError(f"Invalid y_axis_limit range: {y_axis_limit}")
        if len(x_axis_label) > 0:
            plt.xlabel(x_axis_label, fontsize=x_axis_label_size)
        if len(y_axis_label) > 0:
            plt.ylabel(y_axis_label, fontsize=y_axis_label_size)
        plt.xticks(fontsize=x_tick_size)  # Set x-axis tick font size

        plt.yticks(fontsize=y_tick_size)  # Set y-axis tick font size

        # Axis line sizes
        ax = plt.gca()  # Get the current axis
        if axis_line_size_top is not None:
            if axis_line_size_top > 0:
                ax.spines["top"].set_linewidth(axis_line_size_top)  # Top axis line
        else:
            ax.spines["top"].set_visible(False)  # Top axis line
        if axis_line_size_bottom is not None:
            if axis_line_size_bottom > 0:
                ax.spines["bottom"].set_linewidth(axis_line_size_bottom)  # Bottom axis line
        else:
            ax.spines["bottom"].set_visible(False)  # Bottom axis line
        if axis_line_size_left is not None:
            if axis_line_size_left > 0:
                ax.spines["left"].set_linewidth(axis_line_size_left)  # Left axis line
        else:
            ax.spines["left"].set_visible(False)  # Left axis line
        if axis_line_size_right is not None:
            if axis_line_size_right > 0:
                ax.spines["right"].set_linewidth(axis_line_size_right)  # Right axis line
        else:
            ax.spines["right"].set_visible(False)  # Right axis line

        # Plot legend
        if legend:
            if len(legend_title) > 0:
                if legend_title_fontsize <= 0:
                    raise ValueError(f"legend_fontsize must be positive, got: {legend_fontsize}")
                plt.legend(
                    loc=legend_location,
                    fontsize=legend_fontsize,
                    title=legend_title,
                    title_fontsize=legend_title_fontsize,
                )
            else:
                plt.legend(loc=legend_location, fontsize=legend_fontsize)

        # Plot background and margin
        if not background_grid:
            plt.grid(False)  # Remove the grid lines
        plt.subplots_adjust(
            left=plot_margin[0], right=plot_margin[1], top=plot_margin[2], bottom=plot_margin[3]
        )  # Adjust margins

        # Report directory
        dout = self._report_directory + f"Model_evaluation_reports/Data_{self.data_label}_Model_{self.model_label}/"
        if not os.path.exists(dout):
            os.makedirs(dout)

        # Save plot to PNG
        task_time = self._model_time
        plt.savefig(dout + f"ROC_curve_{self.model_label}.png", dpi=300)
        if self.result_backup:
            plt.savefig(dout + f"ROC_curve_{self.model_label}_{task_time}.png", dpi=300)
        if show_plot:
            if not self.silent_all:
                plt.show()
        plt.close()
        plt.ion()

    # Case analysis - residual analysis
    @simple_type_validator
    def _classifier_residual(self) -> None:
        """
        Case analysis report for classification models on the validation data.
        For residual and residual outlier analysis.
        """
        if not self.silent_all:
            print("Calculating probability residual report ...")
        # Classes
        ynames = self._ynames

        # Sample IDs
        sid = self._sid_eva

        # Probability of target variables
        y_true_proba = self._y_true_proba_eva
        y_pred_proba = self._y_pred_proba_eva

        # Validate result existence
        try:
            assert ynames is not None
            assert len(sid) > 0
            assert y_true_proba is not None
            assert y_pred_proba is not None
        except Exception as e:
            raise ValueError(f"Incomplete validation data: {e}") from e

        # Probability residuals
        res = y_true_proba - y_pred_proba
        df_res = pd.DataFrame(
            res,
            columns=[f"Probability_residual_{yn}" for yn in ynames],
            index=list(sid),
        )
        df_res.index.name = "Sample_ID"
        df_res_data = df_res.copy(deep=True)

        # Residual z-score
        for yn in ynames:
            df_res["Z-score_residual_" + str(yn)] = (
                df_res_data[f"Probability_residual_{yn}"] - np.nanmean(df_res_data[f"Probability_residual_{yn}"])
            ) / np.nanstd(df_res_data[f"Probability_residual_{yn}"])

        # Sum of absolute residual
        df_res["Sum_of_absolute_residual"] = df_res_data.abs().sum(axis=1)

        # Root sum of squared residual
        df_res["Root_sum_of_squared_residual"] = np.sqrt((df_res_data**2).sum(axis=1))

        # Normalized entropy
        df_abs_res = df_res_data.abs()
        H = df_abs_res.apply(lambda row: entropy(row, base=2), axis=1)  # noqa: N806
        max_entropy = np.log2(len(ynames))
        df_res["Shannon_entropy_of_absolute_residual"] = H / max_entropy

        # Mahalanobis Distance
        mean_res = np.mean(df_res_data, axis=0)  # Mean per class
        mean_res = np.array(mean_res)
        cov_mat_res = np.cov(df_res_data, rowvar=False)  # Covariance between classes
        cov_inv_res = np.linalg.pinv(cov_mat_res)  # Pseudo-inverse for stability
        # Compute
        mahalanobis_dist = []
        for r in np.array(df_res_data):
            dist = mahalanobis(r, mean_res, cov_inv_res)
            mahalanobis_dist.append(dist)
        df_res["Mahalanobis_distance_of_residual"] = mahalanobis_dist

        # Add true and predicted proba
        df_y_true_proba = pd.DataFrame(y_true_proba, columns=[(str(yn) + "_true") for yn in ynames], index=sid)
        df_y_pred_proba = pd.DataFrame(y_pred_proba, columns=[(str(yn) + "_predicted") for yn in ynames], index=sid)
        df_res["True_label"] = self.y_true_eva
        df_res = pd.concat([df_res, df_y_true_proba], axis=1)
        df_res["Predicted_label"] = self.y_pred_eva
        df_res = pd.concat([df_res, df_y_pred_proba], axis=1)

        # Report directory
        dout = self._report_directory + f"Model_evaluation_reports/Data_{self.data_label}_Model_{self.model_label}/"
        if not os.path.exists(dout):
            os.makedirs(dout)

        # Save case report df to CSV
        task_time = self._model_time
        df_res.to_csv(dout + f"Residual_analysis_{self.model_label}.csv", index=True, index_label="Sample_ID")
        if self.result_backup:
            df_res.to_csv(
                dout + f"Residual_analysis_{self.model_label}_{task_time}.csv",
                index=True,
                index_label="Sample_ID",
            )

    # Data split indices for influence analysis
    @simple_type_validator
    def _influence_data_split_ids(
        self, validation_method: str = 'default', random_state: Optional[int] = None
    ) -> list[tuple[Any, Any]]:
        """
        Validation_method "default" uses "2-fold" or train-test-split if model validation method is train-test-split.
        Validation_method "model" uses model validation method.
        """
        # Validate random state
        if random_state is None:
            random_state = np.random.randint(0, np.iinfo(np.int32).max)

        # Fold indices
        dsps: list[tuple[Any, Any]]
        # Use model validation
        if validation_method.lower() == 'model':
            dsps = self._dsp_inds
        # Model LOOCV
        elif validation_method.lower() == 'loo':
            dsps = self._data_split(
                random_state=random_state, validation_method='loo', update_ids=False, return_ids=True
            )
        # Fast
        elif validation_method.lower() == 'default':
            if type(self.validation_method) is tuple:
                dsps = self._dsp_inds
            else:
                dsps = self._data_split(
                    random_state=random_state, validation_method='2-fold', update_ids=False, return_ids=True
                )
        # Custom
        else:
            dsps = self._data_split(
                random_state=random_state, validation_method=validation_method, update_ids=False, return_ids=True
            )

        # Return result
        return dsps

    # LOO training for Cook's dist-like influence analysis
    @simple_type_validator
    def _classifier_influential_analysis(  # noqa: C901
        self, validation_method: str = 'default', random_state: Optional[int] = None
    ) -> None:
        """
        Calculate the cooks-distance-like average influence on predictions of each sample using LOO approach.

        Validation_method "default" uses "2-fold" or train-test-split if model validation method is train-test-split.
        Set validation_method "model" to use model validation method.
        """
        # Model validation time
        self._model_time = datetime.now().strftime("created_at_%Y-%m-%d_%H-%M-%S")

        # Unseen threshold
        unseen_threshold = self._unseen_threshold

        # Data
        sid = self._sid
        X = self._X  # noqa: N806
        y = self._y

        # Classes
        ynames = self._ynames
        assert ynames is not None

        # For OneHotEncoding
        encoder = OneHotEncoder(sparse_output=False)
        y_p = encoder.fit_transform(y)  # type: ignore[attr-defined]
        # Custom model, independent runtime validated, following the same

        # Fold indices
        dsps = self._influence_data_split_ids(validation_method=validation_method, random_state=random_state)

        # Validation Training
        influence_list = []
        itr1 = 0
        for train_ind, test_ind in dsps:
            if not self.silent_all:
                print(f"\rTraining Cook's distance fold: {itr1 + 1}/{len(dsps)}", end="", flush=True)
            ## Get data
            # Sample ids of target values in validation
            sid_test = sid[test_ind]
            # Fold data for training and testing
            X_train, X_test = X[train_ind], X[test_ind]  # noqa: N806
            y_train = y[train_ind]
            y_p_test = y_p[test_ind]
            y_p_test = pd.DataFrame(y_p_test, columns=ynames)

            # Calculate MSE
            model_full = copy.deepcopy(self._model)
            model_full.fit(X_train, y_train.flatten())  # type: ignore[attr-defined]
            # Custom model, independent runtime validated, following the same
            p_full = model_full.predict_proba(X_test)  # type: ignore[attr-defined]
            p_full = pd.DataFrame(p_full, columns=model_full.classes_, index=sid_test)  # type: ignore[attr-defined]
            p_full = p_full.reindex(columns=ynames)
            # Old: p_full = pd.concat([pd.DataFrame(columns=ynames), p_full], ignore_index=True)

            # Replace pandas nan
            if hasattr(pd.DataFrame, 'map'):
                # pandas >= 2.1.0
                p_full = p_full.map(lambda x: np.nan if pd.isna(x) else x)
            else:
                # pandas < 2.1.0
                p_full = p_full.applymap(lambda x: np.nan if pd.isna(x) else x)
            # Old: p_full = p_full.applymap(lambda x: np.nan if pd.isna(x) else x)

            # Convert back to array
            y_p_test = np.array(y_p_test)
            p_full = np.array(p_full)

            # Unseen class
            for ri, pp_row in enumerate(p_full):
                na_num: int = int(np.sum(np.isnan(pp_row)))
                if na_num > 1:
                    pp_row[np.isnan(pp_row)] = 0
                    p_full[ri] = pp_row
                elif na_num == 1:
                    if np.max(pp_row) <= unseen_threshold:
                        pp_row[np.isnan(pp_row)] = 0.5
                        pp_row = pp_row / np.sum(pp_row)
                    else:
                        pp_row[np.isnan(pp_row)] = 0
                    p_full[ri] = pp_row

            mse = np.mean((y_p_test - p_full) ** 2, axis=0)

            # Cook's dist LOO training
            influence = np.zeros((X.shape[0], len(ynames)))
            for i in range(X_train.shape[0]):
                # LOO-Model
                model_loo = copy.deepcopy(self._model)
                # LOO-data
                X_loo = np.delete(X_train, i, axis=0)  # Leave-one-out dataset  # noqa: N806
                y_loo = np.delete(y_train, i, axis=0)
                # LOO-training-n-prediction
                model_loo.fit(X_loo, y_loo.flatten())  # type: ignore[attr-defined]
                # Custom model, independent runtime validated, following the same
                # Predict on test X
                p_loo = model_loo.predict_proba(X_test)  # type: ignore[attr-defined]
                p_loo = pd.DataFrame(p_loo, columns=model_loo.classes_)  # type: ignore[attr-defined]
                p_loo = p_loo.reindex(columns=ynames)
                # Old: p_loo = pd.concat([pd.DataFrame(columns=ynames), p_loo], ignore_index=True)

                # Replace pandas nan
                if hasattr(pd.DataFrame, 'map'):
                    # pandas >= 2.1.0
                    p_loo = p_loo.map(lambda x: np.nan if pd.isna(x) else x)
                else:
                    # pandas < 2.1.0
                    p_loo = p_loo.applymap(lambda x: np.nan if pd.isna(x) else x)
                # Old: p_loo = p_loo.applymap(lambda x: np.nan if pd.isna(x) else x)

                # Convert back to array
                p_loo = np.array(p_loo)

                # Unseen class
                for ri, pp_row in enumerate(p_loo):
                    na_num = np.sum(np.isnan(pp_row))
                    if na_num > 1:
                        pp_row[np.isnan(pp_row)] = 0
                        p_loo[ri] = pp_row
                    elif na_num == 1:
                        if np.max(pp_row) <= unseen_threshold:
                            pp_row[np.isnan(pp_row)] = 0.5
                            pp_row = pp_row / np.sum(pp_row)
                        else:
                            pp_row[np.isnan(pp_row)] = 0
                        p_loo[ri] = pp_row

                # Calculate Cook's dist
                if X_train.shape[1] < 1:
                    raise ValueError(f"Invalid X_train, got: {X_train}, type: {type(X_train)}, shape: {X_train.shape}")
                mse1 = mse + 1e-30
                influence[train_ind[i]] = np.sum((p_full - p_loo) ** 2, axis=0) / (X_train.shape[1] * mse1)

            # Store results
            influence_list.append(influence)
            itr1 = itr1 + 1
        if not self.silent_all:
            print("")

        # Avg influence
        influence_avg = np.sum(influence_list, axis=0) / (X_train.shape[0] - 1)
        influence_df = pd.DataFrame(influence_avg, columns=ynames, index=sid)
        influence_df.columns = [f"{yn}_probability_Cooks_distance_like" for yn in ynames]
        influence_df.index.name = "Sample_ID"

        # Report directory
        dout = self._report_directory + f"Model_evaluation_reports/Data_{self.data_label}_Model_{self.model_label}/"
        if not os.path.exists(dout):
            os.makedirs(dout)

        # Save metrics df to CSV
        task_time = self._model_time
        influence_df.to_csv(dout + f"Influence_analysis_{self.model_label}.csv", index=True, index_label="Sample_ID")
        if self.result_backup:
            influence_df.to_csv(
                dout + f"Influence_analysis_{self.model_label}_{task_time}.csv",
                index=True,
                index_label="Sample_ID",
            )

    # Evaluation of classifier performance
    @simple_type_validator
    def classifier_evaluation(  # noqa: C901
        self,
        data_split_config: Union[str, dict] = "default",
        validation_config: Union[str, dict] = "default",
        metrics_config: Union[str, dict, None] = "default",
        roc_plot_config: Union[str, dict, None] = "default",
        residual_config: Union[str, dict, None] = "default",
        influence_analysis_config: Union[str, dict, None] = "default",
    ) -> None:
        """
        Perform classifier performance evaluation using test data set or cross validation on the specified data of 'ModelEva'.

        The configuration parameters must be provided in a dictionary of subparameters with the format of {parameter name (key) : parameter values (value)}.
        This method outputs the evaluation reports to files in the reporting directory.

        The evaluation report includes:
        - model validation results
        - performance metrics
        - residual analysis report
        - case analysis of residuals & Cook's distance
        - Response Operating Characteristics plot

        Parameters
        ----------
        data_split_config : Union[str,dict], optional
            Configuration of data split options in dictionary. The default is 'default', using default settings.

            The parameters of data split include:
            - random_state : Optional[int], optional
                Random state for data splitting and shuffling. If None, random_state is not fixed. The default is None.
            - validation_method : Optional[str], optional
                Model validation method, default is using 'ModelEva.validation_method', see 'ModelEva' for details.

        validation_config : Union[str,dict], optional
            Configuration of validation options. The default is 'default', using default settings.

            The parameters of validation include:
            - unseen_threshold : Optional[float]
                For classification models trained on data with missing classes, a sample is assigned to a unknown class if its highest predicted probability among the known classes is below the unseen_threshold.
            - use_original_shape : bool
                Whether the sample data is reshaped to its original shape.
                If False, the flattened data is used. The default is False.
            - save_fold_model: bool
                Whether to save model of each fold. The default is True.
            - save_fold_data: bool
                Whether to save training and validation data of each fold. The default is True.
                Saving fold data and fold models could consuming significant storage when applied to large data with large sample size with large fold numbers.

        metrics_config : Union[str,dict,None], optional
            Configuration of metrics options. The default is 'default', using default settings.
            If None, the performance metrics computation is skipped.

        roc_plot_config : Union[str,dict,None], optional
            Configuration of plotting Response Operating Characteristics (ROC) curves. The default is 'default', using default settings.
            If None, the ROC curve plotting is skipped.

            The parameters of scatter plot and its default values are listed as follows:
            - plot_title : str = 'ROC Curve',
            - title_size : Union[int,float] = 26,
            - title_pad : Union[int,float,None] = None,
            - figure_size : tuple[Union[int,float],Union[int,float]] = (8, 8),
            - plot_margin : tuple[float,float,float,float] = (0.15, 0.95, 0.9, 0.13), # (left,right,top,bottom) Margin
            - plot_line_with : Union[int,float] = 3,
            - plot_line_alpha : float = 0.8,
            - diagnoline_width : Union[int,float] = 3,
            - x_axis_limit : Optional[tuple[Union[int,float],Union[int,float]]] = None,
            - x_axis_label : str = 'False Positive Rate',
            - x_axis_label_size : Union[int,float] = 26,
            - x_tick_size : Union[int,float] = 24,
            - x_tick_number : int = 6,
            - y_axis_limit : Optional[tuple[Union[int,float],Union[int,float]]] = None,
            - y_axis_label : str = 'True Positive Rate',
            - y_axis_label_size : Union[int,float] = 26,
            - y_tick_size : Union[int,float] = 24,
            - y_tick_number : int = 6,
            - axis_line_size_left : Union[int,float,None] = 1.5,
            - axis_line_size_right : Union[int,float,None] = 1.5,
            - axis_line_size_top : Union[int,float,None] = 1.5,
            - axis_line_size_bottom : Union[int,float,None] = 1.5,
            - legend : bool = True,
            - legend_location : str = 'lower right',
            - legend_fontsize : Union[int,float] = 20,
            - legend_title : str = '',
            - legend_title_fontsize : Union[int,float] = 24,
            - background_grid : bool = False,
            - show_plot : bool = False
            If the default value of above plotting parameter is None, it refers to default relative values.

        residual_config : Union[str,dict[str,Any],None]
            Configuration of residual analysis options. The default is 'default', using default settings.
            If None, the residual analysis is skipped.

        influence_analysis_config : Union[str,dict,None], optional
            Configuration of influence analysis. The default is 'default', using default settings.
            If None, Influence analysis is skipped.

            When enabled, calculates the Cook's distance-like influence of each sample on the model's predictions using a Leave-One-Out (LOO) approach.
            Please note this computation is highly time-consuming for large sample size. To save time, use a simple validation method or set this to None.

            The parameters of validation include:
            - validation_method : bool, optional
                Independent validation_method for leave-one-out analysis of data point influence.
                Default is using model validation method if it is train-test split, and "2-fold" if the model validation method is "k-fold" or "loo".
            - random_state : None, optional
                random state for data splitting. If None, the random state is not fixed. The default is None.
        """  # noqa: E501

        # Data split
        if data_split_config == "default":
            self._data_split()
        elif type(data_split_config) is dict:
            self._data_split(**data_split_config)

        # Training
        if validation_config == "default":
            self._classifier_validation()
        elif type(validation_config) is dict:
            self._classifier_validation(**validation_config)

        # Performance metrics
        if metrics_config is not None:
            if metrics_config == "default":
                self._classifier_metrics()
            elif type(metrics_config) is dict:
                self._classifier_metrics(**metrics_config)

        # ROC curve
        if roc_plot_config is not None:
            if roc_plot_config == "default":
                self._classifier_roc_curve()
            elif type(roc_plot_config) is dict:
                self._classifier_roc_curve(**roc_plot_config)

        # Residual analysis
        if residual_config is not None:
            if residual_config == "default":
                self._classifier_residual()
            elif type(residual_config) is dict:
                self._classifier_residual(**residual_config)

        # Cook's distance
        if influence_analysis_config is not None:
            if influence_analysis_config == "default":
                self._classifier_influential_analysis()
            elif type(influence_analysis_config) is dict:
                self._classifier_influential_analysis(**influence_analysis_config)

        # Model for application
        self.train_application_model()

    # Train training folds and predict target fold (Regressor)
    @simple_type_validator
    def _regressor_validation(  # noqa: C901
        self,
        use_original_shape: bool = False,
        save_fold_model: bool = True,
        save_fold_data: bool = True,
    ) -> None:
        """
        Training and validating regression model performance on the specified data of 'ModelEva'.
        By running this function, sample validation results are updated.
        """
        if not self.silent_all:
            print(f"\nModel validation, Dataset: '{self.data_label}', Model: '{self.model_label}'")

        # Model validation time
        self._model_time = datetime.now().strftime("created_at_%Y-%m-%d_%H-%M-%S")

        # Get and validate data
        sid = self._sid
        X = self._X  # noqa: N806
        y = self._y
        if y.dtype.kind in ("U", "S"):
            raise TypeError(f"Expected numeric y dtype, got: '{y.dtype}'")

        # Reshape X sample data if required (e.g. for wrapped torch models)
        if use_original_shape:
            for sdim in self._X_original_shape:
                if sdim <= 0:
                    raise ValueError(
                        f"Dimensions of X_original_shape must be positive, but got shape: {self._X_original_shape}"
                    )
            if len(self._X_original_shape) == 1:
                pass
            elif len(self._X_original_shape) > 1:
                X = X.reshape(X.shape[0], *self._X_original_shape)  # noqa: N806
            else:
                raise ValueError(f"Invalid X_original_shape: {self._X_original_shape}")

        # Fold indices
        dsps = self._dsp_inds

        # Training
        itr = 0
        sid_eva = []
        for train_ind, test_ind in dsps:
            if type(self._validation_method) is tuple:
                if not self.silent_all:
                    print("Model training ...")
            else:
                if not self.silent_all:
                    print(f"\rTraining fold: {itr + 1}/{len(dsps)}", end="", flush=True)
            # Model
            model = copy.deepcopy(self._model)
            # Sample ids of target values in validation
            sid_test = sid[test_ind]
            sid_train = sid[train_ind]
            # Fold data for training and testing
            X_train, X_test = X[train_ind], X[test_ind]  # noqa: N806
            y_train, y_test = y[train_ind], y[test_ind]
            # Fit
            model.fit(X_train, y_train.flatten())  # type: ignore[attr-defined]
            # Custom model, independent runtime validated, following the same
            # Predict
            y_test_pred = model.predict(X_test).reshape(-1, 1)  # type: ignore[attr-defined]
            # Store results
            if itr == 0:
                y_true = y_test
                y_pred = y_test_pred
                sid_eva = list(sid_test)
            else:
                y_true = np.concatenate((y_true, y_test), axis=0)
                y_pred = np.concatenate((y_pred, y_test_pred), axis=0)
                sid_eva = sid_eva + list(sid_test)
            # Save fold results to files
            if save_fold_model:
                self._dump_val_model(
                    fold_i=itr,
                    X_train=X_train,
                    y_train=y_train,
                    sid_train=sid_train,
                    X_test=X_test,
                    y_test=y_test,
                    sid_test=sid_test,
                    y_true_all=y,
                    fold_model=model,
                    dump_associated_data=save_fold_data,
                    attach_proba=False,
                )
            itr = itr + 1
        if not self.silent_all:
            print("")

        # Update model validation results
        self._y_true_eva = y_true
        self._y_pred_eva = y_pred
        self._sid_eva = list(sid_eva)

        # Report directory
        dout = self._report_directory + f"Model_evaluation_reports/Data_{self.data_label}_Model_{self.model_label}/"
        if not os.path.exists(dout):
            os.makedirs(dout)

        # Result dataframe for write to file
        coln_val = ["Sample_ID", "y_true", "y_predicted"]
        df_val = pd.DataFrame(np.zeros((len(y_true), len(coln_val))), columns=coln_val)
        df_val["Sample_ID"] = sid_eva
        df_val["y_true"] = y_true
        df_val["y_predicted"] = y_pred

        # Write result to file
        task_time = self._model_time
        df_val.to_csv(dout + f"Validation_results_{self.model_label}.csv", index=False)
        if self.result_backup:
            df_val.to_csv(dout + f"Validation_results_{self.model_label}_{task_time}.csv", index=False)

    # Regressor metrics
    @simple_type_validator
    def _regressor_metrics(self) -> None:
        """
        Calculate and save regressor performance evaluation metrics based on true and predicted target values.
        Return the dataframe of the evaluation metrics if required.
        """
        # Get data
        y_true = self._y_true_eva
        y_pred = self._y_pred_eva

        if not self.silent_all:
            print("Calculating performance metrics ...")
        # Validate data settings
        if (y_true is None) & (y_pred is None):
            pass
        elif (y_true is not None) & (y_pred is not None):
            pass
        else:
            warnings.warn(
                "y_true, y_pred and y_pred_proba is partially specified, \
                    the specified values must match the unspecified default values",
                UserWarning,
                stacklevel=3,
            )

        # Validate data
        y_true = arraylike_validator(ndim=2)(y_true)
        y_pred = arraylike_validator(ndim=2)(y_pred)

        # Validate type of target variables for classifier
        if y_true.dtype.kind in ("U", "S"):
            raise TypeError(f"Expected discontinuous y dtype, got: '{y_true.dtype}'")
        if y_pred.dtype.kind in ("U", "S"):
            raise TypeError(f"Expected discontinuous y dtype, got: '{y_pred.dtype}'")

        # Regression performance metrics
        me = np.sum(y_true - y_pred) / len(y_true)
        mae = mean_absolute_error(y_true, y_pred)
        nmae = mae / (np.max(y_true) - np.min(y_true))
        cvmae = mae / np.mean(y_true)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse**0.5
        nrmse = rmse / (np.max(y_true) - np.min(y_true))
        cvrmse = rmse / np.mean(y_true)
        sde = (mse - me**2) ** 0.5
        rpd = np.std(y_true) / rmse
        r2 = r2_score(y_true, y_pred)
        metrics_df = pd.DataFrame(
            {
                "Mean_Error": [me],
                "Standard_Deviation_of_Error": [sde],
                "Mean_Absolute_Error": [mae],
                "Normalized_MAE": [nmae],
                "CV_MAE": [cvmae],
                "Mean_Squared_Error": [mse],
                "Root_Mean_Squared_Error": [rmse],
                "Normalized_RMSE": [nrmse],
                "CV_RMSE": [cvrmse],
                "Residual_Prediction_Deviation": [rpd],
                "R2": [r2],
            }
        )

        # Report directory
        dout = self._report_directory + f"Model_evaluation_reports/Data_{self.data_label}_Model_{self.model_label}/"
        if not os.path.exists(dout):
            os.makedirs(dout)

        # Save metrics df to CSV
        task_time = self._model_time
        metrics_df.to_csv(dout + f"Regression_performance_{self.model_label}.csv", index=False)
        if self.result_backup:
            metrics_df.to_csv(dout + f"Regression_performance_{self.model_label}_{task_time}.csv", index=False)

    # Scatter plots for regression performance evaluation
    @simple_type_validator
    def _regressor_scatter_plot(  # noqa: C901
        self,
        plot_title: str = "",
        title_size: Union[int, float] = 26,
        title_pad: Union[int, float, None] = None,
        figure_size: tuple[Union[int, float], Union[int, float]] = (8, 8),
        plot_margin: tuple[float, float, float, float] = (
            0.2,
            0.95,
            0.95,
            0.15,
        ),  # (left,right,top,bottom)
        plot_line_with: Union[int, float] = 3,
        point_size: Union[int, float] = 120,
        point_color: str = "mediumblue",
        point_alpha: float = 0.7,
        x_axis_limit: Optional[tuple[Union[int, float], Union[int, float]]] = None,
        x_axis_label: str = "True target values",
        x_axis_label_size: Union[int, float] = 26,
        x_tick_values: Optional[list[Union[int, float]]] = None,
        x_tick_size: Union[int, float] = 24,
        x_tick_number: int = 5,
        y_axis_limit: Optional[tuple[Union[int, float], Union[int, float]]] = None,
        y_axis_label: str = "Predicted target values",
        y_axis_label_size: Union[int, float] = 26,
        y_tick_values: Optional[list[Union[int, float]]] = None,
        y_tick_size: Union[int, float] = 24,
        y_tick_number: int = 5,
        axis_line_size_left: Union[int, float, None] = 1.5,
        axis_line_size_right: Union[int, float, None] = 1.5,
        axis_line_size_top: Union[int, float, None] = 1.5,
        axis_line_size_bottom: Union[int, float, None] = 1.5,
        background_grid: bool = False,
        show_plot: bool = False,
    ) -> None:
        """
        Plot regressor scatter plot on 'ModelEva' validation data.
        """
        plt.ioff()
        if not self.silent_all:
            print("Generating scatter plot ...")
        # Get data
        y_true = self._y_true_eva
        y_pred = self._y_pred_eva
        # Validate result existence
        assert y_true is not None
        assert y_pred is not None

        # Axis parameters
        tmax = round_digit(max(np.max(y_true), np.max(y_pred)), 2, "ceil")
        tmin = round_digit(min(np.min(y_true), np.min(y_pred)), 2, "floor")
        minlim = tmin - 0.025 * (tmax - tmin)
        maxlim = tmax + 0.025 * (tmax - tmin)

        # Plot size
        plt.figure(figsize=figure_size)

        # Scatter plot
        plt.figure(figsize=figure_size)
        plt.scatter(y_true, y_pred, color=point_color, s=point_size, alpha=point_alpha, edgecolor="none")

        # Plot diagonal line
        plt.plot([minlim, maxlim], [minlim, maxlim], color="black", linewidth=plot_line_with)

        # Plot title
        if len(plot_title) > 0:
            if title_pad is None:
                title_pad = int(title_size / 2) + 1
            plt.title(plot_title, fontsize=title_size, pad=title_pad)

        # Plot axis
        if x_axis_limit is not None:
            if x_axis_limit[0] < x_axis_limit[1]:
                plt.xlim(x_axis_limit[0], x_axis_limit[1])
            else:
                raise ValueError(f"Invalid x_axis_limit range: {x_axis_limit}")
        else:
            plt.xlim(minlim, maxlim)
        if y_axis_limit is not None:
            if y_axis_limit[0] < y_axis_limit[1]:
                plt.ylim(y_axis_limit[0], y_axis_limit[1])
            else:
                raise ValueError(f"Invalid y_axis_limit range: {y_axis_limit}")
        else:
            plt.ylim(minlim, maxlim)
        if len(x_axis_label) > 0:
            plt.xlabel(x_axis_label, fontsize=x_axis_label_size)
        if len(y_axis_label) > 0:
            plt.ylabel(y_axis_label, fontsize=y_axis_label_size)

        # Get the current axis
        ax = plt.gca()

        # Set axis ticks
        if x_tick_values is None:
            if x_tick_number < 2:
                raise ValueError(f"x_tick_number must be larger than 1, but got: {x_tick_number}")
            x_tick_interval = round_digit((tmax - tmin) / (x_tick_number - 1), 2, "ceil")
            x_tick_values = [(tmin + nb * x_tick_interval) for nb in range(x_tick_number)]
        ax.set_xticks(x_tick_values)
        ax.tick_params(axis="x", labelsize=x_tick_size)

        # Set y axis ticks
        if y_tick_values is None:
            if y_tick_number < 2:
                raise ValueError(f"y_tick_number must be larger than 1, but got: {y_tick_number}")
            y_tick_interval = round_digit((tmax - tmin) / (y_tick_number - 1), 2, "ceil")
            y_tick_values = [(tmin + nb * y_tick_interval) for nb in range(y_tick_number)]
        ax.set_yticks(y_tick_values)
        ax.tick_params(axis="y", labelsize=y_tick_size)

        # Axis line sizes
        if axis_line_size_top is not None:
            if axis_line_size_top > 0:
                ax.spines["top"].set_linewidth(axis_line_size_top)  # Top axis line
        else:
            ax.spines["top"].set_visible(False)  # Top axis line
        if axis_line_size_bottom is not None:
            if axis_line_size_bottom > 0:
                ax.spines["bottom"].set_linewidth(axis_line_size_bottom)  # Bottom axis line
        else:
            ax.spines["bottom"].set_visible(False)  # Bottom axis line
        if axis_line_size_left is not None:
            if axis_line_size_left > 0:
                ax.spines["left"].set_linewidth(axis_line_size_left)  # Left axis line
        else:
            ax.spines["left"].set_visible(False)  # Left axis line
        if axis_line_size_right is not None:
            if axis_line_size_right > 0:
                ax.spines["right"].set_linewidth(axis_line_size_right)  # Right axis line
        else:
            ax.spines["right"].set_visible(False)  # Right axis line

        # Adjust margins
        plt.subplots_adjust(left=plot_margin[0], right=plot_margin[1], top=plot_margin[2], bottom=plot_margin[3])

        # Report directory
        dout = self._report_directory + f"Model_evaluation_reports/Data_{self.data_label}_Model_{self.model_label}/"
        if not os.path.exists(dout):
            os.makedirs(dout)

        # Save figure
        task_time = self._model_time
        plt.savefig(dout + f"Scatter_plot_{self.model_label}.png", dpi=300)
        if self.result_backup:
            plt.savefig(dout + f"Scatter_plot_{self.model_label}_{task_time}.png", dpi=300)
        if show_plot:
            if not self.silent_all:
                plt.show()
        plt.close()
        plt.ion()

    # Residual plots for regression performance evaluation
    @simple_type_validator
    def _regressor_residual_plot(  # noqa: C901
        self,
        plot_title: str = "",
        title_size: Union[int, float] = 26,
        title_pad: Union[int, float, None] = None,
        figure_size: tuple[Union[int, float], Union[int, float]] = (8, 8),
        plot_margin: tuple[float, float, float, float] = (
            0.2,
            0.95,
            0.95,
            0.15,
        ),  # (left,right,top,bottom)
        plot_line_with: Union[int, float] = 3,
        point_size: Union[int, float] = 120,
        point_color: str = "firebrick",
        point_alpha: float = 0.7,
        x_axis_limit: Optional[tuple[Union[int, float], Union[int, float]]] = None,
        x_axis_label: str = "Predicted target values",
        x_axis_label_size: Union[int, float] = 26,
        x_tick_values: Optional[list[Union[int, float]]] = None,
        x_tick_size: Union[int, float] = 24,
        x_tick_number: int = 5,
        y_axis_limit: Optional[tuple[Union[int, float], Union[int, float]]] = None,
        y_axis_label: str = "Residuals",
        y_axis_label_size: Union[int, float] = 26,
        y_tick_values: Optional[list[Union[int, float]]] = None,
        y_tick_size: Union[int, float] = 24,
        y_tick_number: int = 5,
        axis_line_size_left: Union[int, float, None] = 1.5,
        axis_line_size_right: Union[int, float, None] = 1.5,
        axis_line_size_top: Union[int, float, None] = 1.5,
        axis_line_size_bottom: Union[int, float, None] = 1.5,
        background_grid: bool = False,
        show_plot: bool = False,
    ) -> None:
        """
        Plot regressor residual plot on 'ModelEva' validation data.
        """
        plt.ioff()
        if not self.silent_all:
            print("Generating residual plot ...")
        # Get data
        y_true = self._y_true_eva
        y_pred = self._y_pred_eva
        # Validate result existence
        assert y_true is not None
        assert y_pred is not None

        residuals = np.array(y_true) - np.array(y_pred)

        # Axis parameters
        xmin = round_digit(min(np.min(y_true), np.min(y_pred)), 2, "floor")
        xmax = round_digit(max(np.max(y_true), np.max(y_pred)), 2, "ceil")
        xrange = xmax - xmin
        minxlim = xmin - 0.025 * (xrange)
        maxxlim = xmax + 0.025 * (xrange)
        rad = max(
            abs(round_digit(np.min(residuals), 2, "floor")),
            abs(round_digit(np.max(residuals), 2, "ceil")),
        )
        ymin = -rad
        ymax = rad
        yrange = ymax - ymin
        minylim = ymin - 0.025 * (yrange)
        maxylim = ymax + 0.025 * (yrange)

        # Plot size
        plt.figure(figsize=figure_size)

        # Residual plot
        plt.scatter(y_pred, residuals, color=point_color, s=point_size, alpha=point_alpha, edgecolor="none")

        # Plot horizontal line
        plt.axhline(0, color="black", linestyle="--", linewidth=plot_line_with)

        # Plot title
        if len(plot_title) > 0:
            if title_pad is None:
                title_pad = int(title_size / 2) + 1
            plt.title(plot_title, fontsize=title_size, pad=title_pad)

        # Plot axis
        if x_axis_limit is not None:
            if x_axis_limit[0] < x_axis_limit[1]:
                plt.xlim(x_axis_limit[0], x_axis_limit[1])
            else:
                raise ValueError(f"Invalid x_axis_limit range: {x_axis_limit}")
        else:
            plt.xlim(minxlim, maxxlim)
        if y_axis_limit is not None:
            if y_axis_limit[0] < y_axis_limit[1]:
                plt.ylim(y_axis_limit[0], y_axis_limit[1])
            else:
                raise ValueError(f"Invalid y_axis_limit range: {y_axis_limit}")
        else:
            plt.ylim(minylim, maxylim)
        if len(x_axis_label) > 0:
            plt.xlabel(x_axis_label, fontsize=x_axis_label_size)
        if len(y_axis_label) > 0:
            plt.ylabel(y_axis_label, fontsize=y_axis_label_size)

        # Get the current axis
        ax = plt.gca()

        # Set axis ticks
        if x_tick_values is None:
            if x_tick_number < 2:
                raise ValueError(f"x_tick_number must be larger than 1, but got: {x_tick_number}")
            x_tick_interval = round_digit((xmax - xmin) / (x_tick_number - 1), 2, "ceil")
            x_tick_values = [(xmin + nb * x_tick_interval) for nb in range(x_tick_number)]
        ax.set_xticks(x_tick_values)
        ax.tick_params(axis="x", labelsize=x_tick_size)

        # Set y axis ticks
        if y_tick_values is None:
            if y_tick_number < 2:
                raise ValueError(f"y_tick_number must be larger than 1, but got: {y_tick_number}")
            y_tick_interval = round_digit(rad / int(y_tick_number / 2), 2, "ceil")
            y_tick_values = (
                [(-y_tick_interval * (hnb + 1)) for hnb in range(math.ceil(y_tick_number / 2))]
                + [0.0]
                + [(y_tick_interval * (hnb + 1)) for hnb in range(math.ceil(y_tick_number / 2))]
            )
            y_tick_values = sorted(y_tick_values)
        ax.set_yticks(y_tick_values)
        ax.tick_params(axis="y", labelsize=y_tick_size)

        # Axis line sizes
        if axis_line_size_top is not None:
            if axis_line_size_top > 0:
                ax.spines["top"].set_linewidth(axis_line_size_top)  # Top axis line
        else:
            ax.spines["top"].set_visible(False)  # Top axis line
        if axis_line_size_bottom is not None:
            if axis_line_size_bottom > 0:
                ax.spines["bottom"].set_linewidth(axis_line_size_bottom)  # Bottom axis line
        else:
            ax.spines["bottom"].set_visible(False)  # Bottom axis line
        if axis_line_size_left is not None:
            if axis_line_size_left > 0:
                ax.spines["left"].set_linewidth(axis_line_size_left)  # Left axis line
        else:
            ax.spines["left"].set_visible(False)  # Left axis line
        if axis_line_size_right is not None:
            if axis_line_size_right > 0:
                ax.spines["right"].set_linewidth(axis_line_size_right)  # Right axis line
        else:
            ax.spines["right"].set_visible(False)  # Right axis line

        # Adjust margins
        plt.subplots_adjust(left=plot_margin[0], right=plot_margin[1], top=plot_margin[2], bottom=plot_margin[3])

        # Report directory
        dout = self._report_directory + f"Model_evaluation_reports/Data_{self.data_label}_Model_{self.model_label}/"
        if not os.path.exists(dout):
            os.makedirs(dout)

        # Save figure
        task_time = self._model_time
        plt.savefig(dout + f"Residual_plot_{self.model_label}.png", dpi=300)
        if self.result_backup:
            plt.savefig(dout + f"Residual_plot_{self.model_label}_{task_time}.png", dpi=300)
        if show_plot:
            if not self.silent_all:
                plt.show()
        plt.close()
        plt.ion()

    # Case analysis - residual analysis
    @simple_type_validator
    def _regressor_residual(self) -> None:
        """
        Case analysis report for regression models on the validation data.
        """
        if not self.silent_all:
            print("Calculating residual report ...")
        # Sample IDs
        sid = self._sid_eva

        # Target variables
        y_true = self._y_true_eva
        y_pred = self._y_pred_eva

        # Validate result existence
        assert sid is not None
        assert y_true is not None
        assert y_pred is not None

        # Residuals
        res = y_true - y_pred
        df_res = pd.DataFrame(res, columns=["Residual"], index=sid)
        df_res.index.name = "Sample_ID"

        # Standardized residuals
        median_res = np.nanmedian(res)
        mad = np.nanmedian(np.abs(res - median_res))
        df_res["Standardized_residuals"] = (res - median_res) / (1.4826 * mad)

        # Add true and predicted proba
        y_true_df = pd.DataFrame(y_true, columns=["True_y"], index=sid)
        y_pred_df = pd.DataFrame(y_pred, columns=["Predicted_y"], index=sid)
        df_res = pd.concat([df_res, y_true_df], axis=1)
        df_res = pd.concat([df_res, y_pred_df], axis=1)

        # Report directory
        dout = self._report_directory + f"Model_evaluation_reports/Data_{self.data_label}_Model_{self.model_label}/"
        if not os.path.exists(dout):
            os.makedirs(dout)

        # Save case report df to CSV
        task_time = self._model_time
        df_res.to_csv(dout + f"Residual_analysis_{self.model_label}.csv", index=True, index_label="Sample_ID")
        if self.result_backup:
            df_res.to_csv(
                dout + f"Residual_analysis_{self.model_label}_{task_time}.csv",
                index=True,
                index_label="Sample_ID",
            )

    # LOO training for data point influential analysis
    @simple_type_validator
    def _regressor_influential_analysis(
        self, validation_method: str = 'default', random_state: Optional[int] = None
    ) -> None:
        """
        Calculate the cooks-distance-like average influence on predictions of each sample using LOO approach.

        Validation_method "default" uses "2-fold" or train-test-split if model validation method is train-test-split.
        Set validation_method "model" to use model validation method.
        """
        # Model validation time
        self._model_time = datetime.now().strftime("created_at_%Y-%m-%d_%H-%M-%S")

        # Data
        sid = self._sid
        X = self._X  # noqa: N806
        y = self._y

        # Fold indices
        dsps = self._influence_data_split_ids(validation_method=validation_method, random_state=random_state)

        # Validation Training
        influence_list = []
        itr1 = 0
        for train_ind, test_ind in dsps:
            if not self.silent_all:
                print(f"\rTraining Cook's distance fold: {itr1 + 1}/{len(dsps)}", end="", flush=True)
            ## Get data
            # Sample ids of target values in validation
            sid_test = sid[test_ind]
            # Fold data for training and testing
            X_train, X_test = X[train_ind], X[test_ind]  # noqa: N806
            y_train, y_test = y[train_ind], y[test_ind]

            # Calculate MSE
            model_full = copy.deepcopy(self._model)
            model_full.fit(X_train, y_train.flatten())  # type: ignore[attr-defined]
            # Custom model, independent runtime validated, following the same
            p_full = model_full.predict(X_test)  # type: ignore[attr-defined]
            p_full = pd.DataFrame(p_full, columns=["y_pred"], index=sid_test)
            mse = np.array(np.mean((y_test - p_full) ** 2, axis=0))
            # Old: mse = np.mean((y_test - p_full) ** 2, axis=0)

            # Replace pandas nan
            if hasattr(pd.DataFrame, 'map'):
                # pandas >= 2.1.0
                p_full = p_full.map(lambda x: np.nan if pd.isna(x) else x)
            else:
                # pandas < 2.1.0
                p_full = p_full.applymap(lambda x: np.nan if pd.isna(x) else x)
            # Old: p_full = p_full.applymap(lambda x: np.nan if pd.isna(x) else x)

            # Convert back to array
            p_full = np.array(p_full)

            # Cook's dist LOO training
            influence = np.zeros((X.shape[0],))
            for i in range(X_train.shape[0]):
                # LOO-Model
                model_loo = copy.deepcopy(self._model)

                # LOO-data
                X_loo = np.delete(X_train, i, axis=0)  # Leave-one-out dataset  # noqa: N806
                y_loo = np.delete(y_train, i, axis=0)

                # LOO-training-n-prediction
                model_loo.fit(X_loo, y_loo.flatten())  # type: ignore[attr-defined]
                # Custom model, independent runtime validated, following the same

                # Predict on test X
                p_loo = model_loo.predict(X_test)  # type: ignore[attr-defined]
                p_loo = pd.DataFrame(p_loo, columns=["y_pred"])

                # Replace pandas nan
                if hasattr(pd.DataFrame, 'map'):
                    # pandas >= 2.1.0
                    p_loo = p_loo.map(lambda x: np.nan if pd.isna(x) else x)
                else:
                    # pandas < 2.1.0
                    p_loo = p_loo.applymap(lambda x: np.nan if pd.isna(x) else x)
                # Old: p_loo = p_loo.applymap(lambda x: np.nan if pd.isna(x) else x)

                # Convert back to array
                p_loo = np.array(p_loo)

                # Calculate Cook's dist
                if X_train.shape[1] < 1:
                    raise ValueError(f"Invalid X_train, got: {X_train}, type: {type(X_train)}, shape: {X_train.shape}")
                mse1 = mse + 1e-30
                influence[train_ind[i]] = np.sum((np.array(p_full) - np.array(p_loo)) ** 2, axis=0) / (
                    X_train.shape[1] * mse1
                )
                # Old: influence[train_ind[i]] = np.sum((p_full - p_loo) ** 2, axis=0) / (X_train.shape[1] * mse1)

            # Store results
            influence_list.append(influence)
            itr1 = itr1 + 1

        if not self.silent_all:
            print("")

        # Avg influence
        influence_avg = np.sum(influence_list, axis=0) / (X_train.shape[0] - 1)
        influence_df = pd.DataFrame(influence_avg, columns=["Cooks_distance_like"], index=sid)
        influence_df.index.name = "Sample_ID"

        # Report directory
        dout = self._report_directory + f"Model_evaluation_reports/Data_{self.data_label}_Model_{self.model_label}/"
        if not os.path.exists(dout):
            os.makedirs(dout)

        # Save metrics df to CSV
        task_time = self._model_time
        influence_df.to_csv(dout + f"Influence_analysis_{self.model_label}.csv", index=True, index_label="Sample_ID")
        if self.result_backup:
            influence_df.to_csv(
                dout + f"Influence_analysis_{self.model_label}_{task_time}.csv",
                index=True,
                index_label="Sample_ID",
            )

    # Evaluation of regressor performance
    @simple_type_validator
    def regressor_evaluation(  # noqa: C901
        self,
        data_split_config: Union[str, dict[str, Any]] = "default",
        validation_config: Union[str, dict[str, Any]] = "default",
        metrics_config: Union[str, dict[str, Any], None] = "default",
        scatter_plot_config: Union[str, dict[str, Any], None] = "default",
        residual_config: Union[str, dict[str, Any], None] = "default",
        residual_plot_config: Union[str, dict[str, Any], None] = "default",
        influence_analysis_config: Union[str, dict[str, Any], None] = "default",
    ) -> None:
        """
        Perform regressor performance evaluation using test dataset or cross-validation.

        The configuration parameters must be provided in a dictionary of subparameters with the format of {parameter name (key) : parameter values (value)}.
        This method evaluates model performance and generates comprehensive reports saved to the reporting directory.

        The evaluation includes:
        - model validation results
        - performance metrics
        - case analysis of residuals & Cook's distance
        - scatter plot of true and predicted target values
        - residual analysis report
        - residual plot

        Parameters
        ----------
        data_split_config : Union[str,dict], optional
            Configuration of data split options in dictionary. The default is 'default', using default settings.

            The parameters of data split include:
            - random_state : Optional[int], optional
                Random state for data splitting and shuffling. If None, random_state is not fixed. The default is None.
            - validation_method : Optional[str], optional
                Model validation method, default is using 'ModelEva.validation_method', see 'ModelEva' for details.

        validation_config : Union[str,dict], optional
            Configuration of validation options. The default is 'default', using default settings.

            The parameters of validation include:
            - unseen_threshold : Optional[float]
                For classification models trained on data with missing classes, a sample is assigned to a unknown class if its highest predicted probability among the known classes is below the unseen_threshold.
            - use_original_shape : bool
                Whether the sample data is reshaped to its original shape.
                If False, the flattened data is used. The default is False.
            - save_fold_model: bool
                Whether to save model of each fold. The default is True.
            - save_fold_data: bool
                Whether to save training and validation data of each fold. The default is True.
                Saving fold data and fold models could consuming significant storage when applied to large data with large sample size with large fold numbers.

        metrics_config : Union[str,dict,None], optional
            Configuration of metrics options. The default is 'default', using default settings.
            If None, the performance metrics computation is skipped.

        scatter_plot_config : Union[str,dict,None], optional
            Configuration of scatter plot. The default is 'default', using default settings.
            If None, the ROC curve plotting is skipped.

            The parameters of scatter_plot_config and its default values are listed as follows:
            - plot_title : str = '',
            - title_size : Union[int,float] = 26,
            - title_pad : Union[int,float,None] = None,
            - figure_size : tuple[Union[int,float],Union[int,float]] = (8, 8),
            - plot_margin : tuple[float,float,float,float] = (0.2, 0.95, 0.95, 0.15), # (left,right,top,bottom) Margin
            - plot_line_with : Union[int,float] = 3,
            - point_size : Union[int,float] = 120,
            - point_color : str = 'firebrick',
            - point_alpha : float = 0.7,
            - x_axis_limit : Optional[tuple[Union[int,float],Union[int,float]]] = None,
            - x_axis_label : str = 'Predicted target values',
            - x_axis_label_size : Union[int,float] = 26,
            - x_tick_values : Optional[list[Union[int,float]]] = None,
            - x_tick_size : Union[int,float] = 24,
            - x_tick_number : int = 5,
            - y_axis_limit : Optional[tuple[Union[int,float],Union[int,float]]] = None,
            - y_axis_label : str = 'Residuals',
            - y_axis_label_size : Union[int,float] = 26,
            - y_tick_values : Optional[list[Union[int,float]]] = None,
            - y_tick_size : Union[int,float] = 24,
            - y_tick_number : int = 5,
            - axis_line_size_left : Union[int,float,None] = 1.5,
            - axis_line_size_right : Union[int,float,None] = 1.5,
            - axis_line_size_top : Union[int,float,None] = 1.5,
            - axis_line_size_bottom : Union[int,float,None] = 1.5,
            - background_grid : bool = False,
            - show_plot : bool = False
            If the default value of above plotting parameter is None, it refers to default relative values.

        residual_config : Union[str,dict[str,Any],None]
            Configuration of residual analysis options. The default is 'default', using default settings.
            If None, the residual analysis is skipped.

        residual_plot_config : Union[str,dict,None], optional
            Configuration of residual plot parameters is same as scatter_plot_config, see 'scatter_plot_config' for details.
            The parameters is same as 'scatter_plot'.

        influence_analysis_config : Union[str,dict,None], optional
            Configuration of influence analysis. The default is 'default', using default settings.
            If None, Influence analysis is skipped.

            When enabled, calculates the Cook's distance-like influence of each sample on the model's predictions using a Leave-One-Out (LOO) approach.
            Please note this computation is highly time-consuming for large sample size. To save time, use a simple validation method or set this to None.

            The parameters of validation include:
            - validation_method : bool, optional
                Independent validation_method for leave-one-out analysis of data point influence.
                Default is using model validation method if it is train-test split, and "2-fold" if the model validation method is "k-fold" or "loo".
            - random_state : None, optional
                random state for data splitting. If None, the random state is not fixed. The default is None.
        """  # noqa: E501

        # Data split
        if data_split_config == "default":
            self._data_split()
        elif type(data_split_config) is dict:
            self._data_split(**data_split_config)
        else:
            raise ValueError(
                f"If provided, data_split_config must be a dictionary of arguments, \
                             but got: {data_split_config}"
            )

        # Training
        if validation_config == "default":
            self._regressor_validation()
        elif type(validation_config) is dict:
            self._regressor_validation(**validation_config)
        else:
            raise ValueError(
                f"If provided, validation must be a dictionary of arguments, \
                             but got: {validation_config}"
            )

        # Performance metrics
        if metrics_config is not None:
            if metrics_config == "default":
                self._regressor_metrics()
            elif type(metrics_config) is dict:
                self._regressor_metrics(**metrics_config)
            else:
                raise ValueError(
                    f"If provided, metrics_config must be None or a dictionary of arguments, \
                                 but got: {metrics_config}"
                )

        # Scatter plot
        if scatter_plot_config is not None:
            if scatter_plot_config == "default":
                self._regressor_scatter_plot()
            elif type(scatter_plot_config) is dict:
                self._regressor_scatter_plot(**scatter_plot_config)
            else:
                raise ValueError(
                    f"If provided, roc_plot must be None or a dictionary of arguments, \
                        but got: {scatter_plot_config}"
                )

        # Residual analysis
        if residual_config is not None:
            if residual_config == "default":
                self._regressor_residual()
            elif type(residual_config) is dict:
                self._regressor_residual(**residual_config)
            else:
                raise ValueError(
                    f"If provided, residual_config must be None or a dictionary of arguments, \
                        but got: {residual_config}"
                )

        # Residual plot
        if residual_plot_config is not None:
            if residual_plot_config == "default":
                self._regressor_residual_plot()
            elif type(residual_plot_config) is dict:
                self._regressor_residual_plot(**residual_plot_config)
            else:
                raise ValueError(
                    f"If provided, roc_plot must be None or a dictionary of arguments, but got: {residual_plot_config}"
                )

        # Cook's distance
        if influence_analysis_config is not None:
            if influence_analysis_config == "default":
                self._regressor_influential_analysis()
            elif type(influence_analysis_config) is dict:
                self._regressor_influential_analysis(**influence_analysis_config)
            else:
                raise ValueError(
                    f"If provided, influence_analysis_config must be None or a dictionary of arguments, \
                        but got: {influence_analysis_config}"
                )

        # Model for application
        self.train_application_model()

    # Application train
    @simple_type_validator
    def train_application_model(self, return_result: bool = False, dump_result: bool = True) -> Optional[object]:
        """
        Train model for application on the entire dataset.

        Parameters
        ----------
        return_result : bool, optional
            Whether the trained application model is returned. The default is False.
        dump_result : bool, optional
            Whether the trained application model is stored as dill file in the report directory. The default is True.

        Returns
        -------
        model
            The trained model instance.
        """
        # Model validation time
        app_model_create_time = datetime.now().strftime("created_at_%Y-%m-%d_%H-%M-%S")
        self._app_model_create_time = app_model_create_time

        # Get and validate data
        X = self._X  # noqa: N806
        y = self._y

        # Model
        model = copy.deepcopy(self._model)

        # Train model on entire data set
        model.fit(X, y.flatten())  # type: ignore[attr-defined]
        # Custom model, independent runtime validated, following the same

        # Store trained model
        self._app_model = model

        # Dump trained model
        if dump_result:
            # Construct saving dict of data config
            dump_name = f"app_model_{self._model_label}"
            dump_name1 = f"app_model_{self._model_label}_{app_model_create_time}"
            dump_dict = {dump_name: model}
            # Dump path
            dout = (
                self._report_directory
                + f"Model_evaluation_reports/Data_{self.data_label}_Model_{self.model_label}/Model_for_application/"
            )
            if not os.path.exists(dout):
                os.makedirs(dout)
            dump_path = dout + dump_name + ".dill"
            dump_path1 = dout + dump_name1 + ".dill"
            # Dump model
            with open(dump_path, "wb") as f:
                dill.dump(dump_dict, f)
            # Dump model backup
            if self.result_backup:
                with open(dump_path1, "wb") as f:
                    dill.dump(dump_dict, f)

        # Return results
        if return_result:
            return model
        else:
            return None

    # Dump trained models in the model evaluation steps
    @simple_type_validator
    def _dump_val_model(
        self,
        fold_i: int,
        X_train: Annotated[np.ndarray, arraylike_validator(ndim=2)],  # noqa: N803
        y_train: Annotated[np.ndarray, arraylike_validator(ndim=2)],
        sid_train: Annotated[np.ndarray, arraylike_validator(ndim=1)],
        X_test: Annotated[np.ndarray, arraylike_validator(ndim=2)],  # noqa: N803
        y_test: Annotated[np.ndarray, arraylike_validator(ndim=2)],
        sid_test: Annotated[np.ndarray, arraylike_validator(ndim=1)],
        y_true_all: Annotated[np.ndarray, arraylike_validator(ndim=2)],
        fold_model: object,
        y_test_proba: Optional[Annotated[np.ndarray, arraylike_validator(ndim=2)]] = None,
        *,
        dump_associated_data: bool = True,
        attach_proba: Optional[bool] = None,
    ) -> None:
        """
        Dump data and model of a validation fold. Train_test data split is deemed as one-fold.
        """
        # Validate sample IDs
        sid_train = np.array(sid_train).flatten()
        sid_test = np.array(sid_test).flatten()

        # Report fold data
        df_y_true_all = pd.DataFrame(y_true_all, columns=["y_true"], index=self._sid)
        df_X0 = pd.DataFrame(X_train, index=sid_train)  # noqa: N806
        df_y0 = pd.DataFrame(y_train, columns=["y_train"], index=sid_train)
        df_X1 = pd.DataFrame(X_test, index=sid_test)  # noqa: N806
        df_y1 = pd.DataFrame(y_test, columns=["y_test"], index=sid_test)
        dfr = pd.DataFrame(columns=["y_true", "y_train", "y_test"], index=self._sid)
        dfr.update(df_y_true_all)
        dfr.update(df_y0)
        dfr.update(df_y1)

        # Add y_proba data
        if self.is_regression:
            attach_proba = False
        else:
            if attach_proba is None:
                attach_proba = True
        if attach_proba:
            if y_test_proba is None:
                raise ValueError(f"y_test_proba is not provided or None at fold {fold_i}")
            df_yp1 = pd.DataFrame(
                y_test_proba,
                columns=[(str(yn) + "_prob") for yn in fold_model.classes_],  # type: ignore[attr-defined]
                # Custom model, independent runtime validated, following the same
                index=sid_test,
            )
            ynames = self._ynames
            assert ynames is not None
            colnc = [(str(yn) + "_prob") for yn in ynames]
            dfrc = pd.DataFrame(columns=colnc, index=self._sid_eva)
            dfrc = dfrc.combine_first(df_yp1)
            dfr = pd.concat([dfr, dfrc], axis=1)

        # Report path
        dout = (
            self._report_directory
            + f"Model_evaluation_reports/Data_{self.data_label}_Model_{self.model_label}/Model_in_validation/"
        )
        if not os.path.exists(dout):
            os.makedirs(dout)

        ## Dump fold model
        # Construct saving dict of data config
        dump_name = f"val_model_fold-{fold_i}_{self._model_label}"
        dump_name1 = f"val_model_fold-{fold_i}_{self._model_label}_{self._model_time}"
        dump_dict = {dump_name: fold_model}

        # Dump path
        dump_path = dout + dump_name + ".dill"
        dump_path1 = dout + dump_name1 + ".dill"

        # Dump model
        with open(dump_path, "wb") as f:
            dill.dump(dump_dict, f)
        if self.result_backup:
            with open(dump_path1, "wb") as f:
                dill.dump(dump_dict, f)

        # Write fold data to file
        if dump_associated_data:
            wname_X0 = f"val_X-train_fold-{fold_i}_{self._model_label}"  # noqa: N806
            wpath_X0 = dout + wname_X0 + ".csv"  # noqa: N806
            df_X0.to_csv(wpath_X0, header=True, index=True, index_label="Sample_ID")
            wname_X1 = f"val_X-test_fold-{fold_i}_{self._model_label}"  # noqa: N806
            wpath_X1 = dout + wname_X1 + ".csv"  # noqa: N806
            df_X1.to_csv(wpath_X1, header=True, index=True, index_label="Sample_ID")
            wname_y = f"val_y_fold-{fold_i}_{self._model_label}"
            wpath_y = dout + wname_y + ".csv"
            dfr.to_csv(wpath_y, header=True, index=True, index_label="Sample_ID")

        # Write backup
        if self.result_backup:
            wname_X0 = f"val_X-train_fold-{fold_i}_{self._model_label}_{self._model_time}"  # noqa: N806
            wpath_X0 = dout + wname_X0 + ".csv"  # noqa: N806
            df_X0.to_csv(wpath_X0, header=True, index=True, index_label="Sample_ID")
            wname_X1 = f"val_X-test_fold-{fold_i}_{self._model_label}_{self._model_time}"  # noqa: N806
            wpath_X1 = dout + wname_X1 + ".csv"  # noqa: N806
            df_X1.to_csv(wpath_X1, header=True, index=True, index_label="Sample_ID")
            wname_y = f"val_y_fold-{fold_i}_{self._model_label}_{self._model_time}"
            wpath_y = dout + wname_y + ".csv"
            dfr.to_csv(wpath_y, header=True, index=True, index_label="Sample_ID")


# %% Model Ensembler - Not implemented in current version

# TODO: Voting - basic
# TODO: Voting - Bagging
# TODO: Boosting
# TODO: Stacking


# %% Torch constructor - Not implemented in current version

# TODO: Input nn structure and wrapped into sklearn-style
