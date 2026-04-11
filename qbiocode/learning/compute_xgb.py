# ====== Base class imports ======

import time

import numpy as np

# ====== Scikit-learn imports ======

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
    _XGBOOST_ERROR = None
except Exception as e:
    # Catch all exceptions including XGBoostError, ImportError, OSError
    XGBOOST_AVAILABLE = False
    _XGBOOST_ERROR = str(e)
    XGBClassifier = None  # type: ignore

from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import modeleval

# ====== Begin functions ======


def compute_xgb(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="xgb",
    data_key="",
    n_estimators=100,
    *,
    criterion="gini",
    max_depth=None,
    subsample=0.5,
    learning_rate=0.5,
    colsample_bytree=1,
    min_child_weight=1,
):
    """
    This function generates a model using an Extreme Gradient Boositing (xgb) Classifier method as implemented in xgboost. It takes in parameter
    arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed.
    The model is trained on the training dataset and validated on the test dataset. The function returns the evaluation of the model
    on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Test data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Test data labels.
        args (dict): Additional arguments, typically from a configuration file.
        verbose (bool): If True, prints additional information during execution.
        model (str): Name of the model being used, default is 'XGBoost'.
        data_key (str): Key for identifying the dataset, default is an empty string.
        n_estimators (int): Number of trees in the forest, default is 100.
        max_depth (int or None): Maximum depth of the tree, default is None.
        subsample (float) : Subsample ratio of the training instances. Default 0.5
        learning_rate (float): Step size shrinkage used in update to prevent overfitting. Default is 0.5
        colsample_bytree  (float): subsample ratio of columns when constructing each tree. Default is 1
        min_child_weight (int) : Minimum sum of instance weight (hessian) needed in a child. Default is 1
     Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model, including accuracy, AUC, F1 score, and the time taken for training and validation.

    Raises:
        ImportError: If XGBoost is not properly installed or configured.

    """

    if not XGBOOST_AVAILABLE:
        error_msg = (
            "XGBoost is not properly installed or configured.\n"
            f"Error: {_XGBOOST_ERROR}\n\n"
            "On macOS, you may need to install OpenMP:\n"
            "  brew install libomp\n\n"
            "Then reinstall XGBoost:\n"
            "  pip install --force-reinstall xgboost\n\n"
            "See installation documentation for more details."
        )
        raise ImportError(error_msg)

    beg_time = time.time()
    xgb = OneVsOneClassifier(
        XGBClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,  # type: ignore
            subsample=subsample,
            learning_rate=learning_rate,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
        )
    )
    # Fit the training datset
    model_fit = xgb.fit(X_train, y_train)
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = xgb.predict(X_test)
    return modeleval(
        y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose
    )


def compute_xgb_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    cv=5,
    model="xgb",
    bootstrap=[],
    max_depth=[],
    max_features=[],
    learning_rate=[],
    subsample=[],
    colsample_bytree=[],
    n_estimators=[],
    min_child_weight=[],
):
    """
    This function generates a model using an Extreme Gradient Boositing (xgb) Classifier method as implemented in xgboost.
    The difference here is that this function runs a grid search. The range of the grid search for each parameter is specified in the config.yaml file. The
    combination of parameters that led to the best performance is saved and returned as best_params, which can then be used on similar
    datasets, without having to run the grid search.
    The model is trained on the training dataset and validated on the test dataset. The function returns the evaluation of the model
    on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model across the grid search.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Test data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Test data labels.
        args (dict): Additional arguments, typically from a configuration file.
        verbose (bool): If True, prints additional information during execution.
        cv (int): Number of cross-validation folds, default is 5.
        model (str): Name of the model being used, default is 'Random Forest'.
        bootstrap (list): List of bootstrap options for grid search.
        max_depth (list): List of maximum depth options for grid search.
        subsample (list): List of subsample ratio of the training instances options for grid search.
        learning_rate (list): List of step size shrinkage used in update to prevent overfitting options for grid search.
        colsample_bytree (list): List of subsample ratio of columns when constructing each tree options for grid search.
        n_estimators (list): List of number of estimators options for grid search.
        min_child_weight (list): List of minimum sum of instance weight (hessian) needed in a childoptions for grid search.

    Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model, including accuracy, AUC, F1 score, and the time taken for training and validation.

    Raises:
        ImportError: If XGBoost is not properly installed or configured.
    """

    if not XGBOOST_AVAILABLE:
        error_msg = (
            "XGBoost is not properly installed or configured.\n"
            f"Error: {_XGBOOST_ERROR}\n\n"
            "On macOS, you may need to install OpenMP:\n"
            "  brew install libomp\n\n"
            "Then reinstall XGBoost:\n"
            "  pip install --force-reinstall xgboost\n\n"
            "See installation documentation for more details."
        )
        raise ImportError(error_msg)

    beg_time = time.time()
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "bootstrap": bootstrap,
    }

    # Perform Grid Search to find the best parameters
    grid_search = GridSearchCV(XGBClassifier(), param_grid=params, cv=cv)  # type: ignore
    grid_search.fit(X_train, y_train)

    # Get the best parameters and use them to create the final model
    best_params = grid_search.best_params_
    best_xgb = XGBClassifier(**best_params)  # type: ignore
    best_xgb.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_xgb.predict(X_test)
    return modeleval(y_test, y_predicted, beg_time, best_params, args, model=model, verbose=verbose)
