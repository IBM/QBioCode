# ====== Base class imports ======

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import modeleval

# ====== Scikit-learn imports ======


# ====== Begin functions ======


def compute_rf(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="Random Forest",
    data_key="",
    n_estimators=100,
    *,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None,
):
    """
    This function generates a model using a Random Forest (RF) Classifier method as implemented in
    `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`__.
    It takes in parameter arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed.
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
        model (str): Name of the model being used, default is 'Random Forest'.
        data_key (str): Key for identifying the dataset, default is an empty string.
        n_estimators (int): Number of trees in the forest, default is 100.
        criterion (str): The function to measure the quality of a split, default is 'gini'.
        max_depth (int or None): Maximum depth of the tree, default is None.
        min_samples_split (int): Minimum number of samples required to split an internal node, default is 2.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node, default is 1.
        min_weight_fraction_leaf (float): Minimum weighted fraction of the sum total of weights required to be at a leaf node, default is 0.0.
        max_features (str or int or float): The number of features to consider when looking for the best split, default is 'sqrt'.
        max_leaf_nodes (int or None): Grow trees with max_leaf_nodes in best-first fashion, default is None.
        min_impurity_decrease (float): A node will be split if this split induces a decrease of the impurity greater than or equal to this value, default is 0.0.
        bootstrap (bool): Whether bootstrap samples are used when building trees, default is True.
        oob_score (bool): Whether to use out-of-bag samples to estimate the generalization accuracy, default is False.
        n_jobs (int or None): Number of jobs to run in parallel for both `fit` and `predict`, default is None.
        random_state (int or None): Controls the randomness of the estimator, default is None.
        warm_start (bool): When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, default is False.
        class_weight (dict or str or None): Weights associated with classes in the form {class_label: weight}, default is None.
        ccp_alpha (float): Complexity parameter used for Minimal
     Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model, including accuracy, AUC, F1 score, and the time taken for training and validation.

    """

    beg_time = time.time()
    rf = OneVsOneClassifier(
        RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst,
        )
    )
    # Fit the training datset
    model_fit = rf.fit(X_train, y_train)
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = rf.predict(X_test)
    return modeleval(
        y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose
    )


def compute_rf_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    cv=5,
    model="Random Forest",
    bootstrap=[],
    max_depth=[],
    max_features=[],
    min_samples_leaf=[],
    min_samples_split=[],
    n_estimators=[],
):
    """
    This function also generates a model using a Random Forest (RF) Classifier method as implemented in
    `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`__.
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
        max_features (list): List of maximum features options for grid search.
        min_samples_leaf (list): List of minimum samples leaf options for grid search.
        min_samples_split (list): List of minimum samples split options for grid search.
        n_estimators (list): List of number of estimators options for grid search.

    Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model, including accuracy, AUC, F1 score, and the time taken for training and validation.

    """

    beg_time = time.time()
    params = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }

    # Perform Grid Search to find the best parameters
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid=params, cv=cv)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and use them to create the final model
    best_params = grid_search.best_params_
    best_rf = RandomForestClassifier(**best_params)
    best_rf.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_rf.predict(X_test)
    return modeleval(y_test, y_predicted, beg_time, best_params, args, model=model, verbose=verbose)
