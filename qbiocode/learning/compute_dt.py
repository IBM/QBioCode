# ====== Base class imports ======

import time

from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import modeleval

# ====== Scikit-learn imports ======


# ====== Begin functions ======


def compute_dt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="Decision Tree",
    data_key="",
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0,
    monotonic_cst=None,
):
    """This function generates a model using a Decision Tree (DT) Classifier method as implemented in
    `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>`__.
    It takes in parameter arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed.
    The model is trained on the training dataset and validated on the test dataset.  The model is trained on the training dataset and validated on the test dataset.
    The function returns the evaluation of the model on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Test data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Test data labels.
        args (dict): Additional arguments, typically from config.yaml.
        verbose (bool): If True, prints additional information during execution.
        model (str): Name of the model being used, default is 'Decision Tree'.
        data_key (str): Key for the dataset, if applicable.
        criterion (str): The function to measure the quality of a split. Default is 'gini'.
        splitter (str): The strategy used to choose the split at each node. Default is 'best'.
        max_depth (int or None): The maximum depth of the tree. Default is None.
        min_samples_split (int): The minimum number of samples required to split an internal node. Default is 2.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node. Default is 1.
        min_weight_fraction_leaf (float): The minimum weighted fraction of the sum total of weights required to be at a leaf node. Default is 0.0.
        max_features (int, float, str or None): The number of features to consider when looking for the best split. Default is None.
        random_state (int or None): Controls the randomness of the estimator. Default is None.
        max_leaf_nodes (int or None): Grow a tree with max_leaf_nodes in best-first fashion. Default is None.
        min_impurity_decrease (float): A node will be split if this split induces a decrease of the impurity greater than or equal to this value. Default is 0.0.
        class_weight (dict or 'balanced' or None): Weights associated with classes in the form {class_label: weight}. Default is None.
        ccp_alpha (float): Complexity parameter used for Minimal Cost-Complexity Pruning. Default is 0.0.
        monotonic_cst: Monotonic constraints for tree nodes, if applicable. Default is None.
    Returns:
        modeleval (dict): A dictionary containing the evaluation metrics, model parameters, and time taken for training and validation.
    """

    beg_time = time.time()
    dt = OneVsOneClassifier(
        DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst,
        )
    )
    # Fit the training datset
    model_fit = dt.fit(X_train, y_train)
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = dt.predict(X_test)
    return modeleval(
        y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose
    )


def compute_dt_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="Decision Tree",
    cv=5,
    criterion=[],
    max_depth=[],
    min_samples_split=[],
    min_samples_leaf=[],
    max_features=[],
):
    """This function also generates a model using a Decision Tree (DT) Classifier method as implemented in
    `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>`__.
    The difference here is that this function runs a grid search. The range of the grid search for each parameter is specified in the config.yaml file. The
    combination of parameters that led to the best performance is saved and returned as best_params, which can then be used on similar
    datasets, without having to run the grid search.
    The model is trained on the training dataset and validated on the test dataset.  The model is trained on the training dataset and validated on the test dataset.
    The function returns the evaluation of the model on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model across the grid search.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Test data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Test data labels.
        args (dict): Additional arguments, typically from config.yaml.
        verbose (bool): If True, prints additional information during execution.
        model (str): Name of the model being used, default is 'Decision Tree'.
        cv (int): Number of cross-validation folds. Default is 5.
        criterion (list): List of criteria to consider for splitting. Default is empty list.
        max_depth (list): List of maximum depths to consider. Default is empty list.
        min_samples_split (list): List of minimum samples required to split an internal node. Default is empty list.
        min_samples_leaf (list): List of minimum samples required to be at a leaf node. Default is empty list.
        max_features (list): List of maximum features to consider when looking for the best split. Default is empty list.

    Returns:
        modeleval (dict): A dictionary containing the evaluation metrics, best parameters, and time taken for training and validation.
    """

    beg_time = time.time()
    params = {
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
    }
    # Perform Grid Search to find the best parameters
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=cv)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and use them to create the final Decision Tree model
    best_params = grid_search.best_params_
    best_dt = DecisionTreeClassifier(**best_params)
    best_dt.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_dt.predict(X_test)
    return modeleval(y_test, y_predicted, beg_time, best_params, args, model=model, verbose=verbose)
