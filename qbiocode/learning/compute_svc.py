# ====== Base class imports ======

import time

from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import modeleval

# ====== Scikit-learn imports ======


def compute_svc(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    model="SVC",
    data_key="",
    C=1.0,
    kernel="rbf",
    degree=3,
    gamma="scale",
    coef0=0.0,
    shrinking=True,
    probability=False,
    tol=0.001,
    cache_size=200,
    class_weight=None,
    verbose=False,
    max_iter=-1,
    decision_function_shape="ovr",
    break_ties=False,
    random_state=None,
):
    """This function generates a model using a Support Vector Classifier (SVC) method as implemented in
    `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`__.
    It takes in parameter arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed.
    The model is trained on the training dataset and validated on the test dataset.  The model is trained on the training dataset and validated on the test dataset.
    The function returns the evaluation of the model on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Test data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Test data labels.
        args (dict): Additional arguments, typically from a configuration file.
        model (str): The type of model to use, default is 'SVC'.
        data_key (str): Key for the dataset, default is an empty string.
        C (float): Regularization parameter, default is 1.0.
        kernel (str): Specifies the kernel type to be used in the algorithm, default is 'rbf'.
        degree (int): Degree of the polynomial kernel function ('poly'), default is 3.
        gamma (str or float): Kernel coefficient for 'rbf', 'poly', and 'sigmoid', default is 'scale'.
        coef0 (float): Independent term in kernel function, default is 0.0.
        shrinking (bool): Whether to use the shrinking heuristic, default is True.
        probability (bool): Whether to enable probability estimates, default is False.
        tol (float): Tolerance for stopping criteria, default is 0.001.
        cache_size (int): Size of the kernel cache in MB, default is 200.
        class_weight (dict or None): Weights associated with classes, default is None.
        verbose (bool): Whether to print detailed logs, default is False.
        max_iter (int): Hard limit on iterations within solver, -1 means no limit, default is -1.
        decision_function_shape (str): Determines the shape of the decision function, default is 'ovr'.
        break_ties (bool): Whether to break ties in multiclass classification, default is False.
        random_state (int or None): Controls the randomness of the estimator, default is None.
    Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model, including accuracy, AUC, F1 score, and the time taken to train and validate the model.
    """

    beg_time = time.time()
    svc = OneVsOneClassifier(
        SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )
    )
    # Fit the training datset
    model_fit = svc.fit(X_train, y_train)
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = svc.predict(X_test)
    return modeleval(
        y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose
    )


def compute_svc_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    cv=5,
    model="SVC",
    C=[],
    gamma=[],
    kernel=[],
):
    """This function generates a model using a Support Vector Classifier (SVC) method as implemented in
    `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`__.
    It takes in parameter arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed. The
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
        args (dict): Additional arguments, typically from a configuration file.
        verbose (bool): Whether to print detailed logs, default is False.
        cv (int): Number of cross-validation folds, default is 5.
        model (str): The type of model to use, default is 'SVC'.
        C (list or float): Regularization parameter(s), default is an empty list.
        gamma (list or str): Kernel coefficient(s) for 'rbf', 'poly', and 'sigmoid', default is an empty list.
        kernel (list or str): Specifies the kernel type(s) to be used in the algorithm, default is an empty list.
     Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model, including accuracy, AUC, F1 score, and the time taken to train and validate the model across the grid search.
    """

    beg_time = time.time()
    params = {"C": C, "gamma": gamma, "kernel": kernel}
    # Perform Grid Search to find the best parameters
    grid_search = GridSearchCV(SVC(), param_grid=params, cv=cv)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and use them to create the final SVC model
    best_params = grid_search.best_params_
    best_svc = SVC(**best_params)
    best_svc.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_svc.predict(X_test)
    return modeleval(y_test, y_predicted, beg_time, best_params, args, model=model, verbose=verbose)
