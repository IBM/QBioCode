# ====== Base class imports ======

import time

# ====== Scikit-learn imports ======

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import modeleval

def compute_nb(X_train, X_test, y_train, y_test, args, verbose=False, model='Naive Bayes',  data_key = '', var_smoothing=1e-09):
    
    """This function generates a model using a Random Forest (rf) Classifier method as implemented in scikit-learn 
    (https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.GaussianNB.html). It takes in parameter
    arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed.
    The model is trained on the training dataset and validated on the test dataset. The function returns the evaluation of the model 
    on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.
    
    Args:
        X_train (numpy.ndarray): Training features.
        X_test (numpy.ndarray): Test features.
        y_train (numpy.ndarray): Training labels.
        y_test (numpy.ndarray): Test labels.
        args (dict): Additional arguments, such as config parameters.
        verbose (bool): If True, prints additional information during execution.
        model (str): Name of the model being used.
        data_key (str): Key for the dataset, if applicable.
        var_smoothing (float): Portion of the largest variance of all features added to variances for calculation stability.
    Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model on the test dataset, including accuracy, AUC, F1 score,
                          and the time taken to train and validate the model, along with the model parameters. 
    """ 
    
    beg_time = time.time()
    nb = OneVsOneClassifier(GaussianNB(var_smoothing=var_smoothing))
    # Fit the training datset
    model_fit = nb.fit(X_train, y_train)
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = nb.predict(X_test)
    return(modeleval(y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose))

def compute_nb_opt(X_train, X_test, y_train, y_test, args, verbose=False, model='Naive Bayes', cv=5, 
                   var_smoothing = [1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02]):
    
    """ This function generates a model using a Random Forest (rf) Classifier method as implemented in scikit-learn 
    (https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.GaussianNB.html). It takes in parameter
    arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed. The
    combination of parameters that led to the best performance is saved and returned as best_params, which can then be used on similar
    datasets, without having to run the grid search.  The model is trained on the training dataset and validated on the test dataset. The function returns the evaluation of the model 
    on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model across the grid search.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.
    Args:
        X_train (numpy.ndarray): Training features.
        X_test (numpy.ndarray): Test features.
        y_train (numpy.ndarray): Training labels.
        y_test (numpy.ndarray): Test labels.
        args (dict): Additional arguments, such as config parameters.
        verbose (bool): If True, prints additional information during execution.
        model (str): Name of the model being used.
        cv (int): Number of cross-validation folds for grid search.
        var_smoothing (list): List of values for the var_smoothing parameter to be tested in grid search.
    Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model on the test dataset, including accuracy, AUC, F1 score,
                          and the time taken to train and validate the model, along with the best parameters found during grid search.
    """ 
    
    beg_time = time.time()
    params={'var_smoothing': var_smoothing}
    # Perform Grid Search to find the best parameters
    grid_search = GridSearchCV(GaussianNB(), param_grid=params, cv=cv)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and use them to create the final SVC model
    best_params = grid_search.best_params_
    best_nb = GaussianNB(**best_params)
    best_nb.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_nb.predict(X_test)
    return(modeleval(y_test, y_predicted, beg_time, best_params, args, model=model, verbose=verbose))