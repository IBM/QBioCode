# ====== Base class imports ======

import time
import numpy as np

# ====== Scikit-learn imports ======

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import modeleval

# ====== Begin functions ======

def compute_mlp(X_train, X_test, y_train, y_test, args, verbose=False, model='Multi-layer Perceptron', data_key = '',
                hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', 
                learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=10000, shuffle=True, 
                random_state=None, tol=0.0001, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                early_stopping=False, validation_fraction=0.1, beta_1=0.9, 
                beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000):
        
    """
        This function generates a model using a Multi-layer Perceptron (mlp), a neural network, method as implemented in scikit-learn 
        (https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html). It takes in parameter
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
            hidden_layer_sizes (tuple): The ith element represents the number of neurons in the ith hidden layer.
            activation (str): Activation function for the hidden layer.
            solver (str): The solver for weight optimization.
            alpha (float): L2 penalty (regularization term) parameter.
            batch_size (int or str): Size of minibatches for stochastic optimizers.
            learning_rate (str): Learning rate schedule for weight updates.
            learning_rate_init (float): Initial learning rate used.
            power_t (float): The exponent for inverse scaling learning rate.
            max_iter (int): Maximum number of iterations.
            shuffle (bool): Whether to shuffle samples in each iteration.
            random_state (int or None): Random seed for reproducibility.
            tol (float): Tolerance for stopping criteria.
            warm_start (bool): If True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.
            momentum (float): Momentum for gradient descent update.
            nesterovs_momentum (bool): Whether to use Nesterov's momentum or not.
            early_stopping (bool): Whether to use early stopping to terminate training when validation score is not improving.
            validation_fraction (float): Proportion of training data to set aside as validation set for early stopping.
            beta_1, beta_2, epsilon: Parameters for Adam optimizer.
            n_iter_no_change: Number of iterations with no improvement after which training will be stopped.
            max_fun: Maximum number of function evaluations.

        Returns:
                modeleval (dict): A dictionary containing the evaluation metrics of the model on the test dataset, including accuracy, AUC, F1 score,
                          and the time taken to train and validate the model, along with the model parameters.   
    """  
    
    beg_time = time.time()
    mlp = OneVsOneClassifier(MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, 
                                           batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, 
                                           power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol, 
                                           warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum, 
                                           early_stopping=early_stopping, validation_fraction=validation_fraction, beta_1=beta_1, 
                                           beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change, max_fun=max_fun))
    # Fit the training datset
    model_fit = mlp.fit(X_train, y_train)
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = mlp.predict(X_test) 
    return(modeleval(y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose))

def compute_mlp_opt(X_train, X_test, y_train, y_test, args, verbose=False, cv=5, model='Multi-layer Perceptron',
                    hidden_layer_sizes= [], activation = [], max_iter= [],
                    solver = [], alpha = [], learning_rate= []):
        
    """
        This function also generates a model using a Multi-layer Perceptron (mlp), a neural network, as implemented in scikit-learn 
        (https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html). The difference here is that
        this function runs a grid search. The range of the grid search for each parameter is specified in the config.yaml file. The
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
                cv (int): Number of cross-validation folds.
                model (str): Name of the model being used.
                hidden_layer_sizes (tuple or list): The ith element represents the number of neurons in the ith hidden layer.
                activation (str or list): Activation function for the hidden layer.
                max_iter (int or list): Maximum number of iterations.
                solver (str or list): The solver for weight optimization.
                alpha (float or list): L2 penalty (regularization term) parameter.
                learning_rate (str or list): Learning rate schedule for weight updates.
        Returns:
                modeleval (dict): A dictionary containing the evaluation metrics of the model on the test dataset, including accuracy, AUC, F1 score,
                          and the time taken to train and validate the model, along with the best parameters found during grid search.
    """   
    
    beg_time = time.time()
    params={'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation, 
            'max_iter': max_iter,
            'solver': solver,
            'alpha': alpha,
            'learning_rate': learning_rate,
            }
    
    # Pemlporm Grid Search to find the best parameters
    grid_search = GridSearchCV(MLPClassifier(), param_grid=params, cv=cv)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and use them to create the final model
    best_params = grid_search.best_params_
    best_mlp = MLPClassifier(**best_params)
    best_mlp.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_mlp.predict(X_test)
    return(modeleval(y_test, y_predicted, beg_time, best_params, args, model=model, verbose=verbose))
