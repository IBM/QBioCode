"""
Quantum Data Encoding Utilities
================================

This module provides utility functions for encoding classical data into
quantum states, including normalization, label encoding, and training
set preparation for quantum machine learning algorithms.

These functions are generic and can be used across different quantum
algorithms, not just ensemble learning.
"""

import numpy as np
from typing import List, Tuple


def normalize_data(x: np.ndarray, C: float = 1.0) -> List[complex]:
    """
    Normalize data vector for quantum state encoding.
    
    Normalizes a classical data vector to unit L2 norm and converts to
    complex amplitudes suitable for quantum state initialization.
    
    Parameters
    ----------
    x : np.ndarray
        Classical data vector to normalize
    C : float, optional
        Scaling constant (default: 1.0)
    
    Returns
    -------
    List[complex]
        Normalized vector as list of complex numbers
    
    Examples
    --------
    >>> x = np.array([3.0, 4.0])
    >>> x_norm = normalize_data(x)
    >>> print([abs(xi) for xi in x_norm])
    [0.6, 0.8]
    >>> print(sum([abs(xi)**2 for xi in x_norm]))
    1.0
    """
    M = np.sum(x**2)
    x_normed = [complex(i / np.sqrt(M * C), 0) for i in x]
    return x_normed


def label_to_array(y: np.ndarray) -> np.ndarray:
    """
    Convert binary labels to one-hot encoded arrays.
    
    Transforms binary classification labels (0 or 1) into one-hot encoded
    format required by quantum circuits. Label 0 becomes [1, 0] and label
    1 becomes [0, 1].
    
    Parameters
    ----------
    y : np.ndarray
        Binary labels (0 or 1)
    
    Returns
    -------
    np.ndarray
        One-hot encoded labels, shape (n_samples, 2)
    
    Examples
    --------
    >>> y = np.array([0, 1, 0])
    >>> label_to_array(y)
    array([[1, 0],
           [0, 1],
           [1, 0]])
    """
    Y = []
    for el in y:
        if el == 0:
            Y.append([1, 0])
        else:
            Y.append([0, 1])
    return np.asarray(Y)


def prepare_training_set(X: np.ndarray, y: np.ndarray, 
                        n: int = 4, seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select and prepare balanced training subset for quantum ensemble.
    
    Creates a balanced training set by selecting equal numbers of samples
    from each class and normalizing them for quantum encoding.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Training feature data
    y : np.ndarray, shape (n_samples,)
        Training labels (binary: 0 or 1)
    n : int, optional
        Total number of training samples to select (must be even, default: 4)
    seed : int, optional
        Random seed for reproducibility (default: 123)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X_data : Normalized training samples, shape (n, n_features)
        Y_data : One-hot encoded labels, shape (n, 2)
    
    Examples
    --------
    >>> X = np.random.rand(20, 4)
    >>> y = np.array([0]*10 + [1]*10)
    >>> X_data, Y_data = prepare_training_set(X, y, n=4, seed=42)
    >>> print(X_data.shape, Y_data.shape)
    (4, 4) (4, 2)
    """
    np.random.seed(seed)
    
    # Select balanced samples from each class
    ix_y1 = np.random.choice(np.where(y == 1)[0], int(n / 2), replace=False)
    ix_y0 = np.random.choice(np.where(y == 0)[0], int(n / 2), replace=False)
    
    X_selected = np.concatenate([X[ix_y1], X[ix_y0]])
    Y_data = label_to_array(np.concatenate([y[ix_y1], y[ix_y0]]))
    
    # Normalize each sample
    X_data = np.array([normalize_data(x) for x in X_selected])
    
    return X_data, Y_data


