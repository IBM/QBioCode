"""
Evaluation Module for QBioCode
==============================

This module provides comprehensive evaluation tools for machine learning
models and datasets. It includes functions for model performance assessment,
dataset complexity analysis, and automated model execution.

Available Functions
------------------
- modeleval: Evaluate model performance with multiple metrics
- evaluation_metrics: Calculate accuracy and Brier score from predictions
- evaluate: Comprehensive dataset complexity evaluation
- model_run: Automated model training and evaluation pipeline

Usage
-----
>>> from qbiocode.evaluation import modeleval, evaluate
>>> # Evaluate model performance
>>> metrics = modeleval(y_true, y_pred, y_proba)
>>> # Evaluate dataset complexity
>>> complexity_metrics = evaluate(X, y)
"""

from .model_evaluation import modeleval, evaluation_metrics
from .dataset_evaluation import evaluate
from .model_run import model_run

__all__ = [
    'modeleval',
    'evaluation_metrics',
    'evaluate',
    'model_run',
]
