"""
Utilities Module for QBioCode
=============================

This module provides helper functions and utilities for data preprocessing,
model management, IBM Quantum account handling, and result analysis.

Available Functions
------------------
- scaler_fn: Data scaling and normalization
- feature_encoding: Encode features for quantum circuits
- qml_winner: Identify best performing quantum model
- checkpoint_restart: Save and load model checkpoints
- track_progress: Track progress of dataset processing
- combine_results: Combine evaluation results from multiple runs
- find_duplicate_files: Find duplicate entries in datasets
- find_string_in_files: Search for strings in files
- generate_qml_experiment_configs: Generate config files for QML grid search
- get_creds: Get IBM Quantum credentials
- instantiate_runtime_service: Instantiate Qiskit Runtime Service
- get_backend_session: Get backend session for quantum execution
- get_sampler: Get sampler primitive
- get_estimator: Get estimator primitive
- get_ansatz: Get quantum ansatz circuit
- get_feature_map: Get quantum feature map
- get_optimizer: Get classical optimizer
- normalize_data: Normalize data for quantum state encoding
- label_to_array: Convert binary labels to one-hot encoding
- prepare_training_set: Prepare balanced training subset
- retrieve_probabilities: Extract probabilities from measurement counts
- execute_circuit: Execute quantum circuit on Aer simulator

Usage
-----
>>> from qbiocode.utils import scaler_fn, feature_encoding
>>> # Scale data
>>> X_scaled = scaler_fn(X, scaling='StandardScaler')
>>> # Encode features for quantum circuits
>>> X_encoded = feature_encoding(X, feature_encoding='OneHotEncoder')
>>> # Prepare data for quantum ensemble
>>> from qbiocode.utils import normalize_data, prepare_training_set
>>> X_norm = normalize_data(X[0])
>>> X_train, Y_train = prepare_training_set(X, y, n=4, seed=42)
"""

from .helper_fn import scaler_fn, feature_encoding
from .qc_winner_finder import qml_winner
from .dataset_checkpoint import checkpoint_restart
from .combine_evals_results import track_progress, combine_results
from .find_duplicates import find_duplicate_files
from .find_string import find_string_in_files
from .generate_qml_configs import generate_qml_experiment_configs
from .ibm_account import get_creds, instantiate_runtime_service
from .qutils import (
    get_backend_session,
    get_sampler,
    get_estimator,
    get_ansatz,
    get_feature_map,
    get_optimizer,
    retrieve_probabilities,
    execute_circuit,
)
from .data_encoding import (
    normalize_data,
    label_to_array,
    prepare_training_set,
)

__all__ = [
    # Data preprocessing
    'scaler_fn',
    'feature_encoding',
    
    # Model management
    'qml_winner',
    'checkpoint_restart',
    
    # Results management
    'track_progress',
    'combine_results',
    
    # Configuration generation
    'generate_qml_experiment_configs',
    
    # File utilities
    'find_duplicate_files',
    'find_string_in_files',
    
    # IBM Quantum utilities
    'get_creds',
    'instantiate_runtime_service',
    
    # Quantum utilities
    'get_backend_session',
    'get_sampler',
    'get_estimator',
    'get_ansatz',
    'get_feature_map',
    'get_optimizer',
    
    # Ensemble utilities
    'normalize_data',
    'label_to_array',
    'prepare_training_set',
    'retrieve_probabilities',
    'execute_circuit',
]
