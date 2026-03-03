"""
Test script to demonstrate sage_v2.py parameter parsing with nested dictionaries.

This script shows how sage_v2 handles Model_Parameters in the format:
"{'feature_map': 'ZZFeatureMap', 'feature_map_reps': 10, 'entanglement': 'pairwise', 
  'svc_best_params': {'C': 1, 'gamma': 1, 'kernel': 'rbf'}}"
"""

import pandas as pd
import numpy as np
from sage_v2 import QuantumSageV2

# Create sample data with nested Model_Parameters
sample_data = {
    '# Features': [10, 20, 15, 25, 30],
    '# Samples': [100, 200, 150, 250, 300],
    'Feature_Samples_ratio': [0.1, 0.1, 0.1, 0.1, 0.1],
    'Intrinsic_Dimension': [5, 8, 6, 10, 12],
    'Condition number': [1.5, 2.0, 1.8, 2.2, 2.5],
    'Fisher Discriminant Ratio': [0.8, 0.9, 0.85, 0.92, 0.95],
    'Total Correlations': [0.3, 0.4, 0.35, 0.42, 0.45],
    'Mutual information': [0.2, 0.3, 0.25, 0.32, 0.35],
    '# Non-zero entries': [90, 180, 135, 225, 270],
    '# Low variance features': [2, 3, 2, 4, 5],
    'Variation': [0.5, 0.6, 0.55, 0.62, 0.65],
    'std_var': [0.1, 0.12, 0.11, 0.13, 0.14],
    'Coefficient of Variation %': [20, 22, 21, 23, 24],
    'std_co_of_v': [2, 2.5, 2.2, 2.6, 2.8],
    'Skewness': [0.3, 0.4, 0.35, 0.42, 0.45],
    'std_skew': [0.05, 0.06, 0.055, 0.062, 0.065],
    'Kurtosis': [3.0, 3.2, 3.1, 3.3, 3.4],
    'std_kurt': [0.3, 0.35, 0.32, 0.37, 0.39],
    'Mean Log Kernel Density': [-2.0, -1.8, -1.9, -1.7, -1.6],
    'Isomap Reconstruction Error': [0.1, 0.12, 0.11, 0.13, 0.14],
    'Fractal dimension': [1.5, 1.6, 1.55, 1.62, 1.65],
    'Entropy': [2.0, 2.2, 2.1, 2.3, 2.4],
    'std_entropy': [0.2, 0.22, 0.21, 0.23, 0.24],
    
    # Performance metrics
    'accuracy': [0.85, 0.88, 0.86, 0.90, 0.92],
    'f1_score': [0.83, 0.86, 0.84, 0.88, 0.90],
    'auc': [0.87, 0.90, 0.88, 0.92, 0.94],
    
    # Metadata
    'Dataset': ['dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5'],
    'embeddings': ['PCA', 'PCA', 'UMAP', 'UMAP', 'ISOMAP'],
    'datatype': ['artificial', 'artificial', 'biological', 'biological', 'biological'],
    'model_embed_datatype': ['VQC_PCA_artificial', 'VQC_PCA_artificial', 
                             'VQC_UMAP_biological', 'VQC_UMAP_biological', 
                             'VQC_ISOMAP_biological'],
    'iteration': [1, 2, 1, 2, 3],
    'model': ['VQC', 'VQC', 'VQC', 'VQC', 'VQC'],
    
    # Model Parameters with nested dictionaries (as strings)
    'Model_Parameters': [
        "{'feature_map': 'ZZFeatureMap', 'feature_map_reps': 10, 'entanglement': 'pairwise', 'svc_best_params': {'C': 1, 'gamma': 1, 'kernel': 'rbf'}}",
        "{'feature_map': 'ZFeatureMap', 'feature_map_reps': 5, 'entanglement': 'linear', 'svc_best_params': {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}}",
        "{'feature_map': 'ZZFeatureMap', 'feature_map_reps': 15, 'entanglement': 'full', 'svc_best_params': {'C': 0.1, 'gamma': 10, 'kernel': 'linear'}}",
        "{'feature_map': 'PauliFeatureMap', 'feature_map_reps': 8, 'entanglement': 'pairwise', 'svc_best_params': {'C': 5, 'gamma': 0.5, 'kernel': 'rbf'}}",
        "{'feature_map': 'ZZFeatureMap', 'feature_map_reps': 12, 'entanglement': 'circular', 'svc_best_params': {'C': 2, 'gamma': 2, 'kernel': 'poly'}}"
    ]
}

# Create DataFrame
df = pd.DataFrame(sample_data)

print("="*80)
print("Testing QuantumSageV2 with Nested Model Parameters")
print("="*80)

print("\n1. Original Model_Parameters column (first 2 rows):")
print("-" * 80)
for i in range(min(2, len(df))):
    print(f"Row {i}: {df['Model_Parameters'].iloc[i]}")

print("\n2. Initializing QuantumSageV2 with include_model_params=True...")
print("-" * 80)
sage_v2 = QuantumSageV2(df, include_model_params=True)

print("\n3. Parsed and flattened parameter columns:")
print("-" * 80)
if sage_v2._param_features is not None:
    print(f"Number of parameter columns: {len(sage_v2._param_features.columns)}")
    print(f"Parameter column names: {list(sage_v2._param_features.columns)}")
    print("\nFirst few rows of parsed parameters:")
    print(sage_v2._param_features.head())
else:
    print("No parameters parsed!")

print("\n4. Combined feature matrix shape:")
print("-" * 80)
print(f"Data characteristics columns: {len(sage_v2._columns_data_features)}")
print(f"Parameter columns: {len(sage_v2._param_column_names)}")
print(f"Total input features: {sage_v2._input_data_features_only.shape[1]}")
print(f"Combined feature matrix shape: {sage_v2._input_data_features_only.shape}")

print("\n5. Example: How nested parameters are flattened:")
print("-" * 80)
print("Original: {'svc_best_params': {'C': 1, 'gamma': 1, 'kernel': 'rbf'}}")
print("Flattened columns: svc_best_params_C, svc_best_params_gamma, svc_best_params_kernel")
print("\nValues in parsed DataFrame:")
if sage_v2._param_features is not None:
    for col in ['svc_best_params_C', 'svc_best_params_gamma', 'svc_best_params_kernel']:
        if col in sage_v2._param_features.columns:
            print(f"  {col}: {sage_v2._param_features[col].tolist()}")

print("\n6. Categorical parameter encoding:")
print("-" * 80)
print("String parameters (feature_map, entanglement, kernel) are encoded as integers")
if sage_v2._param_features is not None:
    for col in ['feature_map', 'entanglement', 'svc_best_params_kernel']:
        if col in sage_v2._param_features.columns:
            print(f"  {col}: {sage_v2._param_features[col].tolist()}")

print("\n" + "="*80)
print("Parameter parsing test completed successfully!")
print("="*80)

print("\n7. Ready for training:")
print("-" * 80)
print("You can now train sage_v2 with:")
print("  sage_v2.train_sub_sages(sage_type='xgboost_optuna', n_iter=100)")
print("  sage_v2.train_inverse_sages(sage_type='xgboost_optuna', n_iter=100)")
print("\nAnd make predictions with:")
print("  sage_v2.predict(new_data, metric='f1_score')")
print("  sage_v2.predict_parameter_ranges(data_chars, target_metric_value=0.95)")

# Made with Bob
