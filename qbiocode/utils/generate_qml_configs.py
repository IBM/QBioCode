"""
Generate configuration files for quantum machine learning grid search experiments.

This module provides utilities to generate multiple YAML configuration files
for systematic hyperparameter tuning of quantum machine learning models.
"""

import itertools
import os
import re
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import yaml


def generate_qml_experiment_configs(
    template_config_path: str,
    output_dir: str,
    data_dirs: List[str],
    qmethods: Optional[List[str]] = None,
    reps: Optional[List[int]] = None,
    optimizers: Optional[List[str]] = None,
    entanglements: Optional[List[str]] = None,
    feature_maps: Optional[List[str]] = None,
    ansatz_types: Optional[List[str]] = None,
    n_components: Optional[List[int]] = None,
    Cs: Optional[List[float]] = None,
    max_iters: Optional[List[int]] = None,
    embeddings: Optional[List[str]] = None,
    data_sample_fraction: float = 1.0,
    used_files_path: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> Tuple[int, str]:
    """
    Generate YAML configuration files for quantum ML hyperparameter grid search.

    This function creates multiple configuration files by combining different
    hyperparameter values for quantum machine learning models (QNN, VQC, QSVC).
    Each configuration file can be used with QProfiler to run systematic experiments.

    Parameters
    ----------
    template_config_path : str
        Path to the template YAML configuration file.
    output_dir : str
        Directory where generated config files will be saved.
    data_dirs : List[str]
        List of directories containing CSV dataset files.
    qmethods : List[str], optional
        Quantum methods to test. Default: ['qnn', 'vqc', 'qsvc']
    reps : List[int], optional
        Number of repetitions for ansatz layers. Default: [1, 2]
    optimizers : List[str], optional
        Optimizers to use. Default: ['COBYLA', 'SPSA']
    entanglements : List[str], optional
        Entanglement patterns. Default: ['linear', 'full']
    feature_maps : List[str], optional
        Feature map encodings. Default: ['Z', 'ZZ']
    ansatz_types : List[str], optional
        Ansatz types for QNN/VQC. Default: ['amp', 'esu2']
    n_components : List[int], optional
        Number of components for dimensionality reduction. Default: [5, 10]
    Cs : List[float], optional
        Regularization parameters for QSVC. Default: [0.1, 1, 10]
    max_iters : List[int], optional
        Maximum iterations for optimization. Default: [100, 500]
    embeddings : List[str], optional
        Embedding methods. Default: ['none', 'pca', 'lle', 'isomap', 'spectral', 'umap', 'nmf']
    data_sample_fraction : float, optional
        Fraction of data files to use (0.0-1.0). Default: 1.0
    used_files_path : str, optional
        Path to CSV file tracking previously used data files.
    random_seed : int, optional
        Random seed for reproducible file sampling.

    Returns
    -------
    Tuple[int, str]
        Number of configuration files generated and path to used files CSV.

    Examples
    --------
    >>> from qbiocode.utils import generate_qml_experiment_configs
    >>>
    >>> # Generate configs for quantum model grid search
    >>> num_configs, used_files = generate_qml_experiment_configs(
    ...     template_config_path='configs/config.yaml',
    ...     output_dir='configs/qml_gridsearch',
    ...     data_dirs=['data/tutorial_test_data/lower_dim_datasets'],
    ...     qmethods=['qnn', 'vqc'],
    ...     reps=[1, 2],
    ...     n_components=[5, 10],
    ...     data_sample_fraction=0.1  # Use 10% of files for testing
    ... )
    >>> print(f"Generated {num_configs} configuration files")

    Notes
    -----
    - Quantum models (QNN, VQC, QSVC) don't support automated grid search
    - This function generates separate config files for each hyperparameter combination
    - Run QProfiler separately for each generated config file
    - The function automatically handles model-specific constraints:
        * QSVC uses only 'amp' ansatz and 'COBYLA' optimizer
        * QNN/VQC don't use the C parameter
    - Embedding is set to 'none' when n_components >= original feature count

    See Also
    --------
    qbiocode.apps.qprofiler : Main profiling application
    """
    # Set default hyperparameter values
    if qmethods is None:
        qmethods = ["qnn", "vqc", "qsvc"]
    if reps is None:
        reps = [1, 2]
    if optimizers is None:
        optimizers = ["COBYLA", "SPSA"]
    if entanglements is None:
        entanglements = ["linear", "full"]
    if feature_maps is None:
        feature_maps = ["Z", "ZZ"]
    if ansatz_types is None:
        ansatz_types = ["amp", "esu2"]
    if n_components is None:
        n_components = [5, 10]
    if Cs is None:
        Cs = [0.1, 1, 10]
    if max_iters is None:
        max_iters = [100, 500]
    if embeddings is None:
        embeddings = ["none", "pca", "lle", "isomap", "spectral", "umap", "nmf"]

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set up used files tracking
    if used_files_path is None:
        used_files_path = os.path.join(output_dir, "used_data_files.csv")

    # Generate all hyperparameter combinations
    param_grid = [
        qmethods,
        reps,
        optimizers,
        entanglements,
        feature_maps,
        ansatz_types,
        n_components,
        Cs,
        max_iters,
        embeddings,
    ]

    param_combinations = pd.DataFrame(
        list(itertools.product(*(cast(List[Any], values) for values in param_grid))),
        columns=[
            "method",
            "reps",
            "local_optimizer",
            "entanglement",
            "feature_map",
            "ansatz_type",
            "n_components",
            "C",
            "max_iter",
            "embedding",
        ],
    )

    # Apply model-specific constraints
    param_combinations.loc[param_combinations["method"].isin(["qnn", "vqc"]), "C"] = 1
    param_combinations.loc[param_combinations["method"].isin(["qsvc"]), "ansatz_type"] = "amp"
    param_combinations.loc[param_combinations["method"].isin(["qsvc"]), "max_iter"] = 100
    param_combinations.loc[param_combinations["method"].isin(["qsvc"]), "local_optimizer"] = (
        "COBYLA"
    )

    # Remove duplicates and apply filtering rules
    param_combinations = param_combinations.drop_duplicates()
    param_combinations = param_combinations[
        ~((param_combinations["n_components"] >= 10) & (param_combinations["max_iter"] < 500))
    ]
    param_combinations = param_combinations[
        ~((param_combinations["reps"] > 1) & (param_combinations["n_components"] <= 10))
    ]

    # Load template configuration
    with open(template_config_path, "r") as f:
        cfg_template = yaml.safe_load(f)

    # Load or initialize used files list
    if os.path.exists(used_files_path):
        used_files = pd.read_csv(used_files_path).iloc[:, 0].tolist()
    else:
        used_files = []

    # Generate configuration files
    config_idx = 1

    for data_dir in data_dirs:
        # Get all CSV files in directory
        csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        csv_files.sort()

        # Remove previously used files
        csv_files = list(set(csv_files) - set(used_files))

        # Sample files if requested
        if data_sample_fraction < 1.0:
            n_files = max(1, int(len(csv_files) * data_sample_fraction))
            csv_files = list(np.random.choice(csv_files, n_files, replace=False))

        # Update used files list
        used_files.extend(csv_files)

        # Filter parameter combinations based on data type
        param_subset = param_combinations.copy()
        if ("moons" in data_dir) or ("circles" in data_dir):
            param_subset = param_subset[param_subset["embedding"] == "none"]
        else:
            param_subset = param_subset[param_subset["embedding"] != "none"]

        # Generate config for each combination and file
        for _, params in param_subset.iterrows():
            for csv_file in csv_files:
                config_path = os.path.join(output_dir, f"exp_{config_idx}.yaml")
                key = f"{params['method']}_{csv_file.replace('.csv', '')}"

                # Create config from template
                config = cfg_template.copy()
                config["yaml"] = config_path
                config["model"] = [params["method"]]
                config["file_dataset"] = csv_file
                config["folder_path"] = data_dir.replace("data/", "")
                config["hydra"] = config.get("hydra", {})
                config["hydra"]["run"] = config["hydra"].get("run", {})
                config["hydra"]["run"]["dir"] = os.path.join("results", f"qmlgridsearch_{key}")

                # Check if embedding should be 'none' based on feature count
                df = pd.read_csv(os.path.join(data_dir, csv_file))
                orig_features = df.shape[1] - 1  # Subtract label column

                if params["n_components"] >= orig_features:
                    config["embeddings"] = ["none"]
                else:
                    config["embeddings"] = [params["embedding"]]

                config["n_components"] = params["n_components"]

                # Set method-specific parameters
                method_args_key = f"{params['method']}_args"
                if method_args_key not in config:
                    config[method_args_key] = {}

                config[method_args_key]["reps"] = int(params["reps"])
                config[method_args_key]["entanglement"] = params["entanglement"]
                config[method_args_key]["encoding"] = params["feature_map"]

                if params["method"] != "qsvc":
                    config[method_args_key]["ansatz_type"] = params["ansatz_type"]
                    config[method_args_key]["maxiter"] = int(params["max_iter"])
                else:
                    config[method_args_key]["C"] = float(params["C"])
                    config[method_args_key]["local_optimizer"] = params["local_optimizer"]

                # Write configuration file
                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)

                config_idx += 1

    # Save used files list
    pd.Series(used_files).to_csv(used_files_path, index=False, header=["filename"])

    num_configs = config_idx - 1
    return num_configs, used_files_path
