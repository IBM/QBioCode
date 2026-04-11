"""
Generate synthetic high-dimensional classification datasets.

This module creates multiple configurations of multi-class classification datasets
with varying dimensionality, feature characteristics, and class distributions,
useful for testing machine learning algorithms on high-dimensional data.
"""

import itertools
import json
import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

dataset_config = {}

# parameters to vary across the configurations
N_SAMPLES = list(range(100, 300, 50))
N_FEATURES = list(range(10, 60, 10))
N_INFORMATIVE = list(range(2, 8, 4))
N_REDUNDANT = list(range(2, 8, 4))
N_CLASSES = list(range(2, 4, 6))
N_CLUSTERS_PER_CLASS = list(range(1, 2, 3))
WEIGHTS = [[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]


def generate_classification_datasets(
    n_samples,
    n_features,
    n_informative,
    n_redundant,
    n_classes,
    n_clusters_per_class,
    weights,
    save_path=None,
    random_state=42,
):
    """
    Generate multiple high-dimensional classification datasets with varying parameters.

    Creates a series of synthetic datasets for multi-class classification problems
    with configurable feature characteristics including informative features,
    redundant features, and class distributions.

    Parameters
    ----------
    n_samples : list of int
        List of sample sizes to generate for each configuration.
    n_features : list of int
        List of total feature counts (must be >= n_informative + n_redundant).
    n_informative : list of int
        List of informative feature counts that are useful for prediction.
    n_redundant : list of int
        List of redundant feature counts (linear combinations of informative features).
    n_classes : list of int
        List of class counts for multi-class classification.
    n_clusters_per_class : list of int
        List of cluster counts per class.
    weights : list of list of float
        List of class weight distributions (must sum to 1.0).
    save_path : str, optional
        Directory path where datasets and configuration files will be saved.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    None
        Saves CSV files for each dataset configuration and a JSON file with
        all configuration parameters.

    Notes
    -----
    - Each dataset is saved as 'class_data-{i}.csv' where i is the configuration number
    - Configuration parameters are saved in 'dataset_config.json'
    - The last column 'class' contains class labels
    - Only valid configurations where (n_informative + n_redundant) <= n_features are generated

    Examples
    --------
    >>> from qbiocode.data_generation import generate_classification_datasets
    >>> generate_classification_datasets(
    ...     n_samples=[100], n_features=[20], n_informative=[5],
    ...     n_redundant=[2], n_classes=[2], n_clusters_per_class=[1],
    ...     weights=[[0.5, 0.5]], save_path='data'
    ... )
    Generating classes dataset...
    """
    print("Generating classes dataset...")

    np.random.seed(random_state)

    if save_path is None:
        save_path = "class_data"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # enumerate all possible combinations of parameters based on ranges above
    configurations = list(
        itertools.product(
            *[
                n_samples,
                n_features,
                n_informative,
                n_redundant,
                n_classes,
                n_clusters_per_class,
                weights,
            ]
        )
    )
    count_configs = 1

    # populate all the configs with the corresponding argument values
    for n_s, n_f, n_i, n_r, n_cla, n_clu, weights in configurations:
        if (n_i + n_r) <= n_f:
            config = "n_samples={}, n_features={}, n_informative={}, n_redundant={}, n_classes={}, n_clusters_per_class={}, weights={}".format(
                n_s, n_f, n_i, n_r, n_cla, n_clu, weights
            )
            # print(count_configs)

            # iteratively run the function for each combination of arguments
            X, y = make_classification(
                n_samples=n_s,
                n_features=n_f,
                n_informative=n_i,
                n_redundant=n_r,
                n_classes=n_cla,
                n_clusters_per_class=n_clu,
                weights=weights,
                random_state=random_state,
            )
            # print("Configuration {}/{}: {}".format(count_configs, len(configurations), config))
            dataset = pd.DataFrame(X)
            dataset["class"] = y
            with open(os.path.join(save_path, "dataset_config.json"), "w") as outfile:
                dataset_config.update(
                    {
                        "hd_data-{}.csv".format(count_configs): {
                            "n_samples": n_s,
                            "n_features": n_f,
                            "n_informative": n_i,
                            "n_redundant": n_r,
                            "n_classes": n_cla,
                            "n_clusters_per_class": n_clu,
                            "weights": weights,
                        }
                    }
                )
                json.dump(dataset_config, outfile, indent=4)
            new_dataset = dataset.to_csv(
                os.path.join(save_path, "class_data-{}.csv".format(count_configs)), index=False
            )
            count_configs += 1
            # print(X.shape)
            # print(y.shape)
    return
