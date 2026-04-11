"""
Generate synthetic 3D Swiss roll datasets for manifold learning tasks.

This module creates multiple configurations of 3D Swiss roll datasets with
varying sample sizes, noise levels, and hole configurations, useful for testing
dimensionality reduction and manifold learning algorithms.
"""

import itertools
import json
import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_swiss_roll

# parameters to vary across the configurations
N_SAMPLES = list(range(100, 300, 20))
NOISE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
HOLE = [True, False]


def generate_swiss_roll_datasets(
    n_samples=N_SAMPLES,
    noise=NOISE,
    hole=HOLE,
    save_path=None,
    random_state=42,
):
    """
    Generate multiple 3D Swiss roll datasets with varying parameters.

    Creates a series of 3D datasets where samples lie on a Swiss roll manifold,
    a classic benchmark for manifold learning and dimensionality reduction algorithms.
    Each configuration varies the number of samples, noise level, and whether the
    roll has a hole in the center.

    Parameters
    ----------
    n_samples : list of int, default=range(100, 300, 20)
        List of sample sizes to generate for each configuration.
    noise : list of float, default=[0.1, 0.2, ..., 0.9]
        List of noise standard deviations to apply to the data.
    hole : list of bool, default=[True, False]
        List of boolean values indicating whether to generate Swiss roll with hole.
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
    - Each dataset is saved as 'swiss_roll_data-{i}.csv' where i is the configuration number
    - Configuration parameters are saved in 'dataset_config.json'
    - The last column 'class' contains the position along the manifold (continuous values)
    - Swiss roll is a standard benchmark for testing manifold learning algorithms

    Examples
    --------
    >>> from qbiocode.data_generation import generate_swiss_roll_datasets
    >>> generate_swiss_roll_datasets(n_samples=[200], noise=[0.1], hole=[False], save_path='data')
    Generating swiss roll dataset...
    """
    print("Generating swiss roll dataset...")

    np.random.seed(random_state)

    if save_path is None:
        save_path = "swiss_roll_data"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # enumerate all possible combinations of parameters based on ranges above
    configurations = list(itertools.product(*[n_samples, noise, hole]))
    # print(configurations)
    # print(len(configurations))
    count_configs = 1

    dataset_config = {}

    # populate all the configs with the corresponding argument values
    for n_s, n_n, n_h in configurations:
        config = "n_samples={}, noise={}, hole={}".format(n_s, n_n, n_h)
        # print(count_configs)

        # iteratively run the function for each combination of arguments
        X, y = make_swiss_roll(
            n_samples=n_s,
            noise=n_n,
            hole=n_h,
            random_state=random_state,
        )
        # print("Configuration {}/{}: {}".format(count_configs, len(configurations), config))
        dataset = pd.DataFrame(X)
        dataset["class"] = y
        with open(os.path.join(save_path, "dataset_config.json"), "w") as outfile:
            dataset_config.update(
                {
                    "swiss_roll_data-{}.csv".format(count_configs): {
                        "n_samples": n_s,
                        "noise": n_n,
                        "hole": n_h,
                    }
                }
            )
            json.dump(dataset_config, outfile, indent=4)
        new_dataset = dataset.to_csv(
            os.path.join(save_path, "swiss_roll_data-{}.csv".format(count_configs)), index=False
        )
        count_configs += 1
        # print(X.shape)
        # print(y.shape)
    return
