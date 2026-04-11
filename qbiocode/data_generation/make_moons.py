"""
Generate synthetic two-moons datasets for binary classification tasks.

This module creates multiple configurations of 2D two-moons datasets with
varying sample sizes and noise levels, useful for testing machine learning
algorithms on non-linearly separable data with interleaving classes.
"""

import itertools
import json
import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons

# parameters to vary across the configurations
N_SAMPLES = list(range(100, 300, 20))
NOISE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def generate_moons_datasets(
    n_samples=N_SAMPLES,
    noise=NOISE,
    save_path=None,
    random_state=42,
):
    """
    Generate multiple two-moons datasets with varying parameters.

    Creates a series of 2D datasets where samples form two interleaving half-circles
    (moons), providing a challenging non-linearly separable binary classification problem.
    Each configuration varies the number of samples and noise level.

    Parameters
    ----------
    n_samples : list of int, default=range(100, 300, 20)
        List of sample sizes to generate for each configuration.
    noise : list of float, default=[0.1, 0.2, ..., 0.9]
        List of noise standard deviations to apply to the data.
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
    - Each dataset is saved as 'moons_data-{i}.csv' where i is the configuration number
    - Configuration parameters are saved in 'dataset_config.json'
    - The last column 'class' contains binary labels (0 or 1)
    - Two-moons datasets are commonly used to evaluate algorithms on interleaving patterns

    Examples
    --------
    >>> from qbiocode.data_generation import generate_moons_datasets
    >>> generate_moons_datasets(n_samples=[100, 200], noise=[0.1, 0.3], save_path='data')
    Generating moons dataset...
    """
    print("Generating moons dataset...")

    np.random.seed(random_state)

    if save_path is None:
        save_path = "moons_data"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # enumerate all possible combinations of parameters based on ranges above
    configurations = list(itertools.product(*[n_samples, noise]))
    # print(configurations)
    # print(len(configurations))
    count_configs = 1

    dataset_config = {}

    # populate all the configs with the corresponding argument values
    for n_s, n_n in configurations:
        config = "n_samples={}, noise={}".format(
            n_s,
            n_n,
        )
        # print(count_configs)

        # iteratively run the function for each combination of arguments
        X, y = make_moons(
            n_samples=n_s,
            noise=n_n,
            random_state=random_state,
        )
        # print("Configuration {}/{}: {}".format(count_configs, len(configurations), config))
        dataset = pd.DataFrame(X)
        dataset["class"] = y
        with open(os.path.join(save_path, "dataset_config.json"), "w") as outfile:
            dataset_config.update(
                {"moons_data-{}.csv".format(count_configs): {"n_samples": n_s, "noise": n_n}}
            )
            json.dump(dataset_config, outfile, indent=4)
        new_dataset = dataset.to_csv(
            os.path.join(save_path, "moons_data-{}.csv".format(count_configs)), index=False
        )
        count_configs += 1
        # print(X.shape)
        # print(y.shape)
    return
