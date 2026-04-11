"""
Generate synthetic n-dimensional spiral datasets for multi-class classification.

This module creates multiple configurations of high-dimensional spiral datasets
with varying sample sizes, noise levels, and dimensionality, useful for testing
machine learning algorithms on complex non-linearly separable patterns.
"""

import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_spirals(n_samples=5000, n_classes=2, noise=0.3, dim=3):
    """
    Generate an n-dimensional dataset of intertwined spirals.

    Creates spiral patterns in n-dimensional space where each class forms
    a distinct spiral arm. Supports dimensions 3, 6, 9, and 12.

    Parameters
    ----------
    n_samples : int, default=5000
        Total number of samples to generate.
    n_classes : int, default=2
        Number of spiral arms (classes).
    noise : float, default=0.3
        Standard deviation of Gaussian noise added to each dimension.
    dim : int, default=3
        Dimensionality of the output space (must be 3, 6, 9, or 12).

    Returns
    -------
    X : ndarray of shape (n_samples, dim)
        Generated spiral data points.
    y : ndarray of shape (n_samples,)
        Class labels for each sample.
    """

    X = []
    y = []

    for i in range(n_classes):
        t = np.linspace(0, 4 * np.pi, n_samples // n_classes)
        x = t * np.cos(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
        y_ = t * np.sin(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
        z = t + np.random.normal(0, noise, n_samples // n_classes)
        if dim == 3:
            X.append(
                np.column_stack([x, y_, z])
            )  # any new dimensions need to be added to this list

        # to add more dimensions, apparently you would just keep adding 't' variable from above, to each new dimension,
        # as seen below. The question is, how can we iteratively do this while maintaining the binary classification
        # that this for loop is creating?
        # nesting a loop iterating over the number of dimensions doesn't really work from what I'm seeing. so far
        # However, manually adding repeats of the same 3Ds, does work, as seen below -- is this correct?

        # for j in range(dim-3): # for anything above the first 3D
        if dim == 6:
            new_d1 = t * np.cos(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d2 = t * np.sin(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d3 = t + np.random.normal(0, noise, n_samples // n_classes)
            X.append(
                np.column_stack([x, y_, z, new_d1, new_d2, new_d3])
            )  # any new dimensions need to be added to this list
        if dim == 9:
            new_d1 = t * np.cos(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d2 = t * np.sin(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d3 = t + np.random.normal(0, noise, n_samples // n_classes)
            new_d4 = t * np.cos(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d5 = t * np.sin(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d6 = t + np.random.normal(0, noise, n_samples // n_classes)
            X.append(
                np.column_stack([x, y_, z, new_d1, new_d2, new_d3, new_d4, new_d5, new_d6])
            )  # any new dimensions need to be added to this list
        if dim == 12:
            new_d1 = t * np.cos(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d2 = t * np.sin(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d3 = t + np.random.normal(0, noise, n_samples // n_classes)
            new_d4 = t * np.cos(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d5 = t * np.sin(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d6 = t + np.random.normal(0, noise, n_samples // n_classes)
            new_d7 = t * np.cos(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d8 = t * np.sin(t + i * np.pi) + np.random.normal(0, noise, n_samples // n_classes)
            new_d9 = t + np.random.normal(0, noise, n_samples // n_classes)
            X.append(
                np.column_stack(
                    [
                        x,
                        y_,
                        z,
                        new_d1,
                        new_d2,
                        new_d3,
                        new_d4,
                        new_d5,
                        new_d6,
                        new_d7,
                        new_d8,
                        new_d9,
                    ]
                )
            )
        y.extend([i] * (n_samples // n_classes))

    return np.vstack(X), np.array(y)


# parameters to vary across the configurations
N_SAMPLES = list(range(100, 300, 50))
N_CLASSES = [2]
NOISE = [0.3, 0.6, 0.9]
DIM = [3, 6, 9, 12]


def generate_spirals_datasets(
    n_s=N_SAMPLES,
    n_c=N_CLASSES,
    n_n=NOISE,
    n_d=DIM,
    save_path=None,
    random_state=42,
):
    """
    Generate multiple n-dimensional spiral datasets with varying parameters.

    Creates a series of high-dimensional datasets where samples form intertwined
    spiral patterns, providing challenging non-linearly separable multi-class
    classification problems. Each configuration varies the number of samples,
    classes, noise level, and dimensionality.

    Parameters
    ----------
    n_s : list of int, default=range(100, 300, 50)
        List of sample sizes to generate for each configuration.
    n_c : list of int, default=[2]
        List of class counts (number of spiral arms).
    n_n : list of float, default=[0.3, 0.6, 0.9]
        List of noise standard deviations to apply to the data.
    n_d : list of int, default=[3, 6, 9, 12]
        List of dimensionalities (must be 3, 6, 9, or 12).
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
    - Each dataset is saved as 'spirals_data-{i}.csv' where i is the configuration number
    - Configuration parameters are saved in 'dataset_config.json'
    - The last column 'class' contains class labels
    - Spiral patterns become increasingly complex in higher dimensions

    Examples
    --------
    >>> from qbiocode.data_generation import generate_spirals_datasets
    >>> generate_spirals_datasets(n_s=[200], n_c=[2], n_n=[0.3], n_d=[3], save_path='data')
    Generating spirals dataset...
    """
    print("Generating spirals dataset...")

    np.random.seed(random_state)

    if save_path is None:
        save_path = "spirals_data"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # enumerate all possible combinations of parameters based on ranges above
    configurations = list(itertools.product(*[n_s, n_c, n_n, n_d]))
    # print(configurations)
    # print(len(configurations))
    count_configs = 1

    dataset_config = {}

    # populate all the configs with the corresponding argument values
    for n_s, n_c, n_n, n_d in configurations:
        config = "samples={}, classes={}, noise={}, dimensions={}".format(n_s, n_c, n_n, n_d)
        # print(count_configs)

        X, y = make_spirals(n_samples=n_s, n_classes=n_c, noise=n_n, dim=n_d)
        # print("Configuration {}/{}: {}".format(count_configs, len(configurations), config))
        dataset = pd.DataFrame(X)
        dataset["class"] = y
        with open(os.path.join(save_path, "dataset_config.json"), "w") as outfile:
            dataset_config.update(
                {
                    "spirals_data-{}.csv".format(count_configs): {
                        "n_samples": n_s,
                        "noise": n_n,
                        "dimensions": n_d,
                    }
                }
            )
            json.dump(dataset_config, outfile, indent=4)
        new_dataset = dataset.to_csv(
            os.path.join(save_path, "spirals_data-{}.csv".format(count_configs)), index=False
        )
        count_configs += 1

        # plot the last 3 dimensions in each case
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(X[:, n_d-3], X[:, n_d-2],X[:, n_d-1], c=y, cmap='viridis')
        # plt.savefig('spirals_data/spirals_data-{}.png'.format(count_configs))
        # print(X.shape)
        # print(y.shape)
    return
