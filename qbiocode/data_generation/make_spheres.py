"""
Generate synthetic concentric n-dimensional spheres datasets for binary classification.

This module creates multiple configurations of high-dimensional concentric spheres
datasets with varying sample sizes, dimensionality, and radii, useful for testing
machine learning algorithms on high-dimensional non-linearly separable data.
"""

import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_points_in_nd_sphere(n_s, dim=3, radius=1, thresh=0.9):
    """
    Generate random points within an n-dimensional spherical shell.

    Parameters
    ----------
    n_s : int
        Number of points to generate.
    dim : int, default=3
        Dimensionality of the sphere.
    radius : float, default=1
        Outer radius of the spherical shell.
    thresh : float, default=0.9
        Inner radius threshold as fraction of outer radius (creates shell).

    Returns
    -------
    points : ndarray of shape (n_s, dim)
        Generated points within the spherical shell.
    """
    cnt = 0
    points = []
    while cnt < n_s:
        pnts = np.random.rand(dim) * 2 * radius - radius
        pnts_nrm = np.linalg.norm(pnts)
        if (pnts_nrm <= radius) & (pnts_nrm >= radius * thresh):
            points.append(pnts)
            cnt += 1
    points = np.asarray(points)
    return points


# parameters to vary across the configurations
N_SAMPLES = list(range(100, 300, 25))
DIM = list(range(5, 15, 5))
RAD = list(range(5, 20, 5))


def generate_spheres_datasets(
    n_s=N_SAMPLES,
    dim=DIM,
    radius=RAD,
    save_path=None,
    random_state=42,
):
    """
    Generate multiple concentric n-dimensional spheres datasets with varying parameters.

    Creates a series of high-dimensional datasets where samples form two concentric
    spherical shells, providing a challenging non-linearly separable binary classification
    problem in high dimensions. Each configuration varies the number of samples,
    dimensionality, and sphere radii.

    Parameters
    ----------
    n_s : list of int, default=range(100, 300, 25)
        List of sample sizes per class to generate for each configuration.
    dim : list of int, default=range(5, 15, 5)
        List of dimensionalities for the spheres.
    radius : list of float, default=range(5, 20, 5)
        List of outer sphere radii (inner sphere is 0.5 * outer radius).
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
    - Each dataset is saved as 'spheres_data-{i}.csv' where i is the configuration number
    - Configuration parameters are saved in 'dataset_config.json'
    - The last column 'class' contains binary labels (0 for outer, 1 for inner sphere)
    - Samples are generated in spherical shells (not solid spheres) for better separation

    Examples
    --------
    >>> from qbiocode.data_generation import generate_spheres_datasets
    >>> generate_spheres_datasets(n_s=[100], dim=[5], radius=[10], save_path='data')
    Generating spheres dataset...
    """
    print("Generating spheres dataset...")

    np.random.seed(random_state)

    if save_path is None:
        save_path = "spheres_data"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # enumerate all possible combinations of parameters based on ranges above
    configurations = list(itertools.product(*[n_s, dim, radius]))
    # print(configurations)
    # print(len(configurations))
    count_configs = 1

    dataset_config = {}

    # populate all the configs with the corresponding argument values
    for n_s, n_d, n_r in configurations:
        config = "samples={}, dimensions={}, radius={}".format(n_s, n_d, n_r)
        # print(count_configs)
        radius1 = n_r
        radius2 = radius1 * 0.5
        Xa = generate_points_in_nd_sphere(n_s, dim=n_d, radius=radius1, thresh=0.9)
        Xb = generate_points_in_nd_sphere(n_s, dim=n_d, radius=radius2, thresh=0.9)
        X = np.concatenate((Xa, Xb))
        y = [0] * len(Xa) + [1] * len(Xb)

        # print("Configuration {}/{}: {}".format(count_configs, len(configurations), config))
        X_df = pd.DataFrame(X)
        y_dict = {"class": y}
        y_df = pd.DataFrame(y_dict)
        df = pd.concat([X_df, y_df], axis=1)
        with open(os.path.join(save_path, "dataset_config.json"), "w") as outfile:
            dataset_config.update(
                {
                    "spheres_data-{}.csv".format(count_configs): {
                        "n_samples": n_s,
                        "dimensions": n_d,
                        "radius": n_r,
                    }
                }
            )
            json.dump(dataset_config, outfile, indent=4)
        new_dataset = df.to_csv(
            os.path.join(save_path, "spheres_data-{}.csv".format(count_configs)), index=False
        )
        count_configs += 1

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # # ax.scatter(X[:, 0], X[:, 1],X[:,2], c= y, cmap='viridis')
        # ax.scatter(X[:, n_d-3], X[:, n_d-2],X[:, n_d-1], c=y, cmap='viridis')
        # plt.savefig('spheres_data/spheres_data-{}.png'.format(count_configs))
        # print(X.shape)
        # print(y.shape)
    return
