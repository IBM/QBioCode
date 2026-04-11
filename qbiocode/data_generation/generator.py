"""
Main data generation interface for QBioCode.

This module provides a unified interface to generate various types of synthetic
datasets for machine learning benchmarking and evaluation.
"""

### Imports ###

import qbiocode.data_generation.make_circles as circles
import qbiocode.data_generation.make_class as make_class
import qbiocode.data_generation.make_moons as moons
import qbiocode.data_generation.make_s_curve as s_curve
import qbiocode.data_generation.make_spheres as spheres
import qbiocode.data_generation.make_spirals as spirals
import qbiocode.data_generation.make_swiss_roll as swiss_roll

### Main Function ###

# parameters to vary across the configurations
N_SAMPLES = list(range(100, 300, 20))
NOISE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
HOLE = [True, False]
N_CLASSES = [2]
DIM = [3, 6, 9, 12]
RAD = [3, 6, 9, 12]
N_FEATURES = list(range(10, 60, 20))
N_INFORMATIVE = list(range(2, 8, 4))
N_REDUNDANT = list(range(2, 8, 4))
N_CLUSTERS_PER_CLASS = list(range(1, 2, 3))
WEIGHTS = [[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]


def generate_data(
    type_of_data=None,
    save_path=None,
    n_samples=N_SAMPLES,
    noise=NOISE,
    hole=HOLE,
    n_classes=N_CLASSES,
    dim=DIM,
    rad=RAD,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE,
    n_redundant=N_REDUNDANT,
    n_clusters_per_class=N_CLUSTERS_PER_CLASS,
    weights=WEIGHTS,
    random_state=42,
):
    """
    Generate synthetic datasets for machine learning benchmarking.

    Unified interface to generate various types of synthetic datasets with
    configurable parameters. Each dataset type creates multiple configurations
    by varying the specified parameters.

    Parameters
    ----------
    type_of_data : str
        Type of dataset to generate. Options: 'circles', 'moons', 'classes',
        's_curve', 'spheres', 'spirals', 'swiss_roll'.
    save_path : str
        Directory path where datasets will be saved.
    n_samples : list of int, default=range(100, 300, 20)
        Sample sizes for dataset configurations.
    noise : list of float, default=[0.1, 0.2, ..., 0.9]
        Noise levels to apply.
    hole : list of bool, default=[True, False]
        Whether to include hole (for swiss_roll only).
    n_classes : list of int, default=[2]
        Number of classes (for spirals and classes).
    dim : list of int, default=[3, 6, 9, 12]
        Dimensionalities (for spheres and spirals).
    rad : list of float, default=[3, 6, 9, 12]
        Radii (for spheres only).
    n_features : list of int, default=range(10, 60, 20)
        Feature counts (for classes only).
    n_informative : list of int, default=range(2, 8, 4)
        Informative feature counts (for classes only).
    n_redundant : list of int, default=range(2, 8, 4)
        Redundant feature counts (for classes only).
    n_clusters_per_class : list of int, default=range(1, 2, 3)
        Clusters per class (for classes only).
    weights : list of list of float, default=[[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]
        Class weight distributions (for classes only).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    None
        Saves generated datasets to the specified path.

    Raises
    ------
    ValueError
        If type_of_data is not one of the supported types.

    Examples
    --------
    >>> from qbiocode.data_generation import generate_data
    >>> generate_data(type_of_data='circles', save_path='data/circles')
    Generating circles dataset...
    Dataset generation complete.
    """

    if type_of_data == "circles":
        # Generate circles dataset
        circles.generate_circles_datasets(
            n_samples=n_samples, noise=noise, save_path=save_path, random_state=random_state
        )
    elif type_of_data == "moons":
        # Generate moons dataset
        moons.generate_moons_datasets(
            n_samples=n_samples, noise=noise, save_path=save_path, random_state=random_state
        )
    elif type_of_data == "classes":
        # Generate higher-dimensional classification dataset
        make_class.generate_classification_datasets(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=n_classes,
            n_clusters_per_class=n_clusters_per_class,
            weights=weights,
            save_path=save_path,
            random_state=random_state,
        )
    elif type_of_data == "s_curve":
        # Generate S-curve dataset
        s_curve.generate_s_curve_datasets(
            n_samples=n_samples, noise=noise, save_path=save_path, random_state=random_state
        )
    elif type_of_data == "spheres":
        # Generate spheres dataset
        spheres.generate_spheres_datasets(
            n_s=n_samples, dim=dim, radius=rad, save_path=save_path, random_state=random_state
        )
    elif type_of_data == "spirals":
        # Generate spirals dataset
        spirals.generate_spirals_datasets(
            n_s=n_samples,
            n_c=n_classes,
            n_n=noise,
            n_d=dim,
            save_path=save_path,
            random_state=random_state,
        )
    elif type_of_data == "swiss_roll":
        # Generate Swiss roll dataset
        swiss_roll.generate_swiss_roll_datasets(
            n_samples=n_samples,
            noise=noise,
            hole=hole,
            save_path=save_path,
            random_state=random_state,
        )
    else:
        raise ValueError(
            "Invalid type_of_data. Choose from 'circles', 'moons', 'classes', 's_curve', 'spheres', 'spirals', or 'swiss_roll'."
        )

    print("Dataset generation complete.")
    return
