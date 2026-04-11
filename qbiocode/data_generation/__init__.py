"""
Data Generation Module for QBioCode.

This module provides functions to generate synthetic datasets for testing
machine learning algorithms. Each function creates multiple dataset configurations
with varying parameters, useful for benchmarking and evaluation.

Available dataset generators:
- generate_blobs_datasets: Isotropic Gaussian blobs (clusters)
- generate_circles_datasets: 2D concentric circles
- generate_moons_datasets: 2D interleaving half-circles
- generate_classification_datasets: High-dimensional multi-class data
- generate_s_curve_datasets: 3D S-shaped manifold
- generate_spheres_datasets: N-dimensional concentric spheres
- generate_spirals_datasets: N-dimensional intertwined spirals
- generate_swiss_roll_datasets: 3D Swiss roll manifold
"""

from .make_blobs import generate_blobs_datasets, generate_default_blobs_datasets
from .make_circles import generate_circles_datasets
from .make_class import generate_classification_datasets
from .make_moons import generate_moons_datasets
from .make_s_curve import generate_s_curve_datasets
from .make_spheres import generate_spheres_datasets
from .make_spirals import generate_spirals_datasets
from .make_swiss_roll import generate_swiss_roll_datasets

__all__ = [
    "generate_blobs_datasets",
    "generate_default_blobs_datasets",
    "generate_circles_datasets",
    "generate_moons_datasets",
    "generate_classification_datasets",
    "generate_s_curve_datasets",
    "generate_spheres_datasets",
    "generate_spirals_datasets",
    "generate_swiss_roll_datasets",
]

