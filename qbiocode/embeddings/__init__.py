"""
Embeddings Module for QBioCode
==============================

This module provides feature embedding and encoding methods for quantum
machine learning. It includes functions for computing various embeddings
and quantum feature maps.

Available Functions
-------------------
- get_embeddings: Compute embeddings using various methods (PCA, t-SNE, UMAP, etc.)
- pqk: Projected Quantum Kernel embedding

Available Classes
-----------------
- ConvAutoencoder: Convolutional autoencoder for dimensionality reduction

Usage
-----
>>> from qbiocode.embeddings import get_embeddings, pqk
>>> # Compute PCA embedding
>>> X_embedded = get_embeddings(X, method='pca', n_components=2)
>>> # Compute PQK embedding
>>> X_pqk = pqk(X, n_components=4)
"""

from .compute_autoencoder import ConvAutoencoder
from .embed import get_embeddings, pqk

__all__ = [
    "get_embeddings",
    "pqk",
    "ConvAutoencoder",
]
