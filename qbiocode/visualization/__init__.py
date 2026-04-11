"""
Visualization Module for QBioCode
=================================

This module provides visualization tools for analyzing and presenting
machine learning results, including correlation analysis and performance
comparisons between classical and quantum models.

Available Functions
-------------------
- compute_results_correlation: Compute Spearman correlation between metrics
- plot_results_correlation: Create correlation plots and visualizations

Usage
-----
>>> from qbiocode.visualization import plot_results_correlation
>>> # Create correlation plots
>>> plot_results_correlation(results_df, output_dir='plots/')
"""

from .visualize_correlation import compute_results_correlation, plot_results_correlation

__all__ = [
    "compute_results_correlation",
    "plot_results_correlation",
]
