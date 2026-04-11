"""
Checkpoint and restart utilities for resuming interrupted batch processing jobs.

This module provides functions to identify completed datasets from previous runs,
enabling efficient restart of interrupted batch processing workflows.
"""

import os
from typing import List, Optional


def checkpoint_restart(
    previous_results_dir: str,
    completion_marker: str = "RawDataEvaluation.csv",
    prefix_length: int = 8,
    verbose: bool = False,
) -> List[str]:
    """
    Identify completed datasets from a previous run to enable checkpoint restart.

    This function scans a results directory to find which datasets were fully processed
    in a previous run by checking for the presence of a completion marker file. This
    allows you to resume interrupted batch processing jobs without reprocessing
    completed datasets.

    The function assumes that each dataset has its own subdirectory in the results
    directory, and that a specific file (completion marker) is created when processing
    completes successfully.

    Parameters
    ----------
    previous_results_dir : str
        Path to the directory containing results from the previous (interrupted) run.
        Each subdirectory should correspond to one dataset.
    completion_marker : str, optional
        Name of the file that indicates successful completion of a dataset.
        Default is 'RawDataEvaluation.csv' (used by QProfiler).
    prefix_length : int, optional
        Number of characters to strip from the beginning of directory names to get
        the dataset name. Default is 8 (strips ``dataset_`` prefix used by QProfiler).
        Set to 0 to use the full directory name.
    verbose : bool, optional
        If True, print the list of completed datasets and count. Default is False.

    Returns
    -------
    List[str]
        List of dataset names that were fully processed in the previous run.
        These can be excluded when restarting the batch job.

    Examples
    --------
    Basic usage with QProfiler default settings:

    >>> completed = checkpoint_restart('/path/to/previous_results')
    >>> print(f"Found {len(completed)} completed datasets")

    Resume processing only incomplete datasets:

    >>> import os
    >>> all_datasets = [f for f in os.listdir('/path/to/data') if f.endswith('.csv')]
    >>> completed = checkpoint_restart('/path/to/previous_results')
    >>> remaining = [d for d in all_datasets if d not in completed]
    >>> print(f"Need to process {len(remaining)} more datasets")

    Custom completion marker and no prefix stripping:

    >>> completed = checkpoint_restart(
    ...     '/path/to/results',
    ...     completion_marker='ModelResults.csv',
    ...     prefix_length=0,
    ...     verbose=True
    ... )

    Integration with QProfiler batch processing:

    >>> from qbiocode.utils.dataset_checkpoint import checkpoint_restart
    >>>
    >>> # Get list of completed datasets from previous run
    >>> completed_datasets = checkpoint_restart(
    ...     previous_results_dir='./previous_run_results',
    ...     verbose=True
    ... )
    >>>
    >>> # Get all datasets to process
    >>> all_datasets = [f.replace('.csv', '') for f in os.listdir('./data')
    ...                 if f.endswith('.csv')]
    >>>
    >>> # Filter to only incomplete datasets
    >>> datasets_to_process = [d for d in all_datasets if d not in completed_datasets]
    >>>
    >>> # Run QProfiler only on remaining datasets
    >>> # (use datasets_to_process in your batch processing loop)

    Notes
    -----
    - The function only checks for the presence of the completion marker file,
      not its contents or validity
    - When restarting, you may need to manually combine results from the previous
      and current runs
    - Directory names are expected to have a consistent prefix (e.g., ``dataset_``)
      that can be stripped using the prefix_length parameter
    - Non-directory entries in previous_results_dir are ignored

    See Also
    --------
    qbiocode.evaluation.model_run : Main QProfiler batch processing function
    """
    completed_files = []

    # Validate input directory
    if not os.path.exists(previous_results_dir):
        raise FileNotFoundError(f"Previous results directory not found: {previous_results_dir}")

    if not os.path.isdir(previous_results_dir):
        raise NotADirectoryError(f"Path is not a directory: {previous_results_dir}")

    # Scan for completed datasets
    for entry in os.scandir(previous_results_dir):
        if entry.is_dir():
            # Check if completion marker exists in this dataset's directory
            marker_path = os.path.join(entry.path, completion_marker)
            if os.path.exists(marker_path):
                # Extract dataset name by removing prefix
                if prefix_length > 0 and len(entry.name) > prefix_length:
                    dataset_name = entry.name[prefix_length:]
                else:
                    dataset_name = entry.name
                completed_files.append(dataset_name)

    if verbose:
        print(f"Found {len(completed_files)} completed datasets:")
        for dataset in sorted(completed_files):
            print(f"  - {dataset}")

    return completed_files
