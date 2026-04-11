"""
Utilities for tracking progress and combining results from interrupted jobs.

This module provides functions to help manage and combine results when
computational jobs are interrupted and need to be restarted. These are
generic utilities that can be used with any pipeline that produces CSV
output files in subdirectories.
"""

import os
from typing import List, Optional, Tuple

import pandas as pd


def track_progress(
    input_dataset_dir: str,
    current_results_dir: str,
    completion_marker: str = "RawDataEvaluation.csv",
    prefix_length: int = 8,
    input_extension: str = "csv",
    verbose: bool = True,
) -> Tuple[List[str], int, int]:
    """
    Track progress of a computational job by checking for completed datasets.

    This function scans the results directory for completed datasets (identified
    by the presence of a specific marker file) and compares against the total
    number of input datasets to determine how many remain to be processed.

    Parameters
    ----------
    input_dataset_dir : str
        Path to the directory containing input datasets.
    current_results_dir : str
        Path to the directory containing outputs of the current job.
    completion_marker : str, optional
        Name of the file that indicates a dataset has been fully processed.
        Default is 'RawDataEvaluation.csv'.
    prefix_length : int, optional
        Number of characters to skip from the beginning of directory names
        when extracting dataset identifiers. Default is 8 (e.g., skips ``dataset_``
        prefix).
    input_extension : str, optional
        File extension of input datasets (without dot). Default is 'csv'.
    verbose : bool, optional
        If True, prints progress information. Default is True.

    Returns
    -------
    completed_datasets : List[str]
        List of dataset identifiers that have been completed.
    num_completed : int
        Number of completed datasets.
    num_remaining : int
        Number of datasets remaining to be processed.

    Examples
    --------
    >>> from qbiocode.utils import track_progress
    >>> completed, done, remaining = track_progress(
    ...     input_dataset_dir='data/inputs',
    ...     current_results_dir='results/run1'
    ... )
    The completed datasets are: ['dataset1', 'dataset2']
    You have finished running program on 2 out of a total of 10 input datasets.
    You have 8 input datasets left before program finishes.

    >>> # Custom completion marker
    >>> completed, done, remaining = track_progress(
    ...     input_dataset_dir='data/inputs',
    ...     current_results_dir='results/run1',
    ...     completion_marker='final_output.csv',
    ...     prefix_length=0  # No prefix to skip
    ... )
    """
    completed_files = []

    # Scan results directory for completed datasets
    for entry in os.scandir(current_results_dir):
        if entry.is_dir():
            for file in os.listdir(entry):
                if file == completion_marker:
                    # Extract dataset identifier by skipping prefix
                    dataset_id = entry.name[prefix_length:] if prefix_length > 0 else entry.name
                    completed_files.append(dataset_id)

    # Count total input datasets
    num_input_datasets = []
    for file in os.listdir(input_dataset_dir):
        if file.endswith(input_extension):
            num_input_datasets.append(file)

    num_completed = len(completed_files)
    num_total = len(num_input_datasets)
    num_remaining = num_total - num_completed

    if verbose:
        print(f"The completed datasets are: {completed_files}")
        print(
            f"You have finished running program on {num_completed} out of a total of {num_total} input datasets."
        )
        print(f"You have {num_remaining} input datasets left before program finishes.")

    return completed_files, num_completed, num_remaining


def combine_results(
    prev_results_dir: str,
    recent_results_dir: str,
    eval_file_prefix: str = "Raw",
    results_file_prefix: str = "Model",
    output_eval_file: str = "RawDataEvaluation_Combined.csv",
    output_results_file: str = "ModelResults_Combined.csv",
    save_intermediate: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine results from interrupted and resumed computational jobs.

    This function merges CSV files from a previous (interrupted) job run with
    files from a recent (resumed) job run. It's useful when a long-running
    computational job needs to be restarted and you want to combine all results.

    Parameters
    ----------
    prev_results_dir : str
        Path to the directory where the previous job stopped prematurely.
        Should contain subdirectories with individual result files.
    recent_results_dir : str
        Path to the directory where the job was resumed and ran to completion.
        Should contain combined result files.
    eval_file_prefix : str, optional
        Prefix of evaluation/assessment files to combine. Default is 'Raw'.
    results_file_prefix : str, optional
        Prefix of model results files to combine. Default is 'Model'.
    output_eval_file : str, optional
        Name of the combined evaluation output file.
        Default is 'RawDataEvaluation_Combined.csv'.
    output_results_file : str, optional
        Name of the combined results output file.
        Default is 'ModelResults_Combined.csv'.
    save_intermediate : bool, optional
        If True, saves intermediate combined files from previous run.
        Default is True.
    verbose : bool, optional
        If True, prints shape information during processing. Default is True.

    Returns
    -------
    combined_eval_df : pd.DataFrame
        Combined dataframe of all evaluation/assessment data.
    combined_results_df : pd.DataFrame
        Combined dataframe of all model results.

    Examples
    --------
    >>> from qbiocode.utils import combine_results
    >>> eval_df, results_df = combine_results(
    ...     prev_results_dir='results/run1_interrupted',
    ...     recent_results_dir='results/run2_resumed'
    ... )
    >>> print(f"Combined {len(eval_df)} evaluation records")
    >>> print(f"Combined {len(results_df)} result records")

    >>> # Custom file prefixes and output names
    >>> eval_df, results_df = combine_results(
    ...     prev_results_dir='results/old',
    ...     recent_results_dir='results/new',
    ...     eval_file_prefix='Evaluation',
    ...     results_file_prefix='Results',
    ...     output_eval_file='AllEvaluations.csv',
    ...     output_results_file='AllResults.csv'
    ... )

    Notes
    -----
    The function expects:
    - prev_results_dir to contain subdirectories, each with individual CSV files
    - recent_results_dir to contain combined CSV files at the top level
    - Files are identified by their prefix (eval_file_prefix, results_file_prefix)
    """
    # Initialize lists for collecting dataframes
    eval_dfs = []
    previous_combined_eval_df = []
    results_dfs = []
    previous_combined_result_df = []

    # Collect all individual CSV files from previous run subdirectories
    for entry in os.scandir(prev_results_dir):
        if entry.is_dir():
            for file in os.listdir(entry):
                if file.startswith(eval_file_prefix):
                    eval_csv_files = os.path.join(entry, file)
                    eval_dfs.append(eval_csv_files)
                if file.startswith(results_file_prefix):
                    results_csv_files = os.path.join(entry, file)
                    results_dfs.append(results_csv_files)

    # Read and collect all previous evaluation dataframes
    for evalfile in eval_dfs:
        df1 = pd.read_csv(evalfile)
        previous_combined_eval_df.append(df1)

    # Read and collect all previous results dataframes
    for resultsfile in results_dfs:
        df2 = pd.read_csv(resultsfile)
        previous_combined_result_df.append(df2)

    # Concatenate all previous dataframes
    concat_previous_eval_df = pd.concat(previous_combined_eval_df, ignore_index=True)
    concat_previous_result_df = pd.concat(previous_combined_result_df, ignore_index=True)

    # Optionally save intermediate combined files
    if save_intermediate:
        concat_previous_eval_df.to_csv(
            f"{eval_file_prefix}DataEvaluation_previous.csv", index=False
        )
        concat_previous_result_df.to_csv(f"{results_file_prefix}Results_previous.csv", index=False)

    # Read recent (resumed run) dataframes
    recent_eval_df = None
    recent_results_df = None

    for file in os.listdir(recent_results_dir):
        if file.startswith(eval_file_prefix):
            recent_eval_csv_file = os.path.join(recent_results_dir, file)
            recent_eval_df = pd.read_csv(recent_eval_csv_file, index_col=0)
            recent_eval_df.reset_index(drop=True, inplace=True)
        if file.startswith(results_file_prefix):
            recent_results_csv_file = os.path.join(recent_results_dir, file)
            recent_results_df = pd.read_csv(recent_results_csv_file, index_col=0)
            recent_results_df.reset_index(drop=True, inplace=True)

    # Verify that recent dataframes were found
    if recent_eval_df is None:
        raise FileNotFoundError(
            f"No evaluation file starting with '{eval_file_prefix}' found in {recent_results_dir}"
        )
    if recent_results_df is None:
        raise FileNotFoundError(
            f"No results file starting with '{results_file_prefix}' found in {recent_results_dir}"
        )

    if verbose:
        print(f"Recent evaluation dataframe shape: {recent_eval_df.shape}")
        print(f"Previous evaluation dataframe shape: {concat_previous_eval_df.shape}")
        print(f"Recent results dataframe shape: {recent_results_df.shape}")
        print(f"Previous results dataframe shape: {concat_previous_result_df.shape}")

    # Combine previous and recent dataframes
    new_combined_eval_df = pd.concat([concat_previous_eval_df, recent_eval_df], ignore_index=True)
    new_combined_result_df = pd.concat(
        [concat_previous_result_df, recent_results_df], ignore_index=True
    )

    # Save final combined dataframes
    new_combined_eval_df.to_csv(output_eval_file, index=False)
    new_combined_result_df.to_csv(output_results_file, index=False)

    if verbose:
        print(f"\nCombined evaluation dataframe shape: {new_combined_eval_df.shape}")
        print(f"Combined results dataframe shape: {new_combined_result_df.shape}")
        print(f"\nSaved combined files:")
        print(f"  - {output_eval_file}")
        print(f"  - {output_results_file}")

    return new_combined_eval_df, new_combined_result_df


# Example usage (commented out to prevent execution at import time):
#
# # Track progress of current job
# completed, done, remaining = track_progress(
#     input_dataset_dir='data/inputs',
#     current_results_dir='results/current_run'
# )
#
# # Combine results from interrupted and resumed runs
# eval_df, results_df = combine_results(
#     prev_results_dir='results/run1_interrupted',
#     recent_results_dir='results/run2_resumed'
# )
