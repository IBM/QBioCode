"""
File duplicate detection utilities for identifying identical files in directories.

This module provides functions to find duplicate files based on content comparison,
useful for cleaning up redundant configuration files or identifying duplicate datasets.
"""

import itertools
import os
from typing import List, Optional, Tuple


def find_duplicate_files(
    directory: str,
    file_pattern: Optional[str] = None,
    ignore_empty_lines: bool = True,
    case_sensitive: bool = True,
    verbose: bool = False,
) -> List[Tuple[str, str]]:
    """
    Find files with identical content in a directory.

    Scans the specified directory for files and compares their content line by line.
    Identifies files that have identical content, even if they have different names.
    Optionally filters files by pattern and provides various comparison options.

    This is particularly useful for:

    - Finding duplicate configuration files (e.g., YAML, JSON)
    - Identifying redundant experiment configurations
    - Cleaning up duplicate datasets before batch processing
    - Validating file uniqueness in automated workflows

    Parameters
    ----------
    directory : str
        Path to the directory to search for duplicate files.
    file_pattern : str, optional
        File extension or pattern to filter (e.g., '.yaml', '.csv', '.txt').
        If None, all files are compared. Default is None.
    ignore_empty_lines : bool, optional
        If True, empty lines are ignored during comparison. Default is True.
    case_sensitive : bool, optional
        If True, comparison is case-sensitive. Default is True.
    verbose : bool, optional
        If True, print progress information during comparison. Default is False.

    Returns
    -------
    List[Tuple[str, str]]
        List of tuples, where each tuple contains paths of two duplicate files.
        Returns empty list if no duplicates are found.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist.
    NotADirectoryError
        If the specified path is not a directory.
    PermissionError
        If files cannot be read due to permission issues.

    Examples
    --------
    Find all duplicate files in a directory:

    >>> duplicates = find_duplicate_files("configs/")
    >>> if duplicates:
    ...     print(f"Found {len(duplicates)} duplicate pairs")

    Find duplicate YAML configuration files:

    >>> duplicates = find_duplicate_files(
    ...     "configs/qml_gridsearch/",
    ...     file_pattern='.yaml',
    ...     verbose=True
    ... )
    >>> for file1, file2 in duplicates:
    ...     print(f"Duplicate: {file1} == {file2}")

    Case-insensitive comparison:

    >>> duplicates = find_duplicate_files(
    ...     "data/",
    ...     file_pattern='.txt',
    ...     case_sensitive=False
    ... )

    Integration with QProfiler workflow:

    >>> # Check for duplicate configs before batch processing
    >>> config_dir = "configs/experiments/"
    >>> duplicates = find_duplicate_files(config_dir, file_pattern='.yaml')
    >>>
    >>> if duplicates:
    ...     print("Warning: Duplicate configurations found!")
    ...     for f1, f2 in duplicates:
    ...         print(f"  {os.path.basename(f1)} == {os.path.basename(f2)}")
    ...     # Optionally remove duplicates or warn user

    Notes
    -----
    - Files are compared line by line after sorting (order-independent)
    - Binary files are not supported; use for text files only
    - Large files may consume significant memory during comparison
    - Symbolic links are followed and treated as regular files
    - Hidden files (starting with '.') are included in comparison

    See Also
    --------
    find_string_in_files : Search for specific strings across multiple files
    checkpoint_restart : Resume interrupted batch processing jobs
    """
    # Validate input directory
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    # Collect files to compare
    files = []
    for entry in os.scandir(directory):
        if entry.is_file():
            # Apply file pattern filter if specified
            if file_pattern is None or entry.name.endswith(file_pattern):
                files.append(entry.path)

    if verbose:
        print(f"Comparing {len(files)} files in {directory}")
        if file_pattern:
            print(f"Filtering by pattern: {file_pattern}")

    # Find duplicates by comparing all pairs
    duplicates = []
    total_comparisons = len(list(itertools.combinations(files, 2)))

    for idx, (file1, file2) in enumerate(itertools.combinations(files, 2)):
        if verbose and idx % 100 == 0:
            print(f"Progress: {idx}/{total_comparisons} comparisons")

        try:
            # Read and process file contents
            with open(file1, "r", encoding="utf-8") as f1:
                content1 = f1.readlines()
            with open(file2, "r", encoding="utf-8") as f2:
                content2 = f2.readlines()

            # Filter empty lines if requested
            if ignore_empty_lines:
                content1 = [line for line in content1 if line.strip()]
                content2 = [line for line in content2 if line.strip()]

            # Apply case sensitivity
            if not case_sensitive:
                content1 = [line.lower() for line in content1]
                content2 = [line.lower() for line in content2]

            # Sort for order-independent comparison
            content1_sorted = sorted(content1)
            content2_sorted = sorted(content2)

            # Compare contents
            if content1_sorted == content2_sorted:
                duplicates.append((file1, file2))
                if verbose:
                    print(
                        f"  Duplicate found: {os.path.basename(file1)} == {os.path.basename(file2)}"
                    )

        except (UnicodeDecodeError, PermissionError) as e:
            if verbose:
                print(f"  Warning: Could not read {file1} or {file2}: {e}")
            continue

    if verbose:
        print(f"\nFound {len(duplicates)} duplicate file pairs")

    return duplicates
