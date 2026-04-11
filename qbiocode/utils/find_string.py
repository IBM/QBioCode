"""
String search utilities for finding specific content across multiple files.

This module provides functions to search for strings or patterns in files within
a directory, useful for auditing configurations, finding specific parameters, or
validating file contents.
"""

import os
from typing import Dict, List, Optional, Tuple


def find_string_in_files(
    directory: str,
    search_string: str,
    file_pattern: Optional[str] = None,
    case_sensitive: bool = True,
    return_lines: bool = False,
    verbose: bool = True,
) -> Dict[str, List[Tuple[int, str]]]:
    """
    Search for a specific string in all files within a directory.

    Scans files in the specified directory and identifies which files contain
    the search string. Optionally returns the matching lines with line numbers.
    Useful for auditing configurations, finding specific parameters, or
    validating settings across multiple files.

    Parameters
    ----------
    directory : str
        Path to the directory containing files to search.
    search_string : str
        The string to search for in the files.
    file_pattern : str, optional
        File extension or pattern to filter (e.g., '.yaml', '.csv', '.txt').
        If None, all files are searched. Default is None.
    case_sensitive : bool, optional
        If True, search is case-sensitive. Default is True.
    return_lines : bool, optional
        If True, return matching lines with line numbers. Default is False.
    verbose : bool, optional
        If True, print progress and results. Default is True.

    Returns
    -------
    Dict[str, List[Tuple[int, str]]]
        Dictionary mapping file paths to list of (line_number, line_content) tuples
        for files containing the search string. If return_lines is False, the list
        contains empty tuples.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist.
    NotADirectoryError
        If the specified path is not a directory.

    Examples
    --------
    Basic search for a string:

    >>> results = find_string_in_files(
    ...     'configs/',
    ...     'embeddings: none'
    ... )
    >>> print(f"Found in {len(results)} files")

    Search with line numbers returned:

    >>> results = find_string_in_files(
    ...     'configs/qml_gridsearch/',
    ...     'n_qubits: 4',
    ...     file_pattern='.yaml',
    ...     return_lines=True
    ... )
    >>> for filepath, matches in results.items():
    ...     print(f"{filepath}:")
    ...     for line_num, line_content in matches:
    ...         print(f"  Line {line_num}: {line_content.strip()}")

    Case-insensitive search:

    >>> results = find_string_in_files(
    ...     'logs/',
    ...     'error',
    ...     file_pattern='.log',
    ...     case_sensitive=False
    ... )

    Integration with QProfiler workflow:

    >>> # Find all configs using a specific embedding
    >>> config_dir = "configs/experiments/"
    >>> results = find_string_in_files(
    ...     config_dir,
    ...     'embeddings: pca',
    ...     file_pattern='.yaml',
    ...     verbose=True
    ... )
    >>>
    >>> if results:
    ...     print(f"Found {len(results)} configs using PCA embedding")
    ...     for config_file in results.keys():
    ...         print(f"  - {os.path.basename(config_file)}")

    Notes
    -----
    - Only text files are supported; binary files will be skipped
    - Large files may consume significant memory if return_lines=True
    - Symbolic links are followed and treated as regular files
    - Hidden files (starting with '.') are included in search

    See Also
    --------
    find_duplicate_files : Find files with identical content
    checkpoint_restart : Resume interrupted batch processing jobs
    """
    # Validate input directory
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    # Prepare search string for case-insensitive search
    search_str = search_string if case_sensitive else search_string.lower()

    # Results dictionary
    results = {}
    total_files = 0
    files_with_match = 0

    # Scan directory
    for entry in os.scandir(directory):
        if entry.is_file():
            # Apply file pattern filter if specified
            if file_pattern is not None and not entry.name.endswith(file_pattern):
                continue

            total_files += 1

            try:
                with open(entry.path, "r", encoding="utf-8") as f:
                    matches = []
                    for line_num, line in enumerate(f, start=1):
                        # Apply case sensitivity
                        line_to_search = line if case_sensitive else line.lower()

                        if search_str in line_to_search:
                            if return_lines:
                                matches.append((line_num, line))
                            else:
                                matches.append((0, ""))  # Placeholder

                    if matches:
                        results[entry.path] = matches
                        files_with_match += 1

                        if verbose:
                            if return_lines:
                                print(f"\n{entry.path} contains '{search_string}':")
                                for line_num, line_content in matches:
                                    print(f"  Line {line_num}: {line_content.rstrip()}")
                            else:
                                print(f"{entry.path} contains '{search_string}'")

            except (UnicodeDecodeError, PermissionError) as e:
                if verbose:
                    print(f"Warning: Could not read {entry.path}: {e}")
                continue

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"Search Summary:")
        print(f"  Total files scanned: {total_files}")
        print(f"  Files containing '{search_string}': {files_with_match}")
        if file_pattern:
            print(f"  File pattern filter: {file_pattern}")
        print(f"{'='*60}")

    return results
