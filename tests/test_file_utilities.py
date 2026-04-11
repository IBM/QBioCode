from pathlib import Path

import pytest

from conftest import load_module


find_duplicates = load_module(
    "tests._find_duplicates",
    "qbiocode/utils/find_duplicates.py",
)
find_string = load_module(
    "tests._find_string",
    "qbiocode/utils/find_string.py",
)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def normalize_pairs(pairs):
    return {tuple(sorted(pair)) for pair in pairs}


def test_find_duplicate_files_detects_matches_ignoring_empty_lines(tmp_path):
    write_text(tmp_path / "one.txt", "alpha\n\nbeta\n")
    write_text(tmp_path / "two.txt", "beta\nalpha\n")
    write_text(tmp_path / "three.txt", "alpha\ngamma\n")

    duplicates = find_duplicates.find_duplicate_files(str(tmp_path))

    assert normalize_pairs(duplicates) == {
        tuple(sorted((str(tmp_path / "one.txt"), str(tmp_path / "two.txt"))))
    }


def test_find_duplicate_files_honors_case_sensitivity_setting(tmp_path):
    write_text(tmp_path / "upper.txt", "Alpha\n")
    write_text(tmp_path / "lower.txt", "alpha\n")

    duplicates = find_duplicates.find_duplicate_files(
        str(tmp_path),
        case_sensitive=False,
    )

    assert normalize_pairs(duplicates) == {
        tuple(sorted((str(tmp_path / "upper.txt"), str(tmp_path / "lower.txt"))))
    }


def test_find_duplicate_files_raises_for_missing_directory(tmp_path):
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        find_duplicates.find_duplicate_files(str(missing_dir))


def test_find_string_in_files_returns_matching_lines_and_filters_by_pattern(tmp_path):
    write_text(tmp_path / "config.yaml", "mode: fast\nEmbedding: PCA\n")
    write_text(tmp_path / "notes.txt", "embedding: pca\n")

    results = find_string.find_string_in_files(
        str(tmp_path),
        "embedding: pca",
        file_pattern=".yaml",
        case_sensitive=False,
        return_lines=True,
        verbose=False,
    )

    assert results == {
        str(tmp_path / "config.yaml"): [(2, "Embedding: PCA\n")],
    }


def test_find_string_in_files_raises_for_non_directory(tmp_path):
    file_path = tmp_path / "data.txt"
    write_text(file_path, "content\n")

    with pytest.raises(NotADirectoryError):
        find_string.find_string_in_files(str(file_path), "content")
