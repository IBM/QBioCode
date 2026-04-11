# Changelog

All notable changes to QBioCode will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [Unreleased]

### Added
- **Data Generation**: New blob dataset generator
  - `generate_blobs_datasets()`: Create isotropic Gaussian blob datasets
  - `generate_default_blobs_datasets()`: Quick generation with default parameters
  - Follows QBioCode data generation patterns
  - Useful for clustering and classification benchmarks

- **Evaluation Metrics**: Added generalized `evaluation_metrics()` function to `qbiocode.evaluation.model_evaluation`
  - Supports multiple metrics: accuracy, brier, f1, precision, recall, auc
  - Configurable via `metrics` parameter (default: ['accuracy', 'brier'])
  - Backward compatible: returns (accuracy, brier) tuple by default
  - Supports both binary and multi-class classification
  - Provides calibration quality assessment via Brier score
  - Handles edge cases (e.g., single class in test set)
  - Now available in main API (previously only in tutorial helpers)

- **Quantum Ensemble Learning**: New unified quantum ensemble classifier
  - `compute_qensemble()`: Quantum ensemble with configurable construction methods
    - Implements quantum ensemble using controlled operations and superposition
    - Two ensemble methods via `ensemble_method` parameter:
      - `"swap"` (default): Fixed controlled-SWAP operations (faster, deterministic)
      - `"random_unitary"`: Haar-random unitaries (more general, potentially better generalization)
    - Three ensemble modes: balanced, unbalanced, and pair_sample
    - Support for configurable ensemble depth (d) and operations per qubit (n_swap)
    - Quantum cosine similarity classifier using SWAP test
  - New utility functions in `qbiocode.utils`:
    - `normalize_data()`: Normalize data for quantum state encoding
    - `label_to_array()`: Convert binary labels to one-hot encoding
    - `prepare_training_set()`: Prepare balanced training subsets
    - `retrieve_probabilities()`: Extract probabilities from measurement counts (generic quantum utility)
    - `execute_circuit()`: General-purpose Aer simulator execution (reusable across quantum algorithms)
  - Based on Macaluso et al., "A variational algorithm for quantum ensemble learning" (2023)
  - Integrated from tutorial/QEnsemble with full API compatibility
  - Code organization: Extracted reusable functions to utils for broader applicability

- **Testing Infrastructure**: Comprehensive test suite for core functionality
  - `tests/test_data_generation.py`: Tests for data generation utilities
  - `tests/test_file_utilities.py`: Tests for file operations and utilities
  - `tests/test_generator_dispatch.py`: Tests for generator dispatch logic
  - `tests/conftest.py`: Pytest configuration and fixtures
  - Test coverage for utility modules and data generation helpers
  
- **Code Quality Tools**: Enhanced development tooling
  - `isort` integration for consistent import ordering
  - Configuration in `pyproject.toml` for isort settings
  - Added to `dev` and `all` dependency groups

- **Documentation**: Testing instructions in README
  - Added "Running Tests" section with pytest usage
  - Instructions for installing development dependencies

### Changed
- **Code Formatting**: Applied consistent code style across entire codebase
  - Ran `black` formatter on all Python files
  - Ran `isort` for standardized import ordering
  - Fixed invalid escape sequences in visualization module
  - Improved code readability and maintainability

- **CI/CD Improvements**: Stabilized continuous integration pipeline
  - Updated GitHub Actions workflows to Node.js 24
  - Fixed CI code quality checks for import ordering
  - Fixed CI type-check issues
  - Fixed documentation build process
  - Fixed Pandoc compatibility issues
  - Enhanced workflow reliability across all platforms

- **Testing**: Improved test reliability
  - Fixed path-order assumptions in duplicate-file tests
  - Tests now work consistently across different file systems

### Fixed
- Invalid escape sequence in `qbiocode/visualization/visualize_correlation.py`
- Import ordering issues throughout codebase
- Type-check errors in CI pipeline
- Documentation build failures
- Path handling in cross-platform tests

### Planned Features
- Additional quantum ML algorithms
- Enhanced meta-learning capabilities
- More dataset complexity metrics
- Performance optimizations
- Extended Galaxy tool integration

## [0.1.0] - 2026-04-06

### ⚠️ Breaking Changes
- **Minimum Python version increased to 3.10** (was 3.9)
  - Required for compatibility with latest Qiskit ecosystem (qiskit-ibm-runtime 0.44.0+)
  - Python 3.9 reaches end-of-life in October 2025

### Added

#### Core Features
- **QProfiler**: Automated machine learning benchmarking with data complexity analysis
  - Support for multiple ML models (RF, SVM, LR, DT, NB, MLP, XGBoost)
  - Support for quantum ML models (QSVC, PQK, VQC, QNN)
  - Comprehensive data profiling and complexity metrics
  - Batch processing mode for large-scale experiments
  - CLI interface for easy usage

- **QSage**: Meta-learning framework for intelligent model selection
  - Model recommendation based on dataset characteristics
  - Pre-trained models for quick predictions
  - Integration with QProfiler results

- **Data Generation**: Artificial dataset generation with controlled complexity
  - Multiple dataset types (circles, moons, spirals, S-curve, Swiss roll, spheres)
  - Configurable noise levels and complexity parameters
  - Support for multi-class classification problems

- **Embeddings**: Dimensionality reduction and feature extraction
  - Autoencoder implementations
  - Integration with classical embedding methods

- **Evaluation**: Comprehensive model and dataset evaluation
  - Multiple metrics (accuracy, F1-score, AUC, training time)
  - Cross-validation support
  - Statistical analysis tools

- **Visualization**: Publication-quality plotting functions
  - Correlation analysis plots with scientific styling
  - Heatmaps with customizable colormaps
  - High-resolution output (600 DPI) for publications

#### Documentation
- Complete API documentation with Sphinx
- Tutorials for QProfiler, QSage, and data generation
- Installation guides for multiple platforms
- Galaxy integration documentation
- Example notebooks and workflows

#### Project Infrastructure
- GitHub issue templates (bug report, feature request, documentation, question)
- Pull request template with comprehensive checklists
- Security policy (SECURITY.md)
- Support documentation (SUPPORT.md)
- Contributing guidelines (CONTRIBUTING.md)
- Code of conduct (CODE_OF_CONDUCT.md)
- Citation file (CITATION.cff)
- Zenodo metadata for DOI generation (.zenodo.json)

#### CI/CD
- GitHub Actions workflow for continuous integration
  - Multi-OS testing (Ubuntu, macOS, Windows)
  - Multi-Python version testing (3.10, 3.11)
  - Code quality checks (flake8, black, isort, mypy)
  - Test coverage reporting
  - Documentation building
- GitHub Actions workflow for automated releases
  - PyPI publishing
  - Zenodo archiving
  - Release asset management

#### Command-Line Tools
- `qprofiler`: Run QProfiler experiments
- `qprofiler-batch`: Batch processing mode
- `qsage`: Model recommendation tool

### Changed
- Updated visualization functions for publication quality
  - Enhanced scatter plots with better colormaps
  - Improved heatmaps with professional styling
  - Better legend positioning and labeling
  - Increased DPI for high-quality output

### Fixed
- NaN handling in correlation visualization
- Windows path compatibility issues
- Automated QSage model download

### Dependencies
- Python >= 3.10, < 3.13
- Qiskit for quantum computing functionality
- scikit-learn for classical ML algorithms
- pandas, numpy for data manipulation
- matplotlib, seaborn for visualization
- XGBoost for gradient boosting
- hydra-core for configuration management

## [Unreleased]

### Added
- **Testing Infrastructure**: Comprehensive test suite for core functionality
  - `tests/test_data_generation.py`: Tests for data generation utilities
  - `tests/test_file_utilities.py`: Tests for file operations and utilities
  - `tests/test_generator_dispatch.py`: Tests for generator dispatch logic
  - `tests/conftest.py`: Pytest configuration and fixtures
  - Test coverage for utility modules and data generation helpers
  
- **Code Quality Tools**: Enhanced development tooling
  - `isort` integration for consistent import ordering
  - Configuration in `pyproject.toml` for isort settings
  - Added to `dev` and `all` dependency groups

- **Documentation**: Testing instructions in README
  - Added "Running Tests" section with pytest usage
  - Instructions for installing development dependencies

### Changed
- **Code Formatting**: Applied consistent code style across entire codebase
  - Ran `black` formatter on all Python files
  - Ran `isort` for standardized import ordering
  - Fixed invalid escape sequences in visualization module
  - Improved code readability and maintainability

- **CI/CD Improvements**: Stabilized continuous integration pipeline
  - Updated GitHub Actions workflows to Node.js 24
  - Fixed CI code quality checks for import ordering
  - Fixed CI type-check issues
  - Fixed documentation build process
  - Fixed Pandoc compatibility issues
  - Enhanced workflow reliability across all platforms

- **Testing**: Improved test reliability
  - Fixed path-order assumptions in duplicate-file tests
  - Tests now work consistently across different file systems

### Fixed
- Invalid escape sequence in `qbiocode/visualization/visualize_correlation.py`
- Import ordering issues throughout codebase
- Type-check errors in CI pipeline
- Documentation build failures
- Path handling in cross-platform tests

### Planned Features
- Additional quantum ML algorithms
- Enhanced meta-learning capabilities
- More dataset complexity metrics
- Performance optimizations
- Extended Galaxy tool integration

---

## Release Notes

### Version 0.1.0 - Initial Public Release

This is the first public release of QBioCode, a comprehensive framework for quantum machine learning applications in healthcare and life sciences. The release includes:

- Complete implementation of QProfiler and QSage applications
- Support for both quantum and classical machine learning models
- Extensive documentation and tutorials
- Professional project infrastructure for open-source collaboration
- Automated CI/CD pipelines
- Ready for PyPI distribution and Zenodo archiving

**Note**: This is an alpha release. APIs may change in future versions. Please report any issues on our [GitHub issue tracker](https://github.com/IBM/QBioCode/issues).

---

[0.1.0]: https://github.com/IBM/QBioCode/releases/tag/v0.1.0