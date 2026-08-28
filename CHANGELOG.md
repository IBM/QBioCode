# Changelog

All notable changes to QBioCode will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [Unreleased]

### Added
- **QuVINE — Quantum View-based Network Embeddings** (`qbiocode.apps.quvine`), a new
  in-tree app that turns a graph into low-dimensional node embeddings by combining
  classical and quantum walks with skip-gram negative sampling.
  - Multi-view graph construction, random walk with restart (RWR) and discrete-/
    continuous-time quantum walks (DTQW/CTQW), SGNS embedding learning, quantum-calibrated
    filter / GAT / GraphGPS variants, view fusion, and classical baselines
    (node2vec, NetMF, APPNP, GraphSAGE, GCN-MF).
  - 83 method names selectable by a single `method` string, resolved by
    `qbiocode.apps.quvine.resolve_method()` and listed by `list_methods()`.
  - New `quvine` command-line tool: embeds a 2- or 3-column edge list and writes
    `embedding.csv` + `embedding_meta.json`.
  - Programmatic API: `embed(G, method)` returning an `EmbedResult`, plus
    `load_config`, `run_sgns`, `build_quantum_targets`.
  - **Installed via an optional extra**: `pip install "qbiocode[quvine]"`. The Python
    modules ship with QBioCode, but the heavy third-party dependencies (gensim,
    hiperwalk, node2vec, torch-geometric, python-louvain, ripser, omegaconf) do not.
    `import qbiocode` and every classical embedding work without the extra.

- **Graph-complexity metrics**: `qbiocode.evaluate_graph(G, name="")` returns a one-row
  `DataFrame` of graph descriptors (spectral gap, von Neumann entropy, modularity,
  density, community and topological metrics, and more). Deliberately separate from the
  embedding app — run it on the same graph to characterize it. Needs nothing beyond
  networkx/scipy/pandas; the persistent-homology metrics are skipped with a warning
  when `ripser` is absent.

- **Actionable errors for optional dependencies**: `qbiocode.apps.quvine._deps` resolves
  every `[quvine]` dependency through `require_module()`, so a missing one raises
  `QuvineDependencyError` naming the method, the extra, the exact pip command and the
  missing distribution, instead of a bare `ModuleNotFoundError` traceback. It subclasses
  `ImportError`, so code probing with `except ImportError` still works.
  `describe_environment()` and `missing_dependencies()` report what an environment has.

- **QuVINE is a first-class embedding.** `qbiocode.get_embeddings("quvine_rwr", X_train,
  X_test, n_components=8)` now works exactly like `"pca"`, `"nmf"` or `"umap"` — same call
  shape, same return of `(Z_train, Z_test)`. Any of the 83 QuVINE method names is accepted;
  a symmetric kNN graph is built over the rows, embedded, and reduced to `n_components`.
  QProfiler reaches them through its existing `embeddings:` config list, and passes
  `quvine_args` through for per-method overrides.
  - `qbiocode.SKLEARN_METHODS`, `qbiocode.QUVINE_HEADLINE_METHODS` and
    `qbiocode.QUVINE_METHODS` name what is available; `QUVINE_METHODS` is empty rather
    than raising when the `[quvine]` extra is absent.
  - Methods needing no extra beyond the base install — `netmf`, `appnp`, `gat_*` — work on
    a bare `pip install qbiocode`.

- **`qbiocode.is_transductive(name)`** reports whether a method sees test *features* at
  embed time. `spectral` and every QuVINE method have no out-of-sample `transform`, so they
  are fitted over the stacked train+test rows and sliced. `get_embeddings` now also emits a
  single `UserWarning` per call for those methods, stating plainly that test features
  participate in the geometry while test *labels* never do. Inductive methods are silent.

- **Quantum Backend**: Noisy simulation support based on IBM device noise models
  - New backend format: `noisy_<device_name>` (e.g., `noisy_ibm_cleveland`)
  - Extracts noise model from actual IBM Quantum devices for realistic local simulation
  - Configurable simulation method via `sim_method` parameter
  - Supports all AerSimulator methods: `statevector`, `matrix_product_state`, `tensor_network`, etc.
  - Enables testing quantum algorithms under realistic noise without hardware queue times
  - No IBM Quantum compute credits consumed (simulation runs locally)
  - Updated `qbiocode.utils.qutils.get_backend_session()` to support noisy backends
  - Updated `qbiocode.utils.ibm_account.py` for improved credential handling
  - Documentation updated with comprehensive examples and best practices

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

- **Packaging**: dependencies are now tiered under `requirements/`
  - `requirements/requirements-base.txt` is the single source of truth for the
    package's runtime dependencies; `pyproject.toml` reads it via
    `[tool.setuptools.dynamic]`. It ships in the sdist, so building from an
    sdist no longer risks an empty dependency list.
  - `requirements/requirements-quvine.txt` backs the new `[quvine]` extra,
    `requirements/requirements-docs.txt` backs `[docs]`, and
    `requirements/requirements.txt` installs a complete development environment.
  - The root `requirements.txt` is now a pointer to
    `requirements/requirements.txt`, so the documented
    `pip install -r requirements.txt` keeps working unchanged.
  - `[all]` is now defined as `qbiocode[apps,quvine,docs,dev]` instead of a
    hand-maintained copy of every other extra's contents, which had already
    drifted out of sync.

- **Packaging**: `setup.py` is now a 33-line shim that passes no arguments, so
  `pyproject.toml` is the only place project metadata is declared. The previous
  189-line version re-declared name, version, dependencies, extras, classifiers
  and entry points, and parsed `requirements.txt` by looking for a
  `# Documentation` comment to split runtime from docs dependencies. Duplicated
  metadata is what allowed the two files to disagree.

- **Dependencies**: removed `tensorflow` from the dependency set. Nothing in
  `qbiocode/`, the tutorials, or the docs imports TensorFlow or Keras — the only
  remaining reference was a stale entry in `autodoc_mock_imports`. This removes
  roughly 600 MB from a default install. `qbiocode/embeddings/compute_autoencoder.py`,
  the one module that could have needed it, is written against PyTorch, which
  remains a dependency.

### Fixed

#### ⚠️ Train/test contamination in QProfiler — results change
- **QProfiler no longer scales the test set with test-set statistics.** Previously
  `qprofiler.py` called `scaler_fn` separately on each split, which fit a *fresh*
  `MinMaxScaler` on `X_test`. The test features were therefore normalized using the test
  set's own min/max — both a use of held-out information and a train/test distribution
  mismatch, since the model was trained under a different transform than it was evaluated
  under. One scaler is now fit on `X_train` and applied to `X_test` via the new
  `qbiocode.scale_train_test`.
- **Consequence: metrics produced by QProfiler before this change are not comparable to
  metrics produced after it.** Any accuracy/F1/AUC numbers carried over from an earlier run
  should be regenerated. Affected tutorial notebooks have been re-executed.
- **`seed` now actually controls the train/test split.** `train_test_split` was called with
  no `random_state`, so iterations were irreproducible and the configured `seed` was silently
  ignored for splitting — despite an inline comment claiming the splits "will be based on the
  seed". Each iteration now uses `random_state = seed + iter`: distinct across iterations,
  deterministic across reruns, and independent of other RNG consumers.

#### Embeddings
- `get_embeddings(..., n_components=None)` — the documented default — raised
  `TypeError: '<=' not supported between instances of 'NoneType' and 'int'` instead of
  defaulting to the feature count. It now defaults to `X_train.shape[1]` as documented.
- `spectral` embedding raised `AttributeError: 'SpectralEmbedding' object has no attribute
  'transform'` on every run. `SpectralEmbedding` has no out-of-sample transform, so it is now
  fit transductively over the combined train+test rows and sliced back. Only feature structure
  participates — no labels — so no label leakage is introduced.

#### Quantum sessions and caching
- Qiskit Runtime sessions in `embed.pqk` and `learning.compute_pqk` are now closed in a
  `finally` block. An exception mid-computation previously leaked the session, leaving jobs
  queued against the user's account.
- **PQK projection caches are now keyed by the feature map that produced them.** The cache
  filename omitted `encoding`, `entanglement`, `reps` and `primitive`, so re-running with a
  different feature map into the same `pqk_projection_dir` silently reloaded the previous
  run's projections and reported them as the new result. A short digest of those parameters
  is now part of the filename.
- Cached projection files are also validated on width (`3 x feat_dimension`), not just row
  count, so a file written with a different feature dimension is rejected rather than reused.

#### QuVINE method resolution and packaging
- **The `quvine` CLI rejected its own default method.** `--method` defaults to
  `quvine_fused`, but the CLI validated it against `list_methods()`, and the five fused
  names (`quvine`, `quvine_fused`, `quvine_sgns`, `quvine_sgns_fused`, `fused`) were
  handled directly inside `embed()` without ever being registered in the alias tables.
  Running `quvine -i edges.csv -o out/` therefore always exited 1 with
  `unknown method 'quvine_fused'`, and `resolve_method("quvine_fused")` raised
  `KeyError`. Fused names are now a first-class kind in
  `qbiocode/apps/quvine/api/aliases.py`: `resolve_method` returns `("fused", "fused")`,
  `list_methods()` includes them (83 names, up from 78), `list_methods("fused")` filters
  to them, and `core.embed` sources its fused-name set from the same table instead of
  duplicating it. The CLI now validates through `resolve_method`, so its message carries
  close-match suggestions and can no longer drift from what `embed()` dispatches.

- **Four QuVINE subpackages were missing from a built wheel.** `walks/`, `corpus/`,
  `utils/` and `configs/` had no `__init__.py`, so `[tool.setuptools.packages.find]`
  skipped them. The modules imported fine from a source checkout — including
  `from qbiocode.apps.quvine.walks.rwr import ...` — which is exactly why this went
  unnoticed; an installed wheel raised `ModuleNotFoundError`. Each directory is now a
  real package, and `tests/test_quvine_packaging.py` fails if any regains modules
  without an `__init__.py`.

- **`import qbiocode` no longer pulls in the `[quvine]` dependency set.**
  `qbiocode/apps/quvine/api/__init__.py` eagerly imported `core`, `config`, `sgns` and
  `targets`, so reaching the stdlib-only `resolve_method` dragged in omegaconf and
  everything behind it. That only appeared to work because `hydra-core` pulls omegaconf
  in transitively. Those attributes are resolved lazily now, which is what lets
  `qbiocode.embeddings` probe method names on a bare install. The `quvine` CLI likewise
  imports `embed` at its point of use, so `--help` and `--list-methods` work without
  the extra.

- **`torch` is declared in the base dependency set only.** It was listed in both the
  base set and the `[quvine]` extra, which implied it was optional —
  `qbiocode.embeddings.__init__` imports `ConvAutoencoder`, which imports torch eagerly,
  so a bare `import qbiocode` already requires it. A missing torch is a broken install,
  not a missing extra, and the `[quvine]` install hint would have been wrong advice.

- **Stale dependency advice removed.** `embedding/word2vec.py` raised
  `ImportError("... Try: pip install gensim==4.3.0 scipy==1.11.0")`, advising a downgrade
  that conflicts with `qiskit-machine-learning==0.9.0` (which requires numpy>=2.0, while
  gensim<4.4 forces numpy<2.0). It now names the `[quvine]` extra. The unused
  `GENSIM_AVAILABLE` module flag was dropped along with it.

- **Documentation claims corrected.** `qbiocode/apps/quvine/__init__.py` and
  `docs/source/apps/quvine.rst` stated QuVINE was "available on a plain
  `pip install qbiocode` / `git clone` with no extra install step". Both now document the
  `[quvine]` extra and what still works without it. A stale comment about avoiding a
  TensorFlow/Keras import at load time was also removed — no TensorFlow exists in the tree.

- **Every registry method was unreachable.** `qbiocode/apps/quvine/data/` — imported by
  `embedding/quantum_filters.py`, and through it by the adapter module that builds the
  method registry — was never committed to the internal repository: an *unanchored* `data/`
  rule in its `.gitignore` matched the directory at every depth and excluded it silently
  (it appears nowhere in that repository's history). All 69 registry methods therefore died
  with `ModuleNotFoundError: No module named 'qbiocode.apps.quvine.data'`.
  `data/subgraph.py` is reimplemented here from its two call sites — bounded-radius ego-net
  expansion is a well-determined graph primitive — which restores the whole registry.
  Four modules remain absent (`data_loader`, `prepare`, `random_graphs`,
  `random_graphs_extended`); they gate only `Pipeline`'s on-disk graph loading and the
  synthetic benchmark generators in `reproducibility.graph_generator`, and they now raise
  `QuvineDataUnavailableError` naming the module and the feature instead of a bare
  `ModuleNotFoundError`. `qbiocode.apps.quvine.Pipeline` is importable again. External's
  `.gitignore` anchors the rule as `/data/`, so dropping the original files in will track
  them with no further change.
- **Missing optional dependencies were attributed to the wrong feature.**
  `walks/ctqw.py` and `walks/dtqw.py` bound `hiperwalk` at module scope, and `walks/base.py`
  imports both eagerly — so an RWR-only run, which never performs a quantum walk, failed
  reporting *"continuous-time quantum walks (CTQW) requires ... missing: hiperwalk"*.
  `baselines/node2vec.py` had the same shape, and because `baselines/__init__.py` wraps its
  imports in `except ImportError: pass`, that left `run_node2vec` unbound and every registry
  method — netmf, appnp, graphgps — failed with a node2vec error. All three now resolve
  their dependency at call time and each names its own. A test asserts structurally that no
  module under `apps/quvine` calls `require_module` at module scope.
- GraphGPS's missing-`torch_geometric` message told the user to `pip install
  torch-geometric` by hand; it now goes through `require_module` and names the `[quvine]`
  extra like every other one.
- **The method registry printed to stdout.** `run_method` wrote `✓ netmf: 0.00 minutes` /
  `✗ node2vec: FAILED - ...` on every call and logged a full traceback at `ERROR` for a
  failure it then returned to its caller to re-raise. QProfiler calls this once per method
  per iteration. Progress now goes through `logging`, the traceback is available at `DEBUG`,
  and the causing exception travels on `MethodResult.exception` so `api.core` chains it
  (`raise QuvineMethodError(...) from exc`) — the cause reaches the caller through the
  exception rather than a stray log line.
- `get_embeddings` validated nothing. Unknown method names raised `KeyError` or fell through
  to an sklearn assertion; a non-string `embedding`, a non-integer, zero, negative, or
  wider-than-the-input `n_components` all produced errors from deep in sklearn. Each now
  raises `ValueError` at the boundary, naming the parameter, the value received and the
  accepted set, with close-name suggestions for typos (`"pcaa"` → *Did you mean: pca?*).
- `qbiocode.embeddings`' module docstring documented `get_embeddings(X, method='pca',
  n_components=2)` — a signature the function has never had.

#### Packaging and dependency declarations
- Three dependencies were imported by the library but never declared:
  `pyyaml` (`qbiocode/utils/generate_qml_configs.py` and
  `qbiocode/apps/qprofiler/qprofiler_batchmode.py`, which backs the
  `qprofiler-batch` console script), `matplotlib` (five modules, including
  `qbiocode/visualization/visualize_correlation.py`), and `joblib`
  (`qbiocode/evaluation/model_run.py` and `qprofiler_batchmode.py`). Installs
  only worked because `seaborn` and `optuna` happened to pull them in
  transitively. All three are now declared explicitly.
- `MANIFEST.in` referenced `apps/qprofiler/configs`, a path that does not exist
  in this tree — the configs live at `qbiocode/apps/qprofiler/configs`. The
  sdist therefore shipped no Hydra config for QProfiler. Likewise
  `[tool.setuptools.packages.find]` listed an `apps*` include glob that matched
  no package.
- `pandoc` was declared as a pip dependency of the `docs` extra. Pandoc is a
  system binary; the PyPI distribution of that name does not provide it.
  Removed, with the platform install commands documented in
  `requirements/requirements-docs.txt` instead.
- `.gitignore` matched `data/` and `results/` unanchored, so those patterns
  excluded a directory of either name at *any* depth — including
  `docs/source/tutorials/QProfiler/data/`, which holds the committed `.h5ad`
  and `sc_binary/*.csv` fixtures the published notebooks read. Because git
  cannot re-include a file whose parent directory is excluded, those fixtures
  were only addable by virtue of already being tracked. Both patterns are now
  anchored to the repository root.
- Stopped tracking generated QProfiler output under
  `docs/source/tutorials/QProfiler/` (`ModelResults.csv`,
  `RawDataEvaluation.csv`, `results.pkl`, and 500 KB of `pqk_projections/*.npy`).
  The notebooks write these files and read them back within the same run, and
  `nbsphinx_execute = 'never'` means the docs build never executes them. The
  cached projections were additionally unreachable after the PQK cache-key fix
  above, since their filenames predate the feature-map fingerprint.

#### Earlier fixes
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