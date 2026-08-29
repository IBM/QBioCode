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

#### ⚠️ Undefined measurements are now `nan`, not a plausible number — results change
Several functions returned an achievable value when they could not compute anything at
all, which made "undefined" indistinguishable from a real result in an aggregated table.
Every one of these now returns `nan`, and every consumer aggregates nan-aware. **Any
table or ranking built from the metrics below should be regenerated**; affected notebooks
have been re-executed.

- **Link-prediction ranking metrics no longer report `0.5` for an undefined AUC.**
  `reproducibility.method_adapters.evaluate_link_prediction` returned
  `{"auc_roc": 0.5, "auc_pr": 0.5, "f1": 0.0}` whenever the evaluation split held a single
  class, or whenever `roc_auc_score` raised. `0.5` is the score of a random ranker — a real,
  and quite common, outcome — so a method that could not be scored sorted level with
  methods that genuinely failed to learn. It now returns `nan` for all three, logs which
  case occurred, and validates that the supplied node indices exist in the embedding matrix
  rather than raising `IndexError` several frames deeper.
- **`summarize_link_prediction_results` is nan-aware.** It averaged with `np.mean`, so one
  undefined method turned every aggregate into `nan` and erased the methods that had
  succeeded. Aggregates now use `np.nanmean`/`np.nanstd` and each metric carries an
  `n_defined_<metric>` count, so a reader can see how many methods the mean is over. An
  empty result set yields `nan` rather than `0.0`.
- **`evaluate_graph` metrics that cannot be computed are absent or `nan`, never `0.0`.**
  `modularity` was set to `0.0` on the exception path and for an edgeless graph;
  `path_length_ratio` likewise. Modularity is `Q = Σ_c (e_c/m − (d_c/2m)²)`, undefined at
  `m = 0`, and `0.0` states "no community structure beyond chance" — a finding. Both now
  report `nan`. `compute_community_metrics` previously disagreed with itself on this,
  returning `modularity: 0.0` alongside `approx_conductance: nan` on the same line.
  (The `0.0` `path_length_ratio` for a *disconnected* graph is retained deliberately: that
  is a measurement, not a failure.)
- **`compute_quantum_advantage_metrics` on an empty graph returns `nan` for all ten
  metrics**, including the composite `quantum_advantage_score`, instead of `0.0`. A `0.0`
  score is rankable, so an empty graph previously sorted alongside genuinely unpromising
  ones. This function has no callers outside `graph_evaluation.py`, so no existing ranking
  changes order — only the values reported for degenerate inputs.
- **`netmf` no longer synthesizes an embedding from random noise.** When its truncated SVD
  failed to converge, and when the deepwalk-matrix construction failed, it returned
  `np.random.randn(n, dim) * 0.01` — which the registry scored, logged and ranked as a
  result. Both paths now raise `RuntimeError` naming the stage that failed, chained from the
  underlying `ArpackError`/`LinAlgError`. A test asserts the module's source contains no
  random-matrix fallback.
- `evaluate_graph(None)` raises `TypeError` instead of returning a size-only summary
  reporting "0 nodes" — which was indistinguishable from a genuinely empty graph. An empty
  `nx.Graph()` remains a valid input and still yields the size-only summary with a warning.
  The two metric families also degrade independently: if one fails, its columns are simply
  absent from the returned frame and a `UserWarning` names the exception type, rather than
  the frame being padded with fabricated values.

#### Silent no-ops reported as success
- **`get_optimizer("L_BFGS_B")` never worked.** The branch read
  `optimizer == L_BFGS_B(maxiter=max_iter)` — a comparison, not an assignment — so it
  constructed the optimizer, discarded it, and left `optimizer` unbound. Every request
  raised `UnboundLocalError: cannot access local variable 'optimizer'`, despite `L_BFGS_B`
  being advertised in the `Literal` type hints of both `compute_vqc` and `compute_qnn`.
- **`get_feature_map` returned its own argument for an unrecognized name.** The if/elif
  chain had no `else`, so `feature_map` stayed bound to the caller's string and failed
  several frames later with `'str' object has no attribute 'num_qubits'`. A mistyped
  `encoding` — `'zz'` for `'ZZ'` — was therefore reported as a qiskit type error.
  `SUPPORTED_FEATURE_MAPS`, `SUPPORTED_ENTANGLEMENTS` and `SUPPORTED_OPTIMIZERS` are now
  module constants in `qbiocode.utils.qutils`, validated at entry and reusable by callers.
  An unknown `entanglement` previously reached qiskit and came back as
  `ValueError: Something went wrong in Rust space`.
- **`scale_train_test` silently returned unscaled data for a mistyped scaler name.** The
  `else` branch fell through to the `"None"` behaviour, so `scaling="minmaxscaler"`
  disabled scaling and reported success. It now raises `ValueError`.
- **QProfiler's `scaling` config was tested with a substring match.** `'True' in
  args['scaling']` accepted `'MinMaxScalerTrue'`, silently ignored the lowercase `['true']`,
  and raised `TypeError: argument of type 'bool' is not iterable` for the most natural YAML
  of all, `scaling: true`. All spellings — bool, `['True']`, `true`/`yes`/`1`,
  `false`/`no`/`0`/`none`, or a scaler name — are now resolved explicitly, and anything else
  is an error rather than a quietly disabled transform.
- **`compute_pqk` accepted a `primitive` it never used.** `primitive="sampler"` changed only
  the cache fingerprint, not the computation: projected quantum kernels are Pauli expectation
  values and the backend is requested as `"estimator"` unconditionally. That produced two
  cache files holding identical projections and a caller who believed they had measured
  something else. Only `"estimator"` is accepted now, and the message says why.
- **A mistyped `--config` path could produce a successful `quvine` run with the wrong
  settings**, by falling back to the packaged config. The path is now checked with
  `os.path.isfile` before anything else runs.
- **`QSage.__init__` aliased its metric list instead of copying it.**
  `self._available_metrics = self._columns_metrics` followed by an in-place `.sort()` bound
  both names to one list, so sorting the public attribute silently reordered the column list
  used to slice the input frame. It is now `sorted(...)`, which returns a new list.
- A QProfiler run whose `folder_path` matched no dataset, or whose `iter` was `0`, exited
  successfully having produced no output at all — indistinguishable from a run whose models
  all failed. Both are now errors.

#### Errors attributed to the wrong cause
- **Dead sentinel guards removed.** `try: from xgboost import XGBClassifier / except
  ImportError: XGBClassifier = None` (in `qbiocode/__init__.py`, `learning/__init__.py` and
  `learning/compute_pqk.py`) replaced an actionable `ImportError` with
  `TypeError: 'NoneType' object is not callable` at the point of use — a message that names
  neither xgboost nor the fix. xgboost is a declared base dependency, so its absence is a
  broken install. `compute_qpl`'s guard is retained (it holds an optional import) but now
  records the original `ImportError` and quotes it in the message. The PyTorch-Geometric
  guard in `reproducibility/graph_generator.py` does the same.
- **Broad `except Exception` handlers narrowed to the failure they actually handle**, so a
  bug in the handler's own body no longer masquerades as a degenerate graph: graph-metric
  handlers in `baselines/gat.py`, `baselines/graphgps.py` and `evaluation/graph_evaluation.py`
  catch `nx.NetworkXException`; SVD and eigendecomposition handlers in `baselines/netmf.py`,
  `baselines/graphsage.py` and `fusion/fuse_fixes.py` catch
  `(scipy.sparse.linalg.ArpackError, np.linalg.LinAlgError, ValueError)`; the edge-list
  reader in the `quvine` CLI catches the four ways a text table fails to parse. Where a
  handler must stay broad — `evaluate_graph`'s two public metric blocks,
  `reproducibility/validator.py` — it now names `type(e).__name__` in the message and keeps
  the traceback at `DEBUG`, because the alternative is a missing column with nothing to
  trace it back to.
- `evaluation/classification.py` discarded the reason the *requested* label strategy failed
  before falling back to `label_propagation`, so a silently-substituted labelling was
  indistinguishable from the requested one. The substitution is now logged with its cause,
  and if the fallback also fails the warning names both failures rather than only the last.
- `graphgps.py` read precomputed node dictionaries with `d[node]`, raising `KeyError` for a
  node absent from a partial computation; it uses `.get(node, default)` now. `get_nodelist` /
  `_stable_nodelist` validate that a supplied `nodelist` is a permutation of the graph's
  nodes, instead of silently producing misaligned feature rows.
- **Validation moved to the boundary.** `get_embeddings` coerces and shape-checks `X_train`
  and `X_test` up front (a list of lists — the natural thing to pass from a notebook — used
  to fail on `'list' object has no attribute 'shape'`), and the new public
  `qbiocode.embeddings.check_embedding_name` lets a caller validate an entire `embeddings`
  list *before* any work begins, with the identical message; QProfiler uses it for exactly
  that, so a typo in the sixth entry no longer surfaces after the first five have run.
  `compute_pqk` performs eleven argument checks before it creates directories or reads a
  projection cache. `evaluation/model_run.py` rejects unknown model names with the available
  set instead of a bare `KeyError` raised inside a joblib worker, and falls back to estimator
  defaults — with a log line — for a model that has no `<model>_args` config block, which
  `xgb` and `qpl` do not in the shipped `config.yaml`. The `qprofiler`, `qsage` and `quvine`
  CLIs validate every argument before creating an output directory or reading input:
  `--cv 1`, `--test-size 1.0`, `--n-iter 0`, an `--input` that is not a directory, an empty
  `--sep`, and a graph with no nodes are all caught up front.
- `tests/test_error_contracts.py` (43 tests) pins the behaviour above: that undefined
  measurements are `nan`, that a silent no-op is now an error, and that each message names
  the parameter, the value received and the accepted set.

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

- **`qbiocode/apps/quvine/data/` was missing entirely, and with it every registry
  method.** The directory was never committed to the internal repository: an *unanchored*
  `data/` rule in its `.gitignore` matched it at every depth, so it was excluded silently
  and appears nowhere in that repository's history. Because
  `embedding/quantum_filters.py` imports it at module scope — and through it the adapter
  module that builds the method registry — all 69 registry methods died with
  `ModuleNotFoundError: No module named 'qbiocode.apps.quvine.data'`, and
  `qbiocode.apps.quvine.Pipeline` was unimportable. The six modules
  (`data_loader`, `prepare`, `sparsify`, `subgraph`, `random_graphs`,
  `random_graphs_extended`) are recovered from the working tree they were written in and
  are now tracked, restoring the registry, `Pipeline`, and all 15 synthetic graph families
  in `reproducibility.graph_generator`. External's `.gitignore` anchors the rule as
  `/data/`, so this cannot recur here. `graph_complexity.py`, which sat alongside them in
  the standalone QuVINE distribution, is deliberately **not** carried over — nothing
  imports it and `qbiocode.evaluate_graph` supersedes it.
- `data.subgraph.expand_neighborhood` filtered roots absent from the graph only for
  `radius >= 1`; at `radius=0` it returned the roots verbatim and so could hand back nodes
  the graph does not contain. Roots are now filtered before the radius check.
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

#### Plotting no longer reconfigures, blocks, or overwrites
- **`import qbiocode` no longer rewrites the importing program's matplotlib.**
  `visualization/visualize_correlation.py` assigned 27 entries into the global
  `plt.rcParams` at module scope — font family, size, tick geometry, spine visibility,
  and a 600-dpi `savefig` default. Because `qbiocode/__init__.py` imports the module,
  merely importing the package restyled every unrelated figure the caller drew
  afterwards, with nothing in the traceback or the call stack to attribute it to. The
  settings are now `qbiocode.visualization.PUBLICATION_STYLE`, applied for the duration
  of a plotting call through `plt.rc_context` and restored even when plotting raises.
  `qbiocode.publication_style()` returns a copy for callers who want the same look on
  their own figures. `tests/test_plotting_hygiene.py` asserts in a subprocess that
  importing the package leaves `rcParams` untouched, and an AST guard fails if any
  module under `qbiocode/` regains a module-level `rcParams` assignment,
  `matplotlib.use()`, or `sns.set_theme()`.
- **⚠️ `plot_results_correlation` no longer overwrites its own output.** It writes three
  figures, deriving the second and third paths with
  `re.sub(".pdf", "_heatmap.pdf", save_file_path)`. That pattern did not match any
  extension other than `.pdf`, so the derived path *equalled the original*: a caller who
  passed `corr.png` had the scatter plot overwritten by the clustered heatmap and that
  overwritten by the non-clustered one, ending with one file where three were requested
  and nothing to indicate two had been lost. The `.` was also an unescaped regex
  wildcard, so `out/spdf_x.pdf` became `out/_heatmap.pdf_x_heatmap.pdf`. Paths are now
  derived with `os.path.splitext`, any image format works, and a missing output directory
  is created rather than raising. `.pdf` filenames are unchanged.
  `QSage.plot_results` had the same `re.sub('.pdf', '', saveFile)` defect and forced a
  `.pdf` extension regardless of the request, producing files named
  `results.png_f1_score_barplot.pdf`; it now honours the extension it was given.
- **Library code no longer blocks on a window.** `plt.show()` was called unconditionally
  in `visualize_correlation` (three times per call), `apps/quvine/analysis/analyze.py`
  (four sites) and `QSage.plot_results` (two) — hanging a batch QProfiler run under a GUI
  backend, and emitting `UserWarning: FigureCanvasAgg is non-interactive` under `Agg`.
  Every site is now guarded on the backend actually being able to display.
  `plot_results_correlation` keeps `show_plots=True`, since five tutorial notebooks rely
  on it for inline display; `QSage.plot_results` gained `show=None`, meaning "show only
  when not saving", which is what its docstring already claimed. No `matplotlib.use("Agg")`
  guard was added: forcing a backend at import is the same hidden global mutation this
  change removes, and matplotlib already falls back to `Agg` when no GUI toolkit is
  importable.
- **`plot_singular_values` and `spectral_info` saved *after* showing**, so under a GUI
  backend the file was not written until someone closed the window — a batch run stalled
  with nothing on disk. They save first now. `spectral_info` also wrote
  `log_spectrum.png`, `loglog_spectrum.png` and `log_normalized_spectrum.png` into the
  process's current working directory under fixed names, so two runs from the same
  directory silently overwrote each other; it takes an `outdir` argument.
- **Figures no longer accumulate.** Every `plt.close()` — which closes whatever figure is
  *current*, not necessarily the one just drawn — is now `plt.close(fig)` on the specific
  figure, and the touched functions use the explicit `fig, ax = plt.subplots()` API rather
  than the pyplot state machine. This matters because QProfiler plots once per metric per
  embedding per iteration.
- **The plotting functions return their figures.** `plot_results_correlation` returns a
  `CorrelationFigures` named tuple (`scatter`, `scatter_ax`, `clustered_heatmap`,
  `ordered_heatmap`), `QSage.plot_results` returns its list of figures, and
  `plot_singular_values` returns its figure — so a notebook can restyle or re-save
  without recomputing. All are closed before returning, which removes them from pyplot's
  manager while leaving each fully usable for `fig.savefig(...)` and inline display.
- `plot_results_correlation`'s docstring documented `Returns: None`; the module docstring
  of `qbiocode.visualization` documented an `output_dir=` parameter the function has never
  had. Three `print()` calls in it, and one in `QSage.plot_results`, now go through
  `logging`.

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