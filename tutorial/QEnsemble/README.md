# Quantum Ensemble Learning Tutorial

This tutorial demonstrates quantum ensemble learning methods for classification tasks using the QBioCode framework.

## Overview

Quantum ensemble learning combines multiple quantum classifiers to improve prediction accuracy and robustness. This implementation uses controlled swap operations and quantum superposition to create ensembles of training data arrangements.

**Key Innovation**: By leveraging quantum superposition, the ensemble evaluates multiple training set configurations simultaneously, potentially offering advantages over classical ensemble methods.

**Note**: The core quantum ensemble functionality has been integrated into the QBioCode package as `qbiocode.learning.compute_qensemble`. This tutorial now demonstrates how to use the integrated API.

## Files

### Tutorial Files

- **`QEnsemble_example_blobs.ipynb`**: Complete tutorial demonstrating quantum ensemble learning on synthetic blob datasets using the QBioCode API

- **`README.md`**: This file - tutorial guide

## Using QBioCode API

The quantum ensemble functionality is now available through the QBioCode package with a unified interface supporting two construction methods:

### Basic Usage

```python
from qbiocode.learning import compute_qensemble

# Fixed swap ensemble (default, faster)
results = compute_qensemble(
    X_train, X_test, y_train, y_test,
    args={'backend': 'simulator', 'grid_search': False},
    n_train=4,              # Training samples (must be even)
    d=2,                    # Ensemble depth (creates 2^2=4 members)
    n_swap=1,               # Operations per control qubit
    mode="balanced",        # Sampling strategy
    ensemble_method="swap", # Fixed swap method (default)
    n_shots=8192,           # Measurement shots
    seed=42
)

# Random unitary ensemble (advanced, more general)
results = compute_qensemble(
    X_train, X_test, y_train, y_test,
    args={'backend': 'simulator', 'grid_search': False},
    n_train=4,
    d=2,
    n_swap=1,
    mode="balanced",
    ensemble_method="random_unitary",  # Use Haar-random unitaries
    n_shots=8192,
    seed=42
)
```

**Ensemble Methods:**
- **`"swap"`** (default): Uses controlled-SWAP gates for deterministic data rearrangement
  - Faster execution
  - Deterministic circuit structure
  - Good for most applications
  
- **`"random_unitary"`**: Uses Haar-random unitaries for more general transformations
  - More general transformation space
  - Potentially better generalization
  - More computationally expensive
  - Samples from the full unitary group U(N)

### Available Utility Functions

From `qbiocode.utils`:
- `normalize_data()`: Normalize data vectors for quantum state encoding
- `label_to_array()`: Convert binary labels to one-hot encoding
- `prepare_training_set()`: Select and prepare balanced training subsets
- `retrieve_probabilities()`: Extract probabilities from measurement counts

## Key Concepts

### Quantum Cosine Classifier

The quantum cosine classifier measures similarity between quantum states using the controlled-SWAP test (also known as the SWAP test):

**Algorithm Steps**:
1. **State Preparation**: Encode training data, test data, and labels as quantum states
2. **Superposition**: Apply Hadamard gate to test label qubit
3. **Controlled-SWAP**: Swap training and test data qubits controlled by test label
4. **Interference**: Apply Hadamard gate to test label qubit
5. **Label Integration**: CNOT from training label to test label
6. **Measurement**: Measure test label qubit

**Mathematical Foundation**: The measurement probability P(0) = 1/2 + 1/2 * |⟨train|test⟩|² encodes the squared cosine similarity between quantum states, providing a natural similarity metric for classification.

### Quantum Ensemble Methods

#### Balanced Mode
- Samples training data pairs from each class separately
- Maintains class balance in superposition states
- Best for balanced datasets

#### Unbalanced Mode
- Randomly samples training data pairs without class constraints
- Simpler circuit structure
- Suitable for imbalanced datasets

#### Pair Sample Mode
- Creates all possible pairwise swaps
- More comprehensive exploration of data arrangements
- Higher circuit depth

#### Random Unitary Mode
- Uses random unitaries sampled from U(N) instead of fixed swaps
- Applies controlled random unitaries to combined data+label registers
- More general transformation providing uniform coverage of unitary group
- Potentially better generalization but computationally more expensive
- Unitary dimension: 2^(n_obs_qubits + n_obs)


## Parameters

### Ensemble Parameters

- **`d`**: Number of control qubits (ensemble depth)
  - Creates 2^d ensemble members in superposition
  - Higher values = deeper ensembles with more diversity
  - Typical range: 1-3 (d=3 creates 8 ensemble members)
  - Constraint: n_train > d (need more samples than control qubits)

- **`n_swap`**: Number of swap/unitary operations per control qubit
  - Controls diversity of ensemble members
  - Typical range: 1-5
  - More operations = more diverse data rearrangements
  - In random unitary mode: number of Haar-random unitaries applied

- **`n_train`**: Number of training samples
  - Must be even for balanced mode (n/2 per class)
  - Typical values: 4, 8, 16
  - Limited by qubit count: log2(n_features) * n_train qubits needed
  - Constraint: n_train > d

- **`mode`**: Ensemble sampling strategy
  - `"balanced"`: Class-balanced sampling (recommended for balanced datasets)
  - `"unbalanced"`: Random sampling (suitable for imbalanced data)
  - `"pair_sample"`: All pairwise swaps (most comprehensive, highest depth)

- **`n_shots`**: Number of measurement shots
  - Higher = more accurate probability estimates
  - Typical range: 1024-8192
  - Trade-off: accuracy vs. execution time
  - IBM hardware: typically 4096 due to cost constraints

- **`device`**: Execution device
  - `'CPU'`: Local CPU simulation (default)
  - `'GPU'`: GPU-accelerated simulation (requires qiskit-aer-gpu)
  - `'ibm_*'`: IBM Quantum hardware (e.g., 'ibm_kyoto')

## Performance Considerations


### Optimization Tips

- Start with small configurations: d=2, n_train=4, n_features=2
- Use PCA/UMAP for dimensionality reduction on high-dimensional data
- For hardware: enable error mitigation and dynamic decoupling
- Monitor qubit count before execution to avoid memory issues

## References

### Primary References
- **This Tutorial**: Part of QBioCode framework for quantum bioinformatics
- **Quantum Ensemble Methods**:
  - Macaluso et al., "A variational algorithm for quantum ensemble learning", IET Quantum Communication (2023)
  - Rhrissorrakrai et al., "Quantum Ensembling Methods for Healthcare and Life Science", arXiv:2506.02213 (2025)
    [https://arxiv.org/abs/2506.02213](https://arxiv.org/abs/2506.02213)
- **Original Implementation**: [GitHub Repository](https://github.com/amacaluso/Quantum-algorithm-for-ensemble-using-bagging)
- **QBioCode Documentation**: See main repository for comprehensive guides
