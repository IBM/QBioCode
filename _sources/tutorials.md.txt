# Tutorials

Welcome to the QBioCode tutorials! These Jupyter notebooks provide hands-on examples demonstrating how to use various features and applications of QBioCode for quantum healthcare and life sciences applications.

## Getting Started

Before running these tutorials, make sure you have:
- Installed QBioCode following the [Installation Guide](installation.md)
- Set up your Python environment with all required dependencies
- Access to quantum computing resources (if running quantum algorithms)

---

## Tutorial Gallery

### 1. Artificial Data Generation

Learn how to generate synthetic datasets for testing and benchmarking quantum machine learning algorithms.

<a href="tutorials/Artificial_data_generation/example_data_generation.html">📓 <strong>View Tutorial Notebook</strong></a>

---

### 2. QProfiler - Automated ML Model Benchmarking

Learn how to use QProfiler to systematically benchmark and compare quantum and classical machine learning models on artificial datasets. This tutorial demonstrates:

<a href="tutorials/QProfiler/example_qprofiler.html">📓 <strong>View Tutorial Notebook</strong></a>

**What You'll Learn:**
- Generate artificial datasets with specific characteristics
- Configure and run QProfiler experiments via YAML configuration
- Evaluate multiple ML models (quantum and classical) automatically
- Analyze performance metrics (accuracy, F1-score, AUC)
- Visualize model comparisons and correlations
- Interpret results for model selection

---

### 3. QSage - Quantum-Inspired Feature Importance

Explore QSage, an intelligent meta-learning system that predicts which machine learning models will perform best on your dataset *before* you run them. By learning from data complexity patterns across multiple datasets, QSage provides data-driven model recommendations. This tutorial shows how to:

<a href="tutorials/QSage/qsage.html">📓 <strong>View Tutorial Notebook</strong></a>

**What You'll Learn:**
- Load pre-trained QSage models
- Analyze dataset characteristics (intrinsic dimension, Fisher discriminant ratio, etc.) from compiled ML benchmark results
- Apply QSAGE to predict the model

---

### 4. Quantum Projection Learning (QPL)

Learn about Quantum Projection Learning (QPL), a technique that combines quantum feature maps with multiple classical machine learning algorithms. This comprehensive tutorial demonstrates how to systematically evaluate quantum-enhanced features across different learners.

<a href="tutorials/Quantum_Projection_Learning/QPL_example.html">📓 <strong>View Tutorial Notebook</strong></a>

**What You'll Learn:**
- Generate synthetic datasets with controlled complexity
- Apply quantum feature maps to create quantum projections
- Train multiple classical models (SVC, RF, XGBoost, MLP, LR) on quantum features
- Compare quantum-enhanced vs. classical baseline performance
- Visualize and analyze comprehensive performance metrics
- Use QProfiler for automated QPL experiments

**Key Concepts:**
- Quantum projection methods and expectation value measurements
- Ensemble learning with quantum features
- Data complexity analysis for quantum advantage prediction
- Systematic model comparison and evaluation
- Integration with classical ML pipelines

**Workflow:**
1. Generate or load classification datasets
2. Configure QPL experiments via YAML files
3. Apply quantum feature maps (ZZ, Pauli, etc.)
4. Extract quantum projections from circuits
5. Train 5+ classical models on quantum features
6. Compare with classical baselines
7. Analyze results and identify quantum advantages

---

### 5. Projected Quantum Kernel (PQK) - Ovarian Cancer Survival Prediction

Learn how to apply Projected Quantum Kernels (PQK) to real-world cancer genomics data for survival prediction. This advanced tutorial demonstrates quantum-enhanced machine learning on multi-omics ovarian cancer data from the Multi-Omics Cancer Benchmark (TCGA preprocessed data).

<a href="tutorials/PQK%20-%20OV.html">📓 <strong>View Tutorial Notebook</strong></a>

**What You'll Learn:**
- Automatically download and process multi-omics cancer data
- Create 3-year survival labels from clinical data
- Apply quantum feature maps to high-dimensional genomics data
- Use PQK to create quantum feature representations
- Compare quantum-enhanced vs. classical SVM performance
- Work with multi-omics data (miRNA, methylation, gene expression)
- Perform comprehensive hyperparameter tuning for quantum kernels
- Evaluate quantum performance on real biomedical datasets

**Dataset:**
- Ovarian cancer (OV) multi-omics data from [Multi-Omics Cancer Benchmark](https://acgt.cs.tau.ac.il/multi_omic_benchmark/download.html)
- TCGA preprocessed data with automatic download
- 3-year survival prediction task
- Four data modalities: miRNA, DNA methylation, gene expression, and integrated

**Key Techniques:**
- Automated data download and preprocessing pipeline
- Patient ID standardization across multi-omics datasets
- Survival label creation from clinical data
- Quantum kernel methods with ZZ feature maps
- Pairwise qubit entanglement strategies
- PCA dimensionality reduction for quantum encoding
- Stratified cross-validation for robust evaluation

---

## Additional Resources

- [API Documentation](api_overview.rst) - Detailed API reference
- [QProfiler App](apps/profiler.rst) - Standalone profiling application
- [QSage App](apps/sage.rst) - Feature selection application
- [GitHub Repository](https://github.com/IBM/QBioCode) - Source code and examples

## Support

If you encounter any issues or have questions about the tutorials:
- Check the [GitHub Issues](https://github.com/IBM/QBioCode/issues)
- Review the [Contributing Guide](https://github.com/IBM/QBioCode/blob/main/CONTRIBUTING.md)
- Consult the API documentation for detailed function references
