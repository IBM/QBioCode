Applications
============

QBioCode provides standalone applications for common quantum machine learning workflows. These apps offer user-friendly interfaces and configuration-based workflows for complex analyses.

QProfiler
---------

QProfiler is an automated benchmarking tool for comparing quantum and classical machine learning models. It provides:

* Systematic model evaluation across multiple algorithms
* YAML-based configuration for reproducible experiments
* Automated performance metrics collection (accuracy, F1-score, AUC)
* Statistical analysis and visualization tools
* Support for custom datasets and embeddings

See the :doc:`QProfiler documentation <apps/profiler>` for detailed usage instructions.

QSage
-----

QSage is an intelligent meta-learning system that predicts which machine learning models will perform best on your dataset *before* you run them. By learning from data complexity patterns across multiple datasets, QSage provides data-driven model recommendations.

* Learns from History: Trains on data complexity metrics and model performance from previous experiments
* Predicts Performance: Estimates how well each model will perform on new, unseen datasets
* Ranks Models: Provides confidence-weighted rankings of classical and quantum models
* Saves Time: Helps you focus computational resources on the most promising models


See the :doc:`QSage documentation <apps/sage>` for detailed usage instructions.

.. Made with Bob
