# ====== Base class imports ======
import json
import logging
import os

import pandas as pd

# ======= Parallelization =====
from joblib import Parallel, delayed

current_dir = os.getcwd()

logger = logging.getLogger(__name__)


def model_run(X_train, X_test, y_train, y_test, data_key, args):
    """This function runs the ML methods, with or without a grid search, as specified in the config.yaml file.
    It returns a python dictionary contatining these results, which can then be parsed out. It is designed to run
    each of the ML methods in parallel, for each data set (this is done by calling the Parallel module in results below).
    The arguments X_train, X_test, y_train, y_test are all passed in from the main script (qmlbench.py) as the input
    datasets are processed, while the remaining arguments are passed from the config.yaml file.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        data_key (str): Key for the dataset being processed.
        args (dict): Dictionary containing configuration parameters, including:
            - model: List of models to run.
            - n_jobs: Number of parallel jobs to run.
            - grid_search: Boolean indicating whether to perform grid search.
            - cross_validation: Cross-validation strategy.
            - gridsearch_<model>_args: Arguments for grid search for each model.
            - <model>_args: Additional arguments for each model.

    Returns:
        model_total_result (dict): A dictionary containing the results of the models run, with keys as model names and values as their respective results.
        This dictionary can readily be converted to a Pandas Dataframe, as seen in the 'ModelResults.csv' files that are produced in the results directory
        when the main profiler is run (qbiocode-profiler.py).

    """

    # Lazy imports to avoid circular dependency
    # These imports happen inside the function, not at module level
    from qbiocode.learning.compute_dt import compute_dt, compute_dt_opt
    from qbiocode.learning.compute_lr import compute_lr, compute_lr_opt
    from qbiocode.learning.compute_rf import compute_rf, compute_rf_opt
    from qbiocode.learning.compute_mlp import compute_mlp, compute_mlp_opt
    from qbiocode.learning.compute_xgb import compute_xgb, compute_xgb_opt
    from qbiocode.learning.compute_pqk import compute_pqk
    from qbiocode.learning.compute_qpl import compute_qpl
    from qbiocode.learning.compute_qnn import compute_qnn
    from qbiocode.learning.compute_qsvc import compute_qsvc
    from qbiocode.learning.compute_nb import compute_nb, compute_nb_opt
    from qbiocode.learning.compute_svc import compute_svc, compute_svc_opt
    from qbiocode.learning.compute_vqc import compute_vqc
    
    # Build model dictionary
    compute_ml_dict = {
        "svc_opt": compute_svc_opt,
        "svc": compute_svc,
        "dt_opt": compute_dt_opt,
        "dt": compute_dt,
        "lr_opt": compute_lr_opt,
        "lr": compute_lr,
        "nb_opt": compute_nb_opt,
        "nb": compute_nb,
        "rf_opt": compute_rf_opt,
        "rf": compute_rf,
        "xgb_opt": compute_xgb_opt,
        "xgb": compute_xgb,
        "mlp_opt": compute_mlp_opt,
        "mlp": compute_mlp,
        "qsvc": compute_qsvc,
        "vqc": compute_vqc,
        "qnn": compute_qnn,
        "pqk": compute_pqk,
        "qpl": compute_qpl,

    }

    # Quantum models don't have _opt versions (use separate configs for hyperparameter tuning)
    quantum_models = {"qsvc", "qnn", "vqc", "pqk", "qpl"}

    # Validate the requested models before dispatching. An unknown name otherwise
    # reached `compute_ml_dict[method]` inside a joblib worker and came back as a
    # bare KeyError with no indication of what the valid names are.
    requested = list(args["model"])
    if not requested:
        raise ValueError(
            "args['model'] is empty; there is nothing to run. Choose at least one "
            f"of {sorted(compute_ml_dict)}."
        )
    unknown = [m for m in requested if m not in compute_ml_dict]
    if unknown:
        raise ValueError(
            f"Unknown model(s) {unknown} in args['model']. Available models: "
            f"{sorted(compute_ml_dict)} (quantum: {sorted(quantum_models)}). "
            f"Note the '_opt' variants are selected with args['grid_search'], not "
            f"by naming them here."
        )
    if grid_search_requested := bool(args.get("grid_search", False)):
        missing_opt = [
            m for m in requested
            if m not in quantum_models and (m + "_opt") not in compute_ml_dict
        ]
        if missing_opt:
            raise ValueError(
                f"grid_search is enabled but {missing_opt} have no '_opt' "
                f"implementation. Disable grid_search or drop those models."
            )
    del grid_search_requested

    # Run classical and quantum models
    n_jobs = len(args["model"])
    if "n_jobs" in args.keys():
        n_jobs = min(args["n_jobs"], len(args["model"]))

    grid_search = False
    if "grid_search" in args.keys():
        grid_search = args["grid_search"]

    # Check if any quantum models are in the model list when grid_search is enabled
    if grid_search:
        quantum_in_models = [m for m in args["model"] if m in quantum_models]
        if quantum_in_models:
            print("\n" + "=" * 80)
            print("WARNING: Grid search is enabled with quantum models:", quantum_in_models)
            print("=" * 80)
            print("Quantum models do not support automated grid search.")
            print("For hyperparameter tuning of quantum models, you should:")
            print("  1. Create multiple configuration files with different hyperparameters")
            print("  2. Run QProfiler separately for each configuration")
            print("  3. Compare results across runs")
            print("\nUse the config generation utility:")
            print("  from qbiocode.utils import generate_qml_experiment_configs")
            print("  num_configs, _ = generate_qml_experiment_configs(")
            print("      template_config_path='configs/config.yaml',")
            print("      output_dir='configs/qml_gridsearch',")
            print("      data_dirs=['data/your_data_dir']")
            print("  )")
            print("\nSee documentation: qbiocode.utils.generate_qml_experiment_configs")
            print("=" * 80 + "\n")

    def _model_args(method):
        """Per-model hyperparameters from the config, or the estimator defaults.

        `args[method + "_args"]` raised KeyError for any model whose config block
        was absent -- which includes ``xgb`` and ``qpl`` in the shipped
        config.yaml, so naming either in ``model`` failed before the estimator was
        ever constructed. The grid-search branch below already used ``.get(...,
        {})``; this makes the two agree, and says so in the log rather than
        substituting silently.
        """
        key = method + "_args"
        if key in args:
            return args[key]
        logger.info(
            "No %r block in the config; running %r with its default "
            "hyperparameters.", key, method,
        )
        return {}

    if grid_search:
        results = []
        for method in args["model"]:
            if method in quantum_models:
                # Quantum models don't have _opt versions, use regular function
                result = delayed(compute_ml_dict[method])(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    args,
                    model=method,
                    data_key=data_key,
                    **args.get(method + "_args", {}),
                    verbose=False,
                )
            else:
                # Classical models have _opt versions with grid search
                result = delayed(compute_ml_dict[method + "_opt"])(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    args,
                    model=method + "_opt",
                    cv=args["cross_validation"],
                    **args.get("gridsearch_" + method + "_args", {}),
                    verbose=False,
                )
            results.append(result)
        results = Parallel(n_jobs=n_jobs)(results)
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_ml_dict[method])(
                X_train,
                X_test,
                y_train,
                y_test,
                args,
                model=method,
                data_key=data_key,
                **_model_args(method),
                verbose=False,
            )
            for method in args["model"]
        )

    model_total_result = pd.melt(pd.concat(results)).dropna()  # type: ignore
    model_total_result["i"] = 0
    model_total_result = model_total_result.pivot(columns="variable", values="value", index="i")
    return model_total_result.to_dict()
