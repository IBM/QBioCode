import time
from typing import Literal

import numpy as np
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# from qiskit.primitives import Sampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from sklearn.model_selection import GridSearchCV

import qbiocode.utils.qutils as qutils

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import modeleval

# ====== Scikit-learn imports ======


# ====== Qiskit imports ======


def compute_qsvc(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    model="QSVC",
    data_key="",
    C=1,
    gamma="scale",
    pegasos=False,
    encoding: Literal["ZZ", "Z", "P"] = "ZZ",
    entanglement="linear",
    primitive="sampler",
    reps=2,
    verbose=False,
    local_optimizer="",
):
    """
    This function computes a quantum support vector classifier (QSVC) using the Qiskit Machine Learning library.
    It takes training and testing datasets, along with various parameters to configure the QSVC model.
    It initializes the quantum feature map, sets up the backend and session, and fits the QSVC model to the training data.
    It then predicts the labels for the test data and evaluates the model's performance.
    The function returns the performance results, including accuracy, F1-score, AUC, runtime, as well as model parameters, and other relevant metrics.

    Args:
        X_train (np.ndarray): Training feature set.
        X_test (np.ndarray): Testing feature set.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
        args (dict): Dictionary containing arguments for the quantum backend and other settings.
        model (str): Model type, default is 'QSVC'.
        data_key (str): Key for the dataset, default is an empty string.
        C (float): Regularization parameter for the SVM, default is 1.
        gamma (str or float): Kernel coefficient, default is 'scale'.
        pegasos (bool): Whether to use Pegasos QSVC, default is False.
        encoding (str): Feature map encoding type, options are 'ZZ', 'Z', or 'P', default is 'ZZ'.
        entanglement (str): Entanglement strategy for the feature map, default is 'linear'.
        primitive (str): Primitive type to use, default is 'sampler'.
        reps (int): Number of repetitions for the feature map, default is 2.
        verbose (bool): Whether to print additional information, default is False.

    Returns:
        modeleval (dict): A dictionary containing the evaluation results, including accuracy, runtime, model parameters, and other relevant metrics.
    """
    beg_time = time.time()

    # choose a method for mapping your features onto the circuit
    feature_map, _ = qutils.get_feature_map(
        feature_map=encoding, feat_dimension=X_train.shape[1], reps=reps, entanglement=entanglement
    )

    #  Generate the backend, session and primitive
    backend, session, prim = qutils.get_backend_session(
        args, primitive, num_qubits=feature_map.num_qubits
    )

    print(f"Currently running a quantum support vector classifier (QSVC) on this dataset.")
    print(f"The number of qubits in your circuit is: {feature_map.num_qubits}")
    print(f"The number of parameters in your circuit is: {feature_map.num_parameters}")

    if "simulator" == args["backend"]:
        fidelity = ComputeUncompute(sampler=prim)
    else:
        # Need to instatiate a basic pass manager to store the chosen hardware backend
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        fidelity = ComputeUncompute(
            sampler=prim, pass_manager=pm
        )  # , num_virtual_qubits = feature_map.num_qubits )

    Qkernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
    if pegasos == True:
        qsvc = PegasosQSVC(C=C, quantum_kernel=Qkernel)
    else:
        qsvc = QSVC(C=C, gamma=gamma, quantum_kernel=Qkernel)

    model_fit = qsvc.fit(X_train, y_train)
    # model_params = model_fit.get_params()
    hyperparameters = {
        "feature_map": feature_map.__class__.__name__,
        "quantum_kernel": Qkernel.__class__.__name__,
        "C": C,
        "gamma": gamma,
    }
    model_params = hyperparameters
    y_predicted = qsvc.predict(X_test)

    if not isinstance(session, type(None)):
        session.close()

    return modeleval(
        y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose
    )
