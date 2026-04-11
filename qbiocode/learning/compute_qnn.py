# ====== Base class imports ======
import time
from typing import Literal

# from qiskit.primitives import Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms.utils import algorithm_globals

# ====== Qiskit imports ======
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.circuit.library import qnn_circuit as QNNCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN

import qbiocode.utils.qutils as qutils

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import modeleval


def compute_qnn(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    model="QNN",
    data_key="",
    primitive: Literal["estimator", "sampler"] = "sampler",
    verbose=False,
    local_optimizer: Literal["COBYLA", "L_BFGS_B", "GradientDescent"] = "COBYLA",
    maxiter=100,
    encoding="Z",
    entanglement="linear",
    reps=2,
    ansatz_type="amp",
):
    """
    This function computes a Quantum Neural Network (QNN) model on the provided training data and evaluates it on the test data.
    It constructs a QNN circuit with a specified feature map and ansatz, optimizes it using a chosen optimizer, and fits the model to the training data.
    It then predicts the labels for the test data and evaluates the model's performance.
    The function returns the performance results, including accuracy, F1-score, AUC, runtime, as well as model parameters, and other relevant metrics.

    Args:
        X_train (array-like): Training feature set.
        X_test (array-like): Test feature set.
        y_train (array-like): Training labels.
        y_test (array-like): Test labels.
        args (dict): Dictionary containing configuration parameters for the QNN.
        model (str, optional): Model type. Defaults to 'QNN'.
        data_key (str, optional): Key for the dataset. Defaults to ''.
        primitive (Literal['estimator', 'sampler'], optional): Type of primitive to use. Defaults to 'sampler'.
        verbose (bool, optional): If True, prints additional information. Defaults to False.
        local_optimizer (Literal['COBYLA', 'L_BFGS_B', 'GradientDescent'], optional): Optimizer to use. Defaults to 'COBYLA'.
        maxiter (int, optional): Maximum number of iterations for the optimizer. Defaults to 100.
        encoding (str, optional): Feature encoding method. Defaults to 'Z'.
        entanglement (str, optional): Entanglement strategy for the circuit. Defaults to 'linear'.
        reps (int, optional): Number of repetitions for the feature map and ansatz. Defaults to 2.
        ansatz_type (str, optional): Type of ansatz to use. Defaults to 'amp'.

    Returns:
        modeleval (dict): A dictionary containing the evaluation results, including accuracy, runtime, model parameters, and other relevant metrics.
    """
    beg_time = time.time()

    # choose a method for mapping your features onto the circuit
    feature_map, _ = qutils.get_feature_map(
        feature_map=encoding, feat_dimension=X_train.shape[1], reps=reps, entanglement=entanglement
    )

    # get ansatz
    ansatz = qutils.get_ansatz(
        ansatz_type=ansatz_type,
        feat_dimension=feature_map.num_qubits,
        reps=reps,
        entanglement=entanglement,
    )

    #  Generate the backend, session and primitive
    backend, session, prim = qutils.get_backend_session(
        args, primitive, num_qubits=feature_map.num_qubits
    )

    # Get Optimizer
    optimizer = qutils.get_optimizer(local_optimizer, max_iter=maxiter)

    # qc, input_params, weight_params = QNNCircuit(num_qubits=X_train.shape[1], feature_map=feature_map, ansatz=ansatz)
    qc, _, _ = QNNCircuit(num_qubits=X_train.shape[1], feature_map=feature_map, ansatz=ansatz)

    print(f"Currently running a quantum neural network (QNN) on this dataset.")
    print(f"The number of qubits in your circuit is: {feature_map.num_qubits}")
    print(f"The number of parameters in your circuit is: {feature_map.num_parameters}")
    print(f"The number of ansatz parameters in your circuit is: {ansatz.num_parameters}")

    neural_network: EstimatorQNN | SamplerQNN

    if primitive == "estimator":
        if args["backend"] == "simulator":
            neural_network = EstimatorQNN(
                circuit=qc, input_params=feature_map.parameters, weight_params=ansatz.parameters
            )
        else:
            pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
            neural_network = EstimatorQNN(
                circuit=qc,
                estimator=prim,
                pass_manager=pm,
                input_params=feature_map.parameters,
                weight_params=ansatz.parameters,
            )

        # QNN maps inputs to [-1, +1]
        neural_network.forward(
            X_train[0, :], algorithm_globals.random.random(neural_network.num_weights)
        )
    else:
        # sampler=Sampler(backend=backend)
        # parity maps bitstrings to 0 or 1
        def parity(x):
            return "{:b}".format(x).count("1") % 2

        output_shape = (
            2  # corresponds to the number of classes, possible outcomes of the (parity) mapping
        )
        # construct QNN
        if "simulator" in args["backend"]:
            neural_network = SamplerQNN(
                circuit=qc,
                interpret=parity,
                output_shape=output_shape,
                input_params=feature_map.parameters,
                weight_params=ansatz.parameters,
            )
        else:
            pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
            neural_network = SamplerQNN(
                circuit=qc,
                sampler=prim,
                interpret=parity,
                output_shape=output_shape,
                pass_manager=pm,
                input_params=feature_map.parameters,
                weight_params=ansatz.parameters,
            )

    # construct classifier
    qnn = NeuralNetworkClassifier(neural_network=neural_network, optimizer=optimizer)

    # fit classifier to data
    model_fit = qnn.fit(X_train, y_train)
    hyperparameters = {
        "feature_map": feature_map.__class__.__name__,
        "ansatz": ansatz.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "optimizer_params": optimizer.settings,
        # Add other hyperparameters as needed
    }
    model_params = hyperparameters
    y_predicted = qnn.predict(X_test)

    if not isinstance(session, type(None)):
        session.close()

    return modeleval(
        y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose
    )
