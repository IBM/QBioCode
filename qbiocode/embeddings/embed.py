# Copyright 2026, IBM Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from functools import reduce

import numpy as np

# ====== Qiskit imports ======
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

# ====== Embedding functions imports ======
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding
from umap import UMAP

import qbiocode.utils.qutils as qutils


def pqk(
    X_train,
    X_test,
    args,
    store=False,
    data_key="",
    encoding="Z",
    data_map=True,
    primitive="estimator",
    entanglement="linear",
    reps=2,
):
    """
    This function generates quantum circuits, computes projections of the data onto these circuits.
    It uses a feature map to encode the data into quantum states and then measures the expectation values
    of Pauli operators to obtain the features.
    This function requires a quantum backend (simulator or real quantum hardware) for execution.
    It supports various configurations such as encoding methods, entanglement strategies, and repetitions
    of the feature map. Optionally the results are saved to files for training and test projections.

    Args:
        X_train (np.ndarray): Training data features.
        X_test (np.ndarray): Test data features.
        args (dict): Arguments containing backend and other configurations.
        store (bool): If true projections are stored, using data_key as indefitier
        data_key (str): Key for the dataset, default is ''.
        encoding (str): Encoding method for the quantum circuit, default is 'Z'.
        data_map (bool): If true ensures that all multiplicative factors of data features inside single qubit gates are 1.0. Not applicable for Hejsemberg feature maps
        primitive (str): Primitive type to use, default is 'estimator'.
        entanglement (str): Entanglement strategy, default is 'linear'.
        reps (int): Number of repetitions for the feature map, default is 2.

    Returns:
        modeleval (dict): A dictionary containing evaluation metrics and model parameters.
    """

    feat_dimension = X_train.shape[1]

    if data_map:
        #  This function ensures that all multiplicative factors of data features inside single qubit gates are 1.0
        def data_map_func(x: np.ndarray) -> float:
            """
            Define a function map from R^n to R.

            Args:
                x: data

            Returns:
                float: the mapped value
            """
            coeff = x[0] / 2 if len(x) == 1 else reduce(lambda m, n: (m * n) / 2, x)
            return float(coeff)

    else:
        data_map_func = None

    # choose a method for mapping your features onto the circuit
    feature_map, _ = qutils.get_feature_map(
        feature_map=encoding,
        feat_dimension=X_train.shape[1],
        reps=reps,
        entanglement=entanglement,
        data_map_func=data_map_func,
    )

    # Build quantum circuit
    circuit = QuantumCircuit(feature_map.num_qubits)
    circuit.compose(feature_map, inplace=True)
    num_qubits = circuit.num_qubits

    #  Generate the backend, session and primitive
    backend, session, prim = qutils.get_backend_session(args, "estimator", num_qubits=num_qubits)
    try:

        # Transpile
        if args["backend"] != "simulator":
            circuit = qutils.transpile_circuit(
                circuit, opt_level=3, backend=backend, PT=True, initial_layout=None
            )

        for f_tr in ["train", "test"]:

            if "train" in f_tr:
                dat = X_train.copy()
            else:
                dat = X_test.copy()

            # Identity operator on all qubits
            id = "I" * feat_dimension

            # We group all commuting observables
            # These groups are the Pauli X, Y and Z operators on individual qubits
            # Apply the circuit layout to the observable if mapped to device
            if args["backend"] != "simulator":
                observables_x = []
                observables_y = []
                observables_z = []
                for i in range(feat_dimension):
                    observables_x.append(
                        Pauli(id[:i] + "X" + id[(i + 1) :]).apply_layout(
                            circuit.layout, num_qubits=backend.num_qubits
                        )
                    )
                    observables_y.append(
                        Pauli(id[:i] + "Y" + id[(i + 1) :]).apply_layout(
                            circuit.layout, num_qubits=backend.num_qubits
                        )
                    )
                    observables_z.append(
                        Pauli(id[:i] + "Z" + id[(i + 1) :]).apply_layout(
                            circuit.layout, num_qubits=backend.num_qubits
                        )
                    )
            else:
                observables_x = [Pauli(id[:i] + "X" + id[(i + 1) :]) for i in range(feat_dimension)]
                observables_y = [Pauli(id[:i] + "Y" + id[(i + 1) :]) for i in range(feat_dimension)]
                observables_z = [Pauli(id[:i] + "Z" + id[(i + 1) :]) for i in range(feat_dimension)]

            # projections[i][j][k] will be the expectation value of the j-th Pauli operator (0: X, 1: Y, 2: Z)
            # of datapoint i on qubit k
            projections = []

            for i in range(len(dat)):

                # Get training sample
                parameters = dat[i]

                # We define the primitive unified blocs (PUBs) consisting of the embedding circuit,
                # set of observables and the circuit parameters
                pub_x = (circuit, observables_x, parameters)
                pub_y = (circuit, observables_y, parameters)
                pub_z = (circuit, observables_z, parameters)

                job = prim.run([pub_x, pub_y, pub_z])
                job_result_x = job.result()[0].data.evs
                job_result_y = job.result()[1].data.evs
                job_result_z = job.result()[2].data.evs

                # Record <X>, <Y> and <Z> on all qubits for the current datapoint
                projections.append([job_result_x, job_result_y, job_result_z])

            if store:
                if not os.path.exists("pqk_projections"):
                    os.makedirs("pqk_projections")

                file_projection = os.path.join(
                    "pqk_projections", "pqk_projection_" + data_key + "_" + f_tr + ".npy"
                )

                np.save(file_projection, projections)

            if "train" in f_tr:
                X_train_prj = np.array(projections.copy()).reshape(len(projections), -1)
            else:
                X_test_prj = np.array(projections.copy()).reshape(len(projections), -1)

    finally:
        if not isinstance(session, type(None)):
            session.close()

    return X_train_prj, X_test_prj


def get_embeddings(embedding: str, X_train, X_test, n_neighbors=30, n_components=None, method=None):
    """This function applies the specified embedding technique to the training and test datasets.

    Args:
        embedding (str): The embedding technique to use. Options are 'none', 'pca', 'nmf', 'lle', 'isomap', 'spectral', or 'umap'.
        X_train (array-like): The training dataset.
        X_test (array-like): The test dataset.
        n_neighbors (int, optional): Number of neighbors for certain embeddings. Defaults to 30.
        n_components (int, optional): Number of components for the embedding. If None, it defaults to the number of features in X_train.
        method (str, optional): Method for Locally Linear Embedding. Defaults to None.

    Returns:
        tuple: Transformed training and test datasets.
    """

    embedding = embedding.lower()

    # Default the embedding width to the feature count. This is the documented behavior, but
    # without it `n_components=None` (the documented default!) reaches the comparison below and
    # raises TypeError: '<=' not supported between 'NoneType' and 'int'.
    if n_components is None:
        n_components = X_train.shape[1]

    valid_modes = ["none", "pca", "lle", "isomap", "spectral", "umap", "nmf"]
    if embedding not in valid_modes:
        raise ValueError(f"Invalid mode: {embedding}. Mode must be one of {valid_modes}")

    assert (
        n_components <= X_train.shape[1]
    ), "number of components greater than number of feature in the dataset"
    if "none" == embedding:
        return X_train, X_test
    else:
        embedding_model = None
        if "pca" == embedding:
            embedding_model = PCA(n_components=n_components)
        elif "nmf" == embedding:
            embedding_model = NMF(n_components=n_components)
        elif "lle" == embedding:
            if method is None:
                embedding_model = LocallyLinearEmbedding(
                    n_neighbors=n_neighbors, n_components=n_components, method="standard"
                )
            else:
                embedding_model = LocallyLinearEmbedding(
                    n_neighbors=n_neighbors, n_components=n_components, method="modified"
                )
        elif "isomap" == embedding:
            embedding_model = Isomap(
                n_neighbors=n_neighbors,
                n_components=n_components,
            )
        elif "umap" == embedding:
            embedding_model = UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
            )

        if "spectral" == embedding:
            # SpectralEmbedding has no out-of-sample `transform`, so it must be fit
            # transductively on the combined train+test rows and then sliced back. Only
            # feature structure is used (no labels), so no label leakage is introduced.
            n_train = X_train.shape[0]
            X_all = np.vstack([X_train, X_test])
            Z = SpectralEmbedding(n_components=n_components, eigen_solver="arpack").fit_transform(X_all)
            return Z[:n_train], Z[n_train:]

        X_train = embedding_model.fit_transform(X_train)
        X_test = embedding_model.transform(X_test)

    return X_train, X_test
