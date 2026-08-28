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

__all__ = []

try:
    from qbiocode.apps.quvine.baselines.node2vec import run_node2vec
    __all__.append("run_node2vec")
except ImportError:
    pass

try:
    from qbiocode.apps.quvine.baselines.netmf import run_netmf
    __all__.append("run_netmf")
except ImportError:
    pass

try:
    from qbiocode.apps.quvine.baselines.graphsage import run_graphsage
    __all__.append("run_graphsage")
except ImportError:
    pass

try:
    from qbiocode.apps.quvine.baselines.appnp import run_appnp, generate_appnp_embedding
    __all__.extend(["run_appnp", "generate_appnp_embedding"])
except ImportError:
    pass

try:
    from qbiocode.apps.quvine.baselines.gcn_mf import (
        GCNMF,
        GCNLayer,
        QuVINEGCNMF,
        normalize_adjacency,
        train_gcn_mf,
        precompute_quantum_diffusion,
        generate_baseline_gcnmf_embedding,
        generate_baseline_filter_embedding_wrapper,
    )
    __all__.extend([
        "GCNMF",
        "GCNLayer",
        "QuVINEGCNMF",
        "normalize_adjacency",
        "train_gcn_mf",
        "precompute_quantum_diffusion",
        "generate_baseline_gcnmf_embedding",
        "generate_baseline_filter_embedding_wrapper",
    ])
except ImportError:
    pass

try:
    from qbiocode.apps.quvine.baselines.graphgps import (
        GraphGPSConfig,
        TrainConfig as GraphGPSTrainConfig,
        generate_graphgps_embedding,
        generate_multiple_graphgps_embeddings,
    )
    __all__.extend([
        "GraphGPSConfig",
        "GraphGPSTrainConfig",
        "generate_graphgps_embedding",
        "generate_multiple_graphgps_embeddings",
    ])
except ImportError:
    pass
