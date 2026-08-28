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

import numpy as np
from qbiocode.apps.quvine._deps import require_module

# The node2vec distribution is provided by the [quvine] extra. Note that it
# imports the deprecated pkg_resources module, so the extra pins
# setuptools<81 -- without that pin this import fails even when node2vec
# itself is installed.
Node2Vec = require_module("node2vec", method="node2vec").Node2Vec


def run_node2vec(
    graph,
    nodes,
    dimensions=64,
    walk_length=10,
    num_walks=10,
    p=1.0,
    q=0.5,
    window=5,
    min_count=1,
    workers=8,
    seed=None,
):
    """
    Run Node2Vec and return embeddings aligned to `nodes`.

    Parameters
    ----------
    graph : networkx.Graph
    nodes : List[node]
        Canonical node ordering (must match graph_data.nodes)
    """

    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        seed=seed,
    )

    model = node2vec.fit(
        window=window,
        min_count=min_count,
        batch_words=4,
    )

    # --- align embeddings to nodes ---
    Z = np.zeros((len(nodes), dimensions), dtype=float)

    for i, node in enumerate(nodes):
        Z[i] = model.wv[node]

    return Z
