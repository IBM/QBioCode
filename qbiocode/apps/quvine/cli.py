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

"""
Command-line interface for the QuVINE embedding app.

Embeds the nodes of a graph (given as an edge list) with a chosen QuVINE method
and writes the embedding matrix + metadata. Mirrors the qsage/qprofiler CLIs.

Usage:
    quvine --edgelist edges.csv --method quvine_fused --output out/
    quvine --edgelist edges.tsv --sep '\\t' --method node2vec --output out/
    quvine --list-methods
"""

import argparse
import json
import os
import sys


def _load_graph(path: str, sep: str, weighted: bool):
    """Load an undirected graph from a 2- or 3-column edge list (CSV/TSV)."""
    import pandas as pd
    import networkx as nx

    df = pd.read_csv(path, sep=sep, engine="python", header=None, comment="#")
    if df.shape[1] < 2:
        raise ValueError(
            f"Edge list {path!r} must have at least 2 columns (source, target); "
            f"found {df.shape[1]}."
        )
    df = df.rename(columns={0: "source", 1: "target"})
    # QuVINE's corpus builder requires string node tokens (word2vec); coerce so
    # integer-id edge lists work too.
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    if weighted and df.shape[1] >= 3:
        df = df.rename(columns={2: "weight"})
        return nx.from_pandas_edgelist(df, "source", "target", edge_attr="weight")
    return nx.from_pandas_edgelist(df, "source", "target")


def main():
    """CLI entry point for ``quvine``."""
    # `list_methods` resolves through api.aliases, which is stdlib-only, so
    # --help and --list-methods work without the [quvine] extra installed.
    # `embed` is imported at its point of use further down, because it pulls in
    # omegaconf and the rest of the extra.
    from qbiocode.apps.quvine import list_methods

    parser = argparse.ArgumentParser(
        description="QuVINE: quantum view-based network embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  quvine --edgelist edges.csv --method quvine_rwr --output out/
  quvine --edgelist edges.csv --method quvine_fused --output out/
  quvine --list-methods

For more information, see: https://github.com/IBM/QBioCode
""",
    )
    parser.add_argument("--edgelist", "-i", help="Path to a 2/3-column edge-list file (source,target[,weight]).")
    parser.add_argument("--output", "-o", help="Output directory for the embedding + metadata.")
    parser.add_argument(
        "--method", "-m", default="quvine_fused",
        help="Embedding method (default: quvine_fused). See --list-methods.",
    )
    parser.add_argument("--sep", default=",", help="Edge-list column separator (default: ',').")
    parser.add_argument("--weighted", action="store_true", help="Use a 3rd column as edge weight.")
    parser.add_argument("--base-seed", type=int, default=None, help="Override the base random seed.")
    parser.add_argument("--config", default=None, help="Path to a QuVINE config YAML (defaults to packaged config).")
    parser.add_argument("--npy", action="store_true", help="Also write the embedding as a .npy array.")
    parser.add_argument("--list-methods", action="store_true", help="List available methods and exit.")
    args = parser.parse_args()

    if args.list_methods:
        for name in list_methods():
            print(name)
        return

    if not args.edgelist or not args.output:
        parser.error("--edgelist/-i and --output/-o are required (unless --list-methods).")
    if not os.path.exists(args.edgelist):
        print(f"Error: edge list {args.edgelist!r} not found.", file=sys.stderr)
        sys.exit(1)

    # Validate through resolve_method rather than a membership test, so the
    # message carries its close-match suggestions and the accepted set can never
    # drift from what embed() actually dispatches.
    from qbiocode.apps.quvine.api.aliases import resolve_method

    try:
        resolve_method(args.method)
    except KeyError as exc:
        print(
            f"Error: {exc.args[0]} Run 'quvine --list-methods' to see all options.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 72)
    print("QuVINE embedding")
    print("=" * 72)
    print(f"Edge list : {args.edgelist}")
    print(f"Method    : {args.method}")
    print(f"Output    : {args.output}")

    print("\nLoading graph...")
    G = _load_graph(args.edgelist, args.sep, args.weighted)
    print(f"  nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    print("Embedding...")
    # A missing [quvine] dependency is a configuration problem, not a crash:
    # print the actionable message and exit 1 instead of dumping a traceback.
    from qbiocode.apps.quvine._deps import QuvineDependencyError

    try:
        from qbiocode.apps.quvine import embed

        result = embed(G, args.method, config=args.config, base_seed=args.base_seed)
    except QuvineDependencyError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"  method={result.method} dim={result.dim} time={result.execution_time:.2f}s")

    # Write embedding as node,dim_0,...,dim_{d-1}
    import pandas as pd
    emb_df = pd.DataFrame(result.embedding, index=result.node_order)
    emb_df.columns = [f"dim_{i}" for i in range(result.dim)]
    emb_df.index.name = "node"
    csv_path = os.path.join(args.output, "embedding.csv")
    emb_df.to_csv(csv_path)
    print(f"  wrote {csv_path}")

    if args.npy:
        import numpy as np
        npy_path = os.path.join(args.output, "embedding.npy")
        np.save(npy_path, result.embedding)
        print(f"  wrote {npy_path}")

    meta = {
        "requested_method": result.requested_method,
        "method": result.method,
        "kind": result.kind,
        "dim": result.dim,
        "n_nodes": len(result.node_order),
        "execution_time": result.execution_time,
        "used_quantum_targets": result.used_quantum_targets,
    }
    meta_path = os.path.join(args.output, "embedding_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"  wrote {meta_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
