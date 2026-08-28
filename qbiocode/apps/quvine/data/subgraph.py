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

"""Ego-net (bounded-radius neighbourhood) extraction."""

from typing import Iterable, Set

import networkx as nx


def expand_neighborhood(G: nx.Graph, roots: Iterable, radius: int = 1) -> Set:
    """Return every node within ``radius`` hops of any node in ``roots``.

    The roots themselves are always included, and roots absent from ``G`` are
    ignored rather than raising -- callers sample roots from ``G.nodes()`` but
    may prune the graph afterwards.

    Args:
        G: Graph to expand within. Direction is ignored: for a ``DiGraph`` the
            expansion follows edges either way, because the callers use this to
            build undirected ego-nets for walk scoring.
        roots: Nodes to expand from.
        radius: Maximum hop distance. ``0`` returns just the roots present in
            ``G``.

    Returns:
        The set of reachable nodes, roots included.

    Raises:
        ValueError: if ``radius`` is negative.

    Examples:
        >>> import networkx as nx
        >>> G = nx.path_graph(5)  # 0-1-2-3-4
        >>> sorted(expand_neighborhood(G, {0}, radius=2))
        [0, 1, 2]
        >>> sorted(expand_neighborhood(G, {0, 4}, radius=1))
        [0, 1, 3, 4]
    """
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius!r}.")

    # Undirected view so a DiGraph expands both ways; nx.Graph.to_undirected(as_view)
    # is cheap and avoids copying the graph for every one of the many centers a
    # target-building sweep visits.
    H = G if not G.is_directed() else G.to_undirected(as_view=True)

    reached: Set = {r for r in roots if r in H}
    frontier = set(reached)
    for _ in range(radius):
        if not frontier:
            break
        nxt = {nb for node in frontier for nb in H[node]} - reached
        reached |= nxt
        frontier = nxt
    return reached
