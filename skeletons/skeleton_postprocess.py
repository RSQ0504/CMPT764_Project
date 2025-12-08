# skeleton_cleaning.py
import numpy as np
from collections import deque

from branch_utils import (
    build_skeleton_graph,
    find_junction_and_endpoint_nodes,
)


def connect_nearby_endpoints(graph, max_gap=0.05):
    """
    Connect endpoint pairs whose Euclidean distance is smaller than max_gap.
    This helps to bridge small gaps in predicted skeletons.

    graph: dict[node -> list[node]]
           node is usually a tuple (x, y, z)
    max_gap: distance threshold for adding a new edge between endpoints
    """
    # Collect current endpoints (degree == 1)
    endpoints = [node for node, nbrs in graph.items() if len(nbrs) == 1]

    # Convert to numpy arrays for distance computation
    endpoints_np = [np.array(p, dtype=float) for p in endpoints]
    n = len(endpoints_np)

    for i in range(n):
        for j in range(i + 1, n):
            p = endpoints_np[i]
            q = endpoints_np[j]
            dist = np.linalg.norm(p - q)
            if dist < max_gap:
                # Add an undirected edge between these endpoints
                u = endpoints[i]
                v = endpoints[j]
                graph[u].append(v)
                graph[v].append(u)

    return graph


def prune_small_components(graph, min_nodes=5, min_length=0.05):
    """
    Remove connected components that are too small (few nodes) or too short
    (sum of edge lengths smaller than min_length).

    graph: dict[node -> list[node]]
    min_nodes: minimum number of nodes required to keep a component
    min_length: minimum total edge length required to keep a component
    """
    visited = set()
    keep_nodes = set()

    for start in graph.keys():
        if start in visited:
            continue

        # BFS to collect one connected component
        comp_nodes = []
        comp_edges = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            u = queue.popleft()
            comp_nodes.append(u)
            for v in graph[u]:
                edge = tuple(sorted((u, v)))
                comp_edges.add(edge)
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        # Compute approximate total length of this component
        comp_length = 0.0
        for (a, b) in comp_edges:
            a_np = np.array(a, dtype=float)
            b_np = np.array(b, dtype=float)
            comp_length += np.linalg.norm(a_np - b_np)

        # Decide whether to keep this component
        if len(comp_nodes) >= min_nodes and comp_length >= min_length:
            keep_nodes.update(comp_nodes)

    # Build pruned graph with only kept nodes
    pruned_graph = {}
    for node, nbrs in graph.items():
        if node not in keep_nodes:
            continue
        new_nbrs = [n for n in nbrs if n in keep_nodes]
        pruned_graph[node] = new_nbrs

    return pruned_graph


def build_clean_graph_from_segments(
    segments,
    max_gap=0.05,
    min_nodes=5,
    min_length=0.05,
):
    """
    Convenience function to build a "clean" skeleton graph from raw segments:

    1) Build graph from segments
    2) Connect nearby endpoints to bridge small gaps
    3) Prune small / short connected components (outliers)

    segments: list of ((x1, y1, z1), (x2, y2, z2))
    max_gap:   see connect_nearby_endpoints
    min_nodes: see prune_small_components
    min_length: see prune_small_components
    """
    # Step 1: build raw graph
    graph = build_skeleton_graph(segments)

    # Step 2: bridge small gaps
    graph = connect_nearby_endpoints(graph, max_gap=max_gap)

    # Step 3: prune small components
    graph = prune_small_components(
        graph,
        min_nodes=min_nodes,
        min_length=min_length,
    )

    return graph


def get_clean_keypoints_from_segments(
    segments,
    max_gap=0.05,
    min_nodes=5,
    min_length=0.05,
):
    """
    High-level helper that returns:
    - a cleaned skeleton graph
    - junction nodes (degree > 2)
    - endpoint nodes (degree == 1)

    This is what you typically want for robust keypoint detection.
    """
    clean_graph = build_clean_graph_from_segments(
        segments,
        max_gap=max_gap,
        min_nodes=min_nodes,
        min_length=min_length,
    )

    junctions, endpoints = find_junction_and_endpoint_nodes(clean_graph)
    return clean_graph, junctions, endpoints
