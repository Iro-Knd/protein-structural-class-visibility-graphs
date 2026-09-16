from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import networkx as nx
import numpy as np
from numpy import linalg as LA

from HVG2D_OR_all import *
from assortativity import assortativity_bct_undirected
from chaos_try2 import chaos_game_representation

def laplacian_energy(graph: nx.Graph) -> float:
    n = graph.number_of_nodes()
    if n == 0:
        return 0.0

    L = nx.laplacian_matrix(graph).toarray().astype(float)
    eigenvalues = np.linalg.eigvalsh(L)

    m = graph.number_of_edges()
    average_degree = (2.0 * m) / n

    return float(np.sum(np.abs(eigenvalues - average_degree)))

def clustering_coefficient(graph: nx.Graph) -> float:
    adjacency_matrix = nx.to_numpy_array(graph)

    triangles_per_node = np.diag(adjacency_matrix @ np.triu(adjacency_matrix) @ adjacency_matrix)
    total_triangles = float(np.sum(triangles_per_node))

    distance_matrix = nx.floyd_warshall_numpy(graph)
    pairs_with_distance_2 = float(np.sum(np.triu(distance_matrix == 2)))

    denom = total_triangles + pairs_with_distance_2
    return float(total_triangles / denom) if denom else 0.0


def largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """Return the LCC subgraph (copy). If graph has <2 nodes, return as-is."""
    if graph.number_of_nodes() < 2:
        return graph
    if nx.is_connected(graph):
        return graph
    lcc = max(nx.connected_components(graph), key=len)
    return graph.subgraph(lcc).copy()


def matlab_aspl_and_diameter(graph: nx.Graph):
    """
    MATLAB-style:
      d = distances(G,'unweighted')
      ASPL = sum(d(:)) / (n*(n-1))
      Diameter = max(d(:))
    Computed on the LCC to avoid inf on disconnected graphs.
    """
    Gc = largest_connected_component(graph)
    n = Gc.number_of_nodes()
    if n < 2:
        return 0.0, 0.0

    dist = np.asarray(nx.floyd_warshall_numpy(Gc), dtype=float)
    aspl = float(dist.sum() / (n * (n - 1)))
    diam = float(np.max(dist))
    return aspl, diam


def compute_graph_metrics(graph: nx.Graph) -> dict:
    assert nx.is_connected(graph), f"Disconnected graph with {graph.number_of_nodes()} nodes"

    degrees = dict(graph.degree)
    max_degree = max(degrees.values()) if degrees else 0

    # MATLAB-style ASPL & diameter (on LCC)
    aspl, diam = matlab_aspl_and_diameter(graph)

    metrics = {
        "max_degree": max_degree,
        "average_shortest_path_length": nx.average_shortest_path_length(graph),
        "diameter": nx.diameter(graph),
        "clustering_coefficient": clustering_coefficient(graph),
        "energy": float(np.sum(np.abs(np.linalg.eigvalsh(nx.to_numpy_array(graph, dtype=float))))),
        "laplacian_energy": laplacian_energy(graph),
        "pearson_correlation_coefficient": assortativity_bct_undirected(graph),
        "average_closeness_centrality": float(np.mean(list(nx.closeness_centrality(graph).values()))),
        "number_of_nodes": graph.number_of_nodes(),
    }
    return metrics


def process_matrix(matrix):
    # x_values = [p[0] for p in matrix]
    # y_values = [p[1] for p in matrix]
    #
    # g = intersection_vg_1d(x_values, y_values)
    g2d = horizontal_visibility_2D_or(matrix)
    return compute_graph_metrics(g2d)


def main():
    with open("PSI_PRED.txt", "r", encoding="utf-8") as file:
        sequences = [line.strip() for line in file if line.strip()]

    matrices = chaos_game_representation(sequences)

    results = Parallel(n_jobs=-1)(
        delayed(process_matrix)(matrix)
        for matrix in tqdm(matrices, desc="Processing Matrices")
    )

    header_list = [
        "max_degree",
        "average_shortest_path_length",
        "diameter",
        "clustering_coefficient",
        "energy",
        "laplacian_energy",
        "pearson_correlation_coefficient",
        "average_closeness_centrality",
        "number_of_nodes",
    ]

    df = pd.DataFrame(results).reindex(columns=header_list)

    out_file = "104HVG_2D_OR_Results_Parallel.xlsx"
    df.to_excel(out_file, index=False, header=False)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
