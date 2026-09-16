from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import networkx as nx
import numpy as np
from numpy import linalg as LA
from VG1D import *
from assortativity import *
from HVG1D_Divide_and_Conquer2 import horizontal_visibility_graph
from chaos_try2 import chaos_game_representation

def laplacian_energy(graph):
    """
    Calculation of Laplacian Energy of graph
    Args: graph (nx.Graph): Graph NetworkX
    Returns:float: Graph's Laplacian Energy
    """
    # Laplacian Matrix
    laplacian_matrix = nx.laplacian_matrix(graph).toarray()

    # Laplacian Matrix eigenvalues
    eigenvalues = np.linalg.eigvals(laplacian_matrix)

    # Nodes and Edges
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    # Laplacian Energy
    average_degree = (2 * m) / n
    laplacian_energy = sum(abs(eigenvalue - average_degree) for eigenvalue in eigenvalues)

    return laplacian_energy

def clustering_coefficient(graph):
    """
    Calculates the clustering coefficient according to MATLAB logic.
    Args: graph (nx.Graph): One graph NetworkX.
    Returns: float: clustering coefficient
    """
    # adjacency matrix from the graph
    adjacency_matrix = nx.to_numpy_array(graph)

    # Calculation of triangles for each node
    triangles_per_node = np.diag(adjacency_matrix @ np.triu(adjacency_matrix) @ adjacency_matrix)
    total_triangles = np.sum(triangles_per_node)

    # compute pairs of nodes that have distance 2 (triu(d) == 2)
    distance_matrix = nx.floyd_warshall_numpy(graph)
    pairs_with_distance_2 = np.sum(np.triu(distance_matrix == 2))

    # Clustering Coefficient
    clustering_coefficient = total_triangles / (total_triangles + pairs_with_distance_2)
    return clustering_coefficient

def compute_graph_metrics(graph):
    # Calculating VG graph features
    metrics = {
        'max_degree': max(dict(graph.degree).values()),
        'average_shortest_path_length': nx.average_shortest_path_length(graph) if nx.is_connected(graph) else None,
        'diameter': nx.diameter(graph) if nx.is_connected(graph) else None,
        'clustering_coefficient': clustering_coefficient(graph),
        'energy': sum(abs(LA.eigvals(nx.to_numpy_array(graph)))),
        'laplacian_energy': laplacian_energy(graph),
        'pearson_correlation_coefficient': assortativity_bct_undirected(graph),
        'average_closeness_centrality': np.mean(list(nx.closeness_centrality(graph).values())),
    }
    return metrics


def process_matrix(matrix):
    # X Y values subtraction
    x_values = [pair[0] for pair in matrix]
    y_values = [pair[1] for pair in matrix]

    # Creation VG
    hvg_x = visibility_graph1D(x_values)
    hvg_y = visibility_graph1D(y_values)

    # Feature calculation
    metrics_x = compute_graph_metrics(hvg_x)
    metrics_y = compute_graph_metrics(hvg_y)

    # Number_of_nodes is the same for both graphs
    number_of_nodes = hvg_x.number_of_nodes()

    combined_metrics = {
        'x_' + key: value for key, value in metrics_x.items()
    }
    combined_metrics.update({
        'y_' + key: value for key, value in metrics_y.items()
    })
    combined_metrics['number_of_nodes'] = number_of_nodes

    return combined_metrics


def main():
    # Step 1: Read the strings from the file
    with open("PSI_PRED.txt") as file:
        sequences = [line.strip() for line in file if line.strip()]

    # Step 2: Processing with Chaos Game Representation
    matrices = chaos_game_representation(sequences)

    # Step 3: Parallel processing of each table
    results = Parallel(n_jobs=-1)(
        delayed(process_matrix)(matrix) for matrix in tqdm(matrices, desc="Processing Matrices")
    )

    # Step 4: Creation of DataFrame and stored in Excel file
    headerList = [
        'x_max_degree', 'x_average_shortest_path_length', 'x_diameter',
        'x_clustering_coefficient', 'x_energy', 'x_laplacian_energy',
        'x_pearson_correlation_coefficient', 'x_average_closeness_centrality',
        'y_max_degree', 'y_average_shortest_path_length', 'y_diameter',
        'y_clustering_coefficient', 'y_energy', 'y_laplacian_energy',
        'y_pearson_correlation_coefficient', 'y_average_closeness_centrality',
        'number_of_nodes'
    ]

    # df = pd.DataFrame(results, columns=headerList) # for headers
    df = pd.DataFrame(results)
    df.to_excel("101VG_1D_Results_Parallel.xlsx", index=False, header=False)
    print("Τα αποτελέσματα αποθηκεύτηκαν στο VG_1D_Results_Parallel_exceptions.xlsx")


if __name__ == "__main__":
    main()
