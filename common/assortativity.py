import numpy as np
import networkx as nx


def assortativity_bct_undirected(graph: nx.Graph) -> float:
    """
    MATLAB assortativity(CIJ,0) equivalent (Brain Connectivity Toolbox style).
    - Ignores weights (treats as unweighted)
    - Uses each undirected edge once (like triu(CIJ,1))
    """
    # degrees (undirected)
    deg = dict(graph.degree())

    # edges once
    edges = list(graph.edges())
    K = len(edges)
    if K == 0:
        return np.nan

    degi = np.array([deg[u] for (u, v) in edges], dtype=float)
    degj = np.array([deg[v] for (u, v) in edges], dtype=float)

    term1 = np.sum(degi * degj) / K
    term2 = (np.sum(0.5 * (degi + degj)) / K) ** 2
    term3 = np.sum(0.5 * (degi ** 2 + degj ** 2)) / K

    denom = term3 - term2
    if denom == 0:
        return np.nan

    r = (term1 - term2) / denom
    return float(r)
