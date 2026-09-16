import networkx as nx
from itertools import combinations

def visibility_graph1D(series):
    n = len(series)
    g = nx.Graph()

    for i, y in enumerate(series, start=1):
        g.add_node(i, mag=float(y))

    for i in range(1, n):
        g.add_edge(i, i + 1)

    for i in range(1, n + 1):
        yi = series[i - 1]
        for j in range(i + 2, n + 1):
            yj = series[j - 1]

            visible = all(
                series[k - 1] < (
                    yj + (yi - yj) * ((j - k) / (j - i))
                )
                for k in range(i + 1, j)
            )

            if visible:
                g.add_edge(i, j)

    return g
