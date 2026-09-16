import networkx as nx

def hvg_1d(series):
    n = len(series)
    g = nx.Graph()

    # add all nodes first
    for i, y in enumerate(series):
        g.add_node(i, mag=float(y))

    # neighbors always visible
    for i in range(n - 1):
        g.add_edge(i, i + 1)

    # non-neighbor visibility (strict HVG)
    for i in range(n):
        yi = series[i]
        for j in range(i + 2, n):
            yj = series[j]
            ymin = min(yi, yj)

            visible = all(series[k] < ymin for k in range(i + 1, j))
            if visible:
                g.add_edge(i, j)

    return g
