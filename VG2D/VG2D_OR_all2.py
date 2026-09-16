import networkx as nx

def VG2D_OR(series):
    n = len(series)
    g = nx.Graph()
    # nodes
    for i, (x, y) in enumerate(series, start=1):
        g.add_node(i, mag=(x, y))

    # neighbor edges always    
    for i in range(1, n):
        g.add_edge(i, i + 1)

    # VG-OR visibility checks
    for i in range(1, n + 1):
        xi, yi = series[i - 1]
        for j in range(i + 2, n + 1):
            xj, yj = series[j - 1]

            connect = all(
                (series[k - 1][0] < (xi + (xj - xi) * ((k - i) / (j - i)))) or
                (series[k - 1][1] < (yi + (yj - yi) * ((k - i) / (j - i))))
                for k in range(i + 1, j)
            )

            if connect:
                g.add_edge(i, j)

    return g
