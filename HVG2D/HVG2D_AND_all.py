import networkx as nx

def horizontal_visibility_2D_and(series):
    n = len(series)
    g = nx.Graph()

    # nodes
    for i, vec in enumerate(series, start=1):
        g.add_node(i, mag=tuple(vec))

    # neighbor edges always
    for i in range(1, n):
        g.add_edge(i, i + 1)

    # HVG-AND visibility checks
    for i in range(1, n + 1):
        xi, yi = series[i - 1]
        for j in range(i + 2, n + 1):
            xj, yj = series[j - 1]

            min_x = min(xi, xj)
            min_y = min(yi, yj)

            connect = all(
                (series[k - 1][0] < min_x) and (series[k - 1][1] < min_y)
                for k in range(i + 1, j)
            )

            if connect:
                g.add_edge(i, j)

    return g
