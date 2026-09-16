import networkx as nx

def visibility_graph2D_and(series):
    n = len(series)
    g = nx.Graph()

    # nodes
    for i, vec in enumerate(series, start=1):
        g.add_node(i, mag=tuple(vec))

    # neighbor edges always
    for i in range(1, n):
        g.add_edge(i, i + 1)

    # VG-AND visibility checks
    for i in range(1, n + 1):
        yi = series[i - 1]
        for j in range(i + 2, n + 1):
            yj = series[j - 1]

            connect = all(
                (series[k - 1][0] <
                    (yj[0] + (yi[0] - yj[0]) * ((j - k) / (j - i))))
                and
                (series[k - 1][1] <
                    (yj[1] + (yi[1] - yj[1]) * ((j - k) / (j - i))))
                for k in range(i + 1, j)
            )

            if connect:
                g.add_edge(i, j)

    return g
