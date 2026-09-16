import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

# Base coordinates for the triangle
BASE_COORDINATES = {
    "H": [0, 0],
    "C": [1, math.sqrt(3)],
    "E": [2, 0]
}


def calculate_centroid():
    # calculate the centroid of the triangle formed by H, C, and E
    x_coords = [coord[0] for coord in BASE_COORDINATES.values()]
    y_coords = [coord[1] for coord in BASE_COORDINATES.values()]
    centroid_x = sum(x_coords) / 3
    centroid_y = sum(y_coords) / 3
    return centroid_x, centroid_y


def chaos_game_representation(sequence_list):
    """
    Generate CGR points for a list of sequences.
    Args: list of str: List of sequences containing 'H', 'C', 'E'.
    Returns: list of np.array: List of coordinate arrays for each sequence.
    """
    centroid_x, centroid_y = calculate_centroid()
    cgr_results = []

    for sequence in sequence_list:
        pos_x, pos_y = centroid_x, centroid_y
        coordinates = []

        for char in sequence:
            if char in BASE_COORDINATES:
                target_x, target_y = BASE_COORDINATES[char]
                pos_x = (target_x + pos_x) / 2
                pos_y = (target_y + pos_y) / 2
                coordinates.append([pos_x, pos_y])

        cgr_results.append(np.array(coordinates))
    return cgr_results


def plot_chaos_game(sequence_list):
    x_coords = [coord[0] for coord in BASE_COORDINATES.values()] + [BASE_COORDINATES["H"][0]]
    y_coords = [coord[1] for coord in BASE_COORDINATES.values()] + [BASE_COORDINATES["H"][1]]
    centroid_x, centroid_y = calculate_centroid()

    # Plot the triangle
    plt.plot(x_coords, y_coords, color="black", label="Triangle")
    plt.scatter(centroid_x, centroid_y, color="purple", label="Centroid", zorder=5)

    # Label the vertices
    for base, (x, y) in BASE_COORDINATES.items():
        plt.annotate(base, (x, y), color='red', fontsize=12, fontweight='bold')

    # Generate and plot CGR points
    cgr_results = chaos_game_representation(sequence_list)
    for idx, coords in enumerate(cgr_results):
        n = len(coords)
        plt.scatter(centroid_x, centroid_y, marker='x', color='black', s=40, zorder=6, label='Start (centroid)')
        plt.scatter(coords[:, 0], coords[:, 1], s=1, label=f"Sequence {idx + 1}")

    # Add plot details
    # plt.colorbar(sc, label="Position in sequence")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Chaos Game Representation')
    # plt.legend()
    plt.grid(False)
    plt.gca().set_aspect('equal')
    plt.show()

def plot_chaos_game(sequence_list, show_steps=True):
    """
    Plot the CGR triangle together with the generated points and the
    trajectory (direction) for each sequence.
    Args:
        sequence_list (list of str): sequences containing 'H', 'C', 'E'.
        show_steps (bool): if True, annotate each point with its position number.
    """
    centroid_x, centroid_y = calculate_centroid()

    # triangle outline (H -> C -> E -> H)
    tri_x = [coord[0] for coord in BASE_COORDINATES.values()] + [BASE_COORDINATES["H"][0]]
    tri_y = [coord[1] for coord in BASE_COORDINATES.values()] + [BASE_COORDINATES["H"][1]]

    fig, ax = plt.subplots(figsize=(7, 6.4))
    ax.plot(tri_x, tri_y, color="black", lw=1.6, zorder=2)

    cgr_results = chaos_game_representation(sequence_list)

    for coords in cgr_results:
        # trajectory centroid -> p1 -> ... -> pn, colored by step
        path = np.vstack([[centroid_x, centroid_y], coords])
        segs = np.stack([path[:-1], path[1:]], axis=1)
        lc = LineCollection(segs, cmap="plasma",
                            array=np.linspace(0, 1, len(segs)),
                            linewidths=1.3, alpha=0.55, zorder=3)
        ax.add_collection(lc)

        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=np.arange(1, len(coords) + 1), cmap="plasma",
                        s=70, zorder=4, edgecolor="white", linewidth=0.6)

        if show_steps:
            for i, (x, y) in enumerate(coords, start=1):
                ax.annotate(str(i), (x, y), textcoords="offset points",
                            xytext=(6, 5), fontsize=8, color="#333333", zorder=6)

    # start (centroid) as X
    ax.scatter([centroid_x], [centroid_y], marker="X", s=130,
               color="black", zorder=7)
    ax.annotate("start (centroid)", (centroid_x, centroid_y),
                textcoords="offset points", xytext=(8, -22),
                fontsize=8.5, color="black", zorder=7)

    # vertex labels pushed slightly OUTSIDE the triangle, with white bbox,
    # so points piling up at a vertex never hide the label
    offset = {"H": (-0.17, -0.15), "C": (0.0, 0.11), "E": (0.13, -0.07)}
    for base, (x, y) in BASE_COORDINATES.items():
        dx, dy = offset[base]
        ax.annotate(base, (x, y), xytext=(x + dx, y + dy), textcoords="data",
                    color="crimson", fontsize=15, fontweight="bold", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.75))

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Position in sequence", fontsize=10)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Chaos Game Representation", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(-0.35, 2.05)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_signals_x_y(sequence_list):
    """
    Plot the two coordinate signals (s_x and s_y) for each sequence.
    Args: list of str: sequences containing 'H', 'C', 'E'.
    """
    cgr_results = chaos_game_representation(sequence_list)

    for idx, coords in enumerate(cgr_results):
        tx = coords[:, 0]
        ty = coords[:, 1]
        position = range(1, len(coords) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # x signal
        axes[0].plot(position, tx, '-o', color='#6A4C93', markersize=4)
        axes[0].set_title('Time series t_x (x-coordinates)')
        axes[0].set_xlabel('Position in sequence')
        axes[0].set_ylabel('t_x')
        axes[0].grid(True)

        # y signal
        axes[1].plot(position, ty, '-o', color='#2A9D8F', markersize=4)
        axes[1].set_title('Time series t_y (y-coordinates)')
        axes[1].set_xlabel('Position in sequence')
        axes[1].set_ylabel('t_y')
        axes[1].grid(True)

        # show the H/C/E symbols on the x-axis
        for ax in axes:
            ax.set_xticks(position)
            ax.set_xticklabels(list(sequence_list[idx]), fontsize=8)
        plt.tight_layout()
        plt.show()

# series = ['CCCCCHHCCHHHHHHHHCCCCCCCCHHHCCCHHHHHCCCHHHHHHHHHHCCHHHHHHHHHHHHHHHHHCCCCEEEEECCCHHHHHHHHHHCCCCEEEEEECHHHHHHHHHHHHHCCCCCEEEEEECCEEECCCCCCCCCEEEECCHHHHCCCHHHHHHHHHHHHHHCCCCCEEECCCCEEEEEECCCCHHHHHCCCCCCCCCCCCCCCCCCCCCCCCEEEECCHHHHCCCCEEEEEEECCCCCCCCCCEEEEEEEEEECCCCEEEEEEEEEEEEECCCCCEEEEECCCCCCCCCEEEEEECCCCCCCCCCCEEEEEEEEEECCCCCEEEEEEEEEEECCCCCCCCCCCCCCCC']
# with open("PSI_PRED.txt") as f:
#     series = [line.strip() for line in f if line.strip()]
# points = chaos_game_representation(series)


# series = ['ECCHHHHHHCCEEECC']

# print("New Chaos: ", points)
# print(type(points))
# plot_chaos_game(series)
# plot_chaos_game(series, show_steps=True)
# plot_signals_x_y(series)
