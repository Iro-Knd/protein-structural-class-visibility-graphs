import pandas as pd
import numpy as np

FILE_Z = "00_Zervou_HVG_1D.xlsx"
FILE_I = "101HVG_1D_Results_Parallel.xlsx"

N_COLS = 17
DECIMALS = 4
ROWS = list(range(1, 1674))  # 1..1672 inclusive

METRIC_NAMES = [
    "x_max_degree",
    "x_average_shortest_path_length",
    "x_diameter",
    "x_clustering_coefficient",
    "x_energy",
    "x_laplacian_energy",
    "x_pearson_correlation_coefficient",
    "x_average_closeness_centrality",
    "y_max_degree",
    "y_average_shortest_path_length",
    "y_diameter",
    "y_clustering_coefficient",
    "y_energy",
    "y_laplacian_energy",
    "y_pearson_correlation_coefficient",
    "y_average_closeness_centrality",
    "number_of_nodes",
]

# helper: print + write
def print_and_write(text, file):
    print(text)
    file.write(text + "\n")

df_z = pd.read_excel(FILE_Z).iloc[:, :N_COLS].round(DECIMALS)
df_i = pd.read_excel(FILE_I, decimal=",").iloc[:, :N_COLS].round(DECIMALS)

df_z.columns = METRIC_NAMES
df_i.columns = METRIC_NAMES

n = min(len(df_z), len(df_i))
df_z = df_z.iloc[:n].reset_index(drop=True)
df_i = df_i.iloc[:n].reset_index(drop=True)

with open("101Exceptions_Which_Line_HVG1D_log.txt", "w") as log_file:

    print_and_write(f"Comparing up to {DECIMALS} decimals | rows in common: {n}\n", log_file)

    found = 0

    for r in ROWS:
        if r < 0 or r >= n:
            continue

        a = df_z.iloc[r].to_numpy(dtype=float)
        b = df_i.iloc[r].to_numpy(dtype=float)

        valid = ~np.isnan(a) & ~np.isnan(b)
        diff_mask = valid & (a != b)

        idxs = np.where(diff_mask)[0]
        if len(idxs) == 0:
            continue  # only diffs

        found += 1
        print_and_write("\n" + "=" * 80, log_file)
        print_and_write(f"python_row={r} | excel_row={r+2} | diffs={len(idxs)}", log_file)

        for j in idxs:
            metric = METRIC_NAMES[j]
            print_and_write(
                f"- {metric}: Zervou={a[j]} | Iro={b[j]} | abs_diff={abs(a[j]-b[j])}",
                log_file
            )

    print_and_write(f"\nTotal rows with differences: {found}", log_file)
