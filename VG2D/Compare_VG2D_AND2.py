import pandas as pd

def mse(actual_values, predicted_values, digits=4):
    if len(actual_values) != len(predicted_values):
        raise ValueError("Length of actual and predicted values lists must be equal.")
    squared_errors = [(float(a) - float(p)) ** 2 for a, p in zip(actual_values, predicted_values)]
    return round(sum(squared_errors) / len(squared_errors), digits)


# ---------------------------------
# CONFIG
# ---------------------------------
FILE_Z = "00_Zervou_VG_2D_AND.xlsx"
FILE_I = "104VG_2D_AND_Results_Parallel.xlsx"
N_COLS = 9
DIGITS = 4

# Optional pretty names (must match N_COLS)
HEADERS = [
    "Max Degree",
    "Average Shortest Path Length",
    "Diameter",
    "Clustering Coefficient",
    "Energy",
    "Laplacian Energy",
    "Pearson correlation coefficient",
    "Average Closeness Centrality",
    "Number of nodes",
][:N_COLS]


# ---------------------------------
# MSE COMPUTATION
# ---------------------------------
results = []

print("\nMSE per metric (2D, no headers)")
print("=" * 70)

for i in range(N_COLS):
    # read one column from each file
    df_z = pd.read_excel(FILE_Z, header=None, usecols=[i])
    df_i = pd.read_excel(FILE_I, header=None, usecols=[i])

    # align row-by-row and drop pairs where either is NaN/None/non-numeric
    pair = pd.concat([df_z.rename(columns={i: "z"}), df_i.rename(columns={i: "i"})], axis=1)
    pair["z"] = pd.to_numeric(pair["z"], errors="coerce")
    pair["i"] = pd.to_numeric(pair["i"], errors="coerce")
    pair = pair.dropna(subset=["z", "i"])

    if pair.empty:
        mmse_val = None
        marker = " <-- !!!"
        n_used = 0
    else:
        mmse_val = mse(pair["z"].tolist(), pair["i"].tolist(), DIGITS)
        marker = " <-- !!!" if mmse_val != 0 else ""
        n_used = len(pair)

    results.append(mmse_val)
    name = HEADERS[i] if i < len(HEADERS) else f"col_{i}"
    print(f"[{i:02d}] {name:35s} : MSE = {mmse_val}{marker} (n={n_used})")

print("=" * 70)

# save
out_file = "104MMSE_Results_VG2D_AND.xlsx"
df_out = pd.DataFrame([results], columns=HEADERS if len(HEADERS) == N_COLS else None)
df_out.to_excel(out_file, index=False, header=(len(HEADERS) == N_COLS))

print(f"\nResults saved to {out_file}")
