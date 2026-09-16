import pandas as pd
import csv

def min_mean_square_error(actual_values, predicted_values, digits=4):
    if len(actual_values) != len(predicted_values):
        raise ValueError("Length of actual and predicted values lists must be equal.")
    squared_errors = [
        (float(actual) - float(predicted)) ** 2
        for actual, predicted in zip(actual_values, predicted_values)
    ]
    mean_square_error = sum(squared_errors) / len(actual_values)
    return round(mean_square_error, digits)


# ---------------------------------
# CONFIG
# ---------------------------------
FILE_Z = '00_Zervou_VG_1D.xlsx'
FILE_I = '101VG_1D_Results_Parallel.xlsx'
N_COLS = 17
DIGITS = 4

HEADERS = [
    'Max Value',
    'Average Shortest Path Length',
    'Diameter',
    'Clustering Coefficient',
    'Energy',
    'Laplacian Energy',
    'Pearson correlation coefficient',
    'Average Closeness Centrality',
    'Max Value',
    'Average Shortest Path Length',
    'Diameter',
    'Clustering Coefficient',
    'Energy',
    'Laplacian Energy',
    'Pearson correlation coefficient',
    'Average Closeness Centrality',
    'Number of nodes'
]

# ---------------------------------
# MMSE COMPUTATION
# ---------------------------------
results = []

print("\nMMSE per metric")
print("=" * 60)

for i in range(N_COLS):
    # Read column data
    df_z = pd.read_excel(FILE_Z, usecols=[i])
    df_i = pd.read_excel(FILE_I, usecols=[i])

    list_z = [float(x) for x in df_z.values.flatten() if pd.notna(x)]
    list_i = [float(x) for x in df_i.values.flatten() if pd.notna(x)]

    mmse = min_mean_square_error(list_z, list_i, DIGITS)
    results.append(mmse)

    # ---- PRINT TO CONSOLE ----
    marker = " <-- !!!" if mmse != 0 else ""
    print(f"[{i:02d}] {HEADERS[i]:35s} : MMSE = {mmse}{marker}")

print("=" * 60)

headers = ['Max Value', 'Average Shortest Path Length', 'Diameter', 'Clustering Coefficient',
           'Energy', 'Laplacian Energy', 'Pearson correlation coefficient',
           'Average Closeness Centrality', 'Max Value', 'Average Shortest Path Length', 'Diameter', 'Clustering Coefficient',
           'Energy', 'Laplacian Energy', 'Pearson correlation coefficient',
           'Average Closeness Centrality','Number of nodes']

output_file = '101MMSE_Results_VG1D_parallel.xlsx'
df_out = pd.DataFrame([results], columns=headers)
df_out.to_excel('101MMSE_Results_VG1D_parallel.xlsx', index=False)

print(f"\nResults saved to {output_file}")
