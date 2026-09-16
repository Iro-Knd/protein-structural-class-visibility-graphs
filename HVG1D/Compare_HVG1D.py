import pandas as pd
import csv

def min_mean_square_error(actual_values, predicted_values):
    digits = 4
    if len(actual_values) != len(predicted_values):
        raise ValueError("Length of actual and predicted values lists must be equal.")
    squared_errors = [(float(actual) - float(predicted)) ** 2 for actual, predicted in zip(actual_values, predicted_values)]
    mean_square_error = sum(squared_errors) / len(actual_values)
    return round(mean_square_error, digits)

# Initialize variables
results = []

# Loop through each column index
for i in range(17):
    # Read column data from both Excel files
    df_zervou = pd.read_excel('00_Zervou_HVG_1D.xlsx', usecols=[i])
    list_zervou = df_zervou.values.flatten().tolist()

    df_iro = pd.read_excel('101HVG_1D_Results_Parallel.xlsx', usecols=[i])
    list_iro = df_iro.values.flatten().tolist()

    # Remove NaN and normalize types
    list_zervou = [float(x) for x in list_zervou if pd.notna(x)]
    list_iro = [float(x) for x in list_iro if pd.notna(x)]

    # Compute MMSE for the current column
    mmse = min_mean_square_error(list_zervou, list_iro)
    results.append(mmse)

# Define headers
headers = ['Max Value', 'Average Shortest Path Length', 'Diameter', 'Clustering Coefficient',
           'Energy', 'Laplacian Energy', 'Pearson correlation coefficient',
           'Average Closeness Centrality', 'Max Value', 'Average Shortest Path Length', 'Diameter', 'Clustering Coefficient',
           'Energy', 'Laplacian Energy', 'Pearson correlation coefficient',
           'Average Closeness Centrality','Number of nodes']

# Save results to excel
output_file = '101MMSE_Results_HVG1D_parallel.xlsx'
df_out = pd.DataFrame([results], columns=headers)
df_out.to_excel('101MMSE_Results_HVG1D_parallel.xlsx', index=False)

print(f"Results saved to {output_file}")
