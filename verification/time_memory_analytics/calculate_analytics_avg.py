import pandas as pd

def calculate_averages_and_stds(input_file, output_file):
    df = pd.read_csv(input_file)

    # Calculate averages and standard deviations only for numeric columns.
    averages = df.mean(numeric_only=True).round(2)
    stds = df.std(numeric_only=True).round(2)

    results_df = pd.DataFrame()

    # Append averages and standard deviations for only the numeric columns.
    for col in averages.index:
        results_df[str(col) + '_avg'] = [averages[col]]
        results_df[str(col) + '_std'] = [stds[col]]

    results_df.to_csv(output_file, index=False)

input_csv = 'analytics.csv'  # The file should be in the same directory
output_csv = 'time_avg_std.csv'  # The result will be saved in the same directory

calculate_averages_and_stds(input_csv, output_csv)
