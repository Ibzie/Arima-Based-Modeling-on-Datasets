import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_and_preprocess_csv(folder_path, file_name, output_file=None):
    # Construct full file path
    file_path = os.path.join(folder_path, file_name)

    # Read CSV into DataFrame
    df = pd.read_csv(file_path)

    # Drop rows with null values
    df.dropna(inplace=True)

    # Convert time data to standard format (if applicable)
    for column in df.columns:
        if 'date' in column.lower():
            df[column] = pd.to_datetime(df[column])

    # Set the datetime column as the index
    if 'date_column_name' in df.columns:
        df.set_index('date_column_name', inplace=True)
        df.sort_index(inplace=True)  # Sort by index

    # Apply differencing
    df_diff = df.diff().dropna()

    # Apply logarithmic transformation
    df_diff = df_diff.apply(lambda x: np.log(x))

    # Save processed DataFrame to a new CSV file
    if output_file:
        df_diff.to_csv(output_file)

    return df_diff


def plot_time_series(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['value_column_name'], marker='o', linestyle='-')
    plt.title('Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()


folder_path = "data"
file_name = "StockData.csv"
output_file = "preprocessed_data.csv"  # Specify the output file name

# Read and preprocess CSV, and save the preprocessed data to a file
df = read_and_preprocess_csv(folder_path, file_name, output_file=output_file)

print(df.head())

# Plot time series data
plot_time_series(df)
