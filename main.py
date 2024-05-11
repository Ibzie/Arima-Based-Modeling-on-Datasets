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

    # Convert date column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Set the datetime column as the index
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)  # Sort by index

    # Convert numeric columns to appropriate data types
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Drop rows with non-numeric values
    df.dropna(subset=numeric_columns, inplace=True)

    # Apply differencing
    df_diff = df.diff().dropna()

    # Apply logarithmic transformation to numeric columns
    df_diff[numeric_columns] = df_diff[numeric_columns].apply(lambda x: np.log(x))

    # Save processed DataFrame to a new CSV file
    if output_file:
        df_diff.to_csv(output_file)

    return df_diff


def plot_time_series(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['close'], marker='o', linestyle='-')
    plt.title('Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Close Price (log scale)')
    plt.yscale('log')  # Using log scale for better visualization of price changes
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
