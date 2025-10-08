
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

def plot_autocorrelations(data_path, output_dir):
    """
    Reads time series data from a CSV file, plots its ACF and PACF,
    and saves the plot to a specified directory. Also creates a separate
    plot for the ACF and PACF of the returns.
    """
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Read the data
        if '1d' in data_path:
            df = pd.read_csv(data_path)
            close_col = 'Close'
        elif '1h' in data_path:
            df = pd.read_csv(data_path, sep='\t')
            df.columns = [col.strip().replace('<','').replace('>','') for col in df.columns]
            close_col = 'CLOSE'
        else:
            print(f"Unsupported file: {data_path}")
            return

        # Ensure the close column exists
        if close_col not in df.columns:
            print(f"Column '{close_col}' not found in {data_path}")
            return

        stock_prices = df[close_col]

        # Create a figure for raw prices with two subplots
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig1.suptitle(f'ACF and PACF for {os.path.basename(data_path)}')

        # Plot ACF for raw prices
        plot_acf(stock_prices, ax=ax1, lags=100)
        ax1.set_title('Autocorrelation Function (ACF)')
        ax1.set_xlabel("Lag")
        ax1.set_ylabel("ACF")

        # Plot PACF for raw prices
        plot_pacf(stock_prices, ax=ax2, lags=100)
        ax2.set_title('Partial Autocorrelation Function (PACF)')
        ax2.set_xlabel("Lag")
        ax2.set_ylabel("PACF")

        # Save the figure for raw prices
        file_name = os.path.basename(data_path).replace('.csv', '_acf_pacf.png')
        output_path = os.path.join(output_dir, file_name)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig1)
        print(f"Plot saved to {output_path}")

        # It's more common to analyze returns than prices for stationarity
        returns = stock_prices.pct_change().dropna()

        # Create a figure for returns with two subplots
        fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 8))
        fig2.suptitle(f'ACF and PACF for Returns of {os.path.basename(data_path)}')

        # Plot ACF for returns
        plot_acf(returns, ax=ax3, lags=100)
        ax3.set_title('ACF of Returns')
        ax3.set_xlabel("Lag")
        ax3.set_ylabel("ACF")

        # Plot PACF for returns
        plot_pacf(returns, ax=ax4, lags=100)
        ax4.set_title('PACF of Returns')
        ax4.set_xlabel("Lag")
        ax4.set_ylabel("PACF")

        # Save the figure for returns
        returns_file_name = os.path.basename(data_path).replace('.csv', '_returns_acf_pacf.png')
        returns_output_path = os.path.join(output_dir, returns_file_name)
        plt.tight_layout()
        plt.savefig(returns_output_path)
        plt.close(fig2)
        print(f"Plot saved to {returns_output_path}")

    except Exception as e:
        print(f"An error occurred while processing {data_path}: {e}")

if __name__ == "__main__":
    # Define file paths
    data_files = [
        "/Users/dimitris/LU/Thesis/thesis-code-new/data/MSFT/MSFT_1d.csv",
        "/Users/dimitris/LU/Thesis/thesis-code-new/data/MSFT/MSFT_1h.csv"
    ]

    output_dirs = [
        "/Users/dimitris/LU/Thesis/thesis-code-new/figures/MSFT/1d",
        "/Users/dimitris/LU/Thesis/thesis-code-new/figures/MSFT/1h"
    ]

    # Generate and save plots for each dataset
    for data_file, out_dir in zip(data_files, output_dirs):
        plot_autocorrelations(data_file, out_dir)
