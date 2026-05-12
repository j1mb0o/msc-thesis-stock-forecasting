import os

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def _build_plot_labels(data_path, df):
    file_stem = os.path.splitext(os.path.basename(data_path))[0]
    parts = file_stem.split("_")

    ticker = parts[0].upper() if parts else file_stem.upper()
    timeframe = parts[1].upper() if len(parts) > 1 else "UNKNOWN"

    date_label = ""
    if "DATE" in df.columns:
        if "TIME" in df.columns:
            datetime_values = pd.to_datetime(
                df["DATE"].astype(str) + " " + df["TIME"].astype(str),
                errors="coerce",
            )
        else:
            datetime_values = pd.to_datetime(df["DATE"], errors="coerce")

        datetime_values = datetime_values.dropna()
        if not datetime_values.empty:
            date_label = f"({datetime_values.min():%Y-%m-%d} to {datetime_values.max():%Y-%m-%d})"

    instrument_label = f"{ticker} {timeframe}"
    return instrument_label, date_label


def plot_autocorrelations(data_path, output_dir):
    """
    Reads time series data from a CSV file, plots its ACF and PACF,
    and saves the plot to a specified directory. Also creates a separate
    plot for the ACF and PACF of the returns.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df = pd.read_csv(data_path, sep=None, engine="python")
        df.columns = [
            col.strip().replace("<", "").replace(">", "").upper() for col in df.columns
        ]
        close_col = "CLOSE"
        instrument_label, date_label = _build_plot_labels(data_path, df)

        if close_col not in df.columns:
            print(
                f"Column '{close_col}' not found in {data_path}. Available columns: {list(df.columns)}"
            )
            return

        stock_prices = df[close_col]

        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig1.suptitle(
            f"Autocorrelation Diagnostics - {instrument_label} Close Prices {date_label}",
            fontsize=14,
            fontweight="semibold",
        )

        plot_acf(stock_prices, ax=ax1, lags=100)
        ax1.set_title("ACF of Close Prices")
        ax1.set_xlabel("Lag")
        ax1.set_ylabel("ACF")

        plot_pacf(stock_prices, ax=ax2, lags=100)
        ax2.set_title("PACF of Close Prices")
        ax2.set_xlabel("Lag")
        ax2.set_ylabel("PACF")

        file_name = os.path.basename(data_path).replace(".csv", "_acf_pacf.pdf")
        output_path = os.path.join(output_dir, file_name)
        plt.tight_layout(rect=(0, 0.02, 1, 0.96))
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig1)
        print(f"Plot saved to {output_path}")

        # Returns (rather than raw prices) are usually the stationary input for ACF/PACF.
        returns = stock_prices.pct_change().dropna()

        fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 8))
        fig2.suptitle(
            f"Autocorrelation Diagnostics - {instrument_label} Returns {date_label}",
            fontsize=14,
            fontweight="semibold",
        )

        plot_acf(returns, ax=ax3, lags=100)
        ax3.set_title("ACF of Returns")
        ax3.set_xlabel("Lag")
        ax3.set_ylabel("ACF")

        plot_pacf(returns, ax=ax4, lags=100)
        ax4.set_title("PACF of Returns")
        ax4.set_xlabel("Lag")
        ax4.set_ylabel("PACF")

        returns_file_name = os.path.basename(data_path).replace(
            ".csv", "_returns_acf_pacf.pdf"
        )
        returns_output_path = os.path.join(output_dir, returns_file_name)
        plt.tight_layout(rect=(0, 0.02, 1, 0.96))
        plt.savefig(returns_output_path, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"Plot saved to {returns_output_path}")

    except Exception as e:
        print(f"An error occurred while processing {data_path}: {e}")


if __name__ == "__main__":
    data_files = [
        "/Users/dimitris/LU/Thesis/Thesis-Master-Repo/thesis-code-new/data/MSFT/MSFT_1d.csv",
        "/Users/dimitris/LU/Thesis/Thesis-Master-Repo/thesis-code-new/data/MSFT/MSFT_1h.csv",
    ]

    output_dirs = [
        "/Users/dimitris/LU/Thesis/Thesis-Master-Repo/thesis-code-new/figures/MSFT/1d",
        "/Users/dimitris/LU/Thesis/Thesis-Master-Repo/thesis-code-new/figures/MSFT/1h",
    ]

    for data_file, out_dir in zip(data_files, output_dirs):
        plot_autocorrelations(data_file, out_dir)
