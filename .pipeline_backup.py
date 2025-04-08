import argparse
import datetime
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)

from utils.download_data import download_data
from utils.model_data_prep import prepare_data_for_modeling
from methods.naive_forecast import NaiveForecaster


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# def main():
parser = argparse.ArgumentParser(description="Run forecasting pipeline.")
parser.add_argument(
    "--ticker",
    type=str,
    default="MSFT",
    help="Stock ticker symbol (e.g., MSFT)",
)
parser.add_argument(
    "--timefreq", type=str, default="1d", help="Data time frequency (e.g., 1d, 1h)"
)
parser.add_argument(
    "--test_size",
    type=float,
    default=0.2,
    help="Proportion of data for the test set",
)
parser.add_argument(
    "--train_last_n",
    type=float,
    default=1.0,
    help="Proportion of train data to keep",
)
parser.add_argument(
    "--method",
    type=str,
    choices=["naive", "arima", "fm"],
    default="naive",
    help="Forecasting method (naive, arima)",
)
parser.add_argument(
    "--target_column", type=str, default="Open", help="Target column to forecast"
)
# parser.add_argument(
#     "--plot_results", action="store_true", help="Whether to display the plot"
# )

args = parser.parse_args()

logging.info(f"Starting forecasting pipeline with parameters: {args}")

TICKER = args.ticker
TIMEFREQ = args.timefreq
TARGET_COLUMN = args.target_column
TEST_SIZE = args.test_size
TRAIN_LAST_N = args.train_last_n

BASE_DATA_DIR = Path("data")
RESULTS_DATA_DIR = Path("results")

logging.info(f"Starting process for {TICKER} ({TIMEFREQ})")
# Step 1 download data if not already done
# try:
#     logging.info("Checking/Downloading data...")
download_data(ticker_input=TICKER, timefreq=TIMEFREQ, base_dir=BASE_DATA_DIR)
# except Exception as e:
#         logging.error(f"Error during data download step: {e}")
    

# 2. Prepare Data (Load, Split, Truncate)
logging.info("Preparing data for modeling...")
train_data = None
test_data = None
# try:
prepared_data = prepare_data_for_modeling(
    ticker=TICKER,
    timefreq=TIMEFREQ,
    train_last_n=TRAIN_LAST_N,
    target_column=TARGET_COLUMN,
    test_size=TEST_SIZE,
    base_dir=BASE_DATA_DIR,
)
if prepared_data:
    train_data, test_data = prepared_data
    logging.info(
        f"Data prepared: Train size={len(train_data)}, Test size={len(test_data)}"
    )
else:
    logging.error("Data preparation failed or returned None.")
    exit()  # Stop if data preparation fails

# # except FileNotFoundError:
# #     logging.error(
# #         f"Required data file not found for {TICKER} ({TIMEFREQ}). Please run download first."
# #     )
# #     exit()
# except ValueError as e:
#     logging.error(f"Invalid parameters or data issue during preparation: {e}")
#     exit()
# except Exception as e:
#     logging.error(f"An unexpected error occurred during data preparation: {e}")
#     exit()

# 3. Use method and forecast
if args.method == "naive":
    naive_forecaster = NaiveForecaster(train_data, test_data)
    forecasts = naive_forecaster.forecast(horizon=1)
    forecasts_series = forecasts

    forecasts_series.name = "Naive Forecast"
elif args.method == "arima":
    pass
elif args.method == "fm":
    pass


mae = mean_absolute_error(test_data, forecasts_series)
rmse = root_mean_squared_error(test_data, forecasts_series)
mape = mean_absolute_percentage_error(test_data, forecasts_series)
mse = mean_squared_error(test_data, forecasts_series)

logging.info(f"Evaluation Metrics for {args.method} Forecast:")
logging.info(f"  MSE:  {mse:.4f}")
logging.info(f"  MAE:  {mae:.4f}")
logging.info(f"  RMSE: {rmse:.4f}")
logging.info(f"  MAPE: {mape*100:.4f}")
logging.info(f"  MSE:  {mse:.4f}")

results_df = pd.DataFrame(
            {"Actual": test_data, f"{args.method}": forecasts_series}
        )


results_df.index.name = "Date"  # Ensure index has a name
results_dir = RESULTS_DATA_DIR / "Naive" / TICKER
results_dir.mkdir(parents=True, exist_ok=True)

# calculate a date time to me the results indentifiable YYYY-MM-DD-HH-MM
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")

results_file_path = (
    results_dir / f"{TICKER}_{TIMEFREQ}_naive_forecast_results_{timestamp}.csv"
)
results_df.to_csv(results_file_path)
logging.info(f"Results saved to: {results_file_path}")


logging.info(f"Process for {TICKER} completed.")
