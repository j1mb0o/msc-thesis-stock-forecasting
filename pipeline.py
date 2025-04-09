import argparse
import datetime
import logging
import os
import yaml
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

from methods.arima import ArimaForecaster
from methods.naive_forecast import NaiveForecaster
from methods.times import TimesFMForecaster

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def save_experiment_config(config_dir, experiment_name, settings, method):
    """Saves the experiment settings to a YAML file."""
    os.makedirs(config_dir/method, exist_ok=True)
    config_file_path = Path(config_dir) / method / f"{experiment_name}.yaml"
    with open(config_file_path, "w") as f:
        yaml.dump(settings, f, default_flow_style=False, sort_keys=False)
    logging.info(f"Experiment config saved to: {config_file_path}")
    return config_file_path



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
parser.add_argument(
    "--horizon_len", type=int, default=1, help="Horizon length"
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
HORIZON = args.horizon_len


BASE_DATA_DIR = Path("data")
RESULTS_DATA_DIR = Path("results")
CONFIG_DIR = Path("configs")

logging.info(f"Starting process for {TICKER} ({TIMEFREQ})")

# Step 1 download data if not already done
download_data(ticker_input=TICKER, timefreq=TIMEFREQ, base_dir=BASE_DATA_DIR)

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

now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

# 3. Use method and forecast
if args.method == "naive":
    naive_forecaster = NaiveForecaster(train_data, test_data)
    forecasts = naive_forecaster.forecast(horizon=HORIZON)
    forecasts_series = forecasts
    forecasts_series.name = "Naive Forecast"
    
elif args.method == "arima":
    arima = ArimaForecaster(train_data, test_data)
    arima.fit()
    forecasts = arima.forecast(horizon=HORIZON)
    forecasts_series = forecasts
    forecasts_series.name = "ARIMA Forecast"
    arima_order = arima.order

elif args.method == "fm":
    tfm = TimesFMForecaster(train_data, test_data, horizon_len=HORIZON)
    forecasts_series = tfm.forecast()
    forecasts_series.name = "TimesFM Forecast"

csv_name = f"{timestamp}_{TICKER}_{TIMEFREQ}_naive_h{HORIZON}_results.csv"

mae = mean_absolute_error(test_data, forecasts_series)
rmse = root_mean_squared_error(test_data, forecasts_series)
mape = mean_absolute_percentage_error(test_data, forecasts_series)
mse = mean_squared_error(test_data, forecasts_series)

logging.info(f"Evaluation Metrics for {args.method} Forecast:")
logging.info(f"  MSE:  {mse:.4f}")
logging.info(f"  MAE:  {mae:.4f}")
logging.info(f"  RMSE: {rmse:.4f}")
logging.info(f"  MAPE: {mape*100:.4f}")

results_df = pd.DataFrame(
            {"Actual": test_data, f"{args.method}": forecasts_series}
        )


results_df.index.name = "Date"  # Ensure index has a name
results_dir = RESULTS_DATA_DIR / args.method / TICKER
results_dir.mkdir(parents=True, exist_ok=True)

results_file_path = results_dir / csv_name
results_df.to_csv(results_file_path)
logging.info(f"Results saved to: {results_file_path}")

# Create experiment name
experiment_name = f"{timestamp}_{TICKER}_{TIMEFREQ}_{args.method}_h{HORIZON}"

# Save experiment settings to config file
experiment_settings = {
    "experiment_name": experiment_name,
    "timestamp": timestamp,
    "ticker": TICKER,
    "timefreq": TIMEFREQ,
    "target_column": TARGET_COLUMN,
    "test_size": TEST_SIZE,
    "train_last_n": TRAIN_LAST_N,
    "method": args.method,
    "horizon": HORIZON,
    "results_file_path": str(results_file_path),
    "arima_order": arima_order if args.method == "arima" else "N/A",
    "metrics": {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape}
}

save_experiment_config(CONFIG_DIR, experiment_name, experiment_settings, args.method)


logging.info(f"Process for {TICKER} completed.")