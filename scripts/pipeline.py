import time
import datetime
import logging
import os
import yaml
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    # root_mean_squared_error, # Available in scikit-learn >= 1.0
    mean_absolute_percentage_error,
)
from pmdarima.metrics import smape
import numpy as np

# For older scikit-learn versions, define root_mean_squared_error if not present
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    import numpy as np  # type: ignore

    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_directional_accuracy(y_true, y_pred):
    """Calculate Mean Directional Accuracy (MDA).
    
    MDA is similar to Direction Accuracy but can be weighted by magnitude.
    For this implementation, we use the same formula as Direction Accuracy.
    
    Args:
        y_true: Array-like of true values
        y_pred: Array-like of predicted values
        
    Returns:
        Mean directional accuracy as a decimal (multiply by 100 for percentage)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.sign(y_true) == np.sign(y_pred))


from utils.download_data import download_data
from utils.model_data_prep import prepare_data_for_modeling
from utils.argfile import get_pipeline_arguments


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def save_experiment_config(
    config_dir, experiment_name, settings, method, ticker, timefreq, exp_name
):
    """Saves the experiment settings to a YAML file."""
    # Ensure experiment_name is filesystem-friendly
    safe_experiment_name = "".join(
        c if c.isalnum() or c in ["-", "_"] else "_" for c in experiment_name
    )

    config_path = Path(config_dir) / ticker / timefreq / method / exp_name
    config_path.mkdir(parents=True, exist_ok=True)
    config_file_path = config_path / f"{safe_experiment_name}.yaml"

    with open(config_file_path, "w") as f:
        yaml.dump(settings, f, default_flow_style=False, sort_keys=False)
    logging.info(f"Experiment config saved to: {config_file_path}")
    return config_file_path


args = get_pipeline_arguments()

if args.exp_name is None:
    EXP_NAME = datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S_experiment"
    )  # Added more detail
else:
    EXP_NAME = args.exp_name

logging.info(f"Starting forecasting pipeline with parameters: {args}")

TICKER = args.ticker
TIMEFREQ = args.timefreq
TARGET_COLUMN = args.target_column
# TRAIN_LAST_N_YEARS = args.train_last_n_years # Keep for clarity if needed
# TRAIN_LAST_N_DAYS = args.train_last_n_days   # Keep for clarity if needed
HORIZON = args.horizon_len


BASE_DATA_DIR = Path("data")
RESULTS_DATA_DIR = Path("results")
CONFIG_DIR = Path("configs")

logging.info(f"Starting process for {TICKER} ({TIMEFREQ})")

# Step 1 download data if not already done
download_data(ticker_input=TICKER, timefreq=TIMEFREQ, base_dir=BASE_DATA_DIR)

# 2. Prepare Data
logging.info("Preparing data for modeling...")
train_data = None
test_data = None

# Determine training period unit and value for logging/config
train_period_value = 0
train_period_unit = ""
if args.train_last_n_days is not None and args.train_last_n_days > 0:
    train_period_value = args.train_last_n_days
    train_period_unit = "days"
else:
    train_period_value = args.train_last_n_years
    train_period_unit = "years"

# Determine test period unit and value for logging/config
test_period_value = 0
test_period_unit = ""
if args.test_n_days is not None and args.test_n_days > 0:
    test_period_value = args.test_n_days
    test_period_unit = "days"
else:
    test_period_value = args.test_years
    test_period_unit = "years"

prepared_data = prepare_data_for_modeling(
    ticker=TICKER,
    timefreq=TIMEFREQ,
    rel_date=args.split_date,
    n_train_years=args.train_last_n_years,  # Pass years
    n_train_days=args.train_last_n_days,  # Pass days (will take precedence if set)
    n_test_years=args.test_years,
    n_test_days=args.test_n_days,  # Pass days (will take precedence if set)
    target_column=TARGET_COLUMN,
    base_dir=BASE_DATA_DIR,
    diff=args.diff,
    pct_change=args.pct_change,
)

if prepared_data:
    train_data, test_data = prepared_data
    if (
        train_data is not None
        and not train_data.empty
        and test_data is not None
        and not test_data.empty
    ):
        logging.info(
            f"Data prepared: Train size={len(train_data)}, Test size={len(test_data)}"
        )
    else:
        logging.error("Data preparation returned empty train or test set. Exiting.")
        exit()
else:
    logging.error("Data preparation failed or returned None. Exiting.")
    exit()

now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

# 3. Use method and forecast
forecasts_series = None
arima_order_info = "N/A"  # Default for non-ARIMA models

if args.method == "naive":
    from methods.naive_forecast import NaiveForecaster

    naive_forecaster = NaiveForecaster(train_data, test_data)
    forecasts = naive_forecaster.forecast(horizon=HORIZON)
    forecasts_series = forecasts
    forecasts_series.name = "Naive Forecast"

elif args.method == "arima":
    from methods.arima import ArimaForecaster

    arima = ArimaForecaster(train_data, test_data)
    arima.fit()  # Fit the model
    forecasts = arima.forecast(horizon=HORIZON)
    forecasts_series = forecasts
    forecasts_series.name = "ARIMA Forecast"
    arima_order_info = str(arima.order)  # Get order after fitting

elif args.method == "fm":
    # Ensure TimesFM is installed or handle import error
    try:
        from methods.times import TimesFMForecaster

        tfm = TimesFMForecaster(train_data, test_data, horizon_len=HORIZON)
        forecasts_series = tfm.forecast()
        forecasts_series.name = "TimesFM Forecast"
    except ImportError:
        logging.error(
            "TimesFM method requires the 'timesfm' package. Please install it."
        )
        exit()
    except Exception as e:
        logging.error(f"Error during TimesFM forecasting: {e}")
        exit()
elif args.method == "sundial":
    try:
        from methods.sundial import SundialForecaster

        sundial_forecaster = SundialForecaster(
            train_data, test_data, horizon_len=HORIZON
        )
        forecasts_series = sundial_forecaster.forecast()
        forecasts_series.name = "Sundial Forecast"
    except ImportError:
        logging.error(
            "Sundial method requires the 'transformers' package. Please ensure it is installed."
        )
        exit()
    except Exception as e:
        logging.error(f"Error during Sundial forecasting: {e}")
        exit()
elif args.method == "chronos_base":
    try:
        from methods.chronos_forcast import ChronosForecaster

        chronos_forecaster = ChronosForecaster(
            train_data, test_data, horizon_len=HORIZON
        )
        forecasts_series = chronos_forecaster.forecast()
        forecasts_series.name = "Chronos-MAC Forecast"
    except ImportError:
        if os.uname().sysname == "Darwin":
            logging.error(
                "Chronos-MAC method requires the 'chronos_mlx' package. Please ensure it is installed."
            )
        exit()
    except Exception as e:
        logging.error(f"Error during Chronos-MAC forecasting: {e}")
        exit()
else:
    logging.error(f"Unsupported method: {args.method}")
    exit()

if forecasts_series is None:
    logging.error(f"Forecast generation failed for method {args.method}. Exiting.")
    exit()

# Ensure test_data and forecasts_series have the same length for metrics calculation
if len(test_data) != len(forecasts_series):
    logging.warning(
        f"Test data length ({len(test_data)}) and forecast length ({len(forecasts_series)}) mismatch. Adjusting forecast to match test data length for evaluation."
    )
    # This can happen if horizon doesn't perfectly divide test set length, or if forecasting logic has issues.
    # A common approach is to take the forecast for the number of periods in test_data.
    min_len = min(len(test_data), len(forecasts_series))
    test_data = test_data.iloc[:min_len]
    forecasts_series = forecasts_series.iloc[:min_len]
    if min_len == 0:
        logging.error(
            "No overlapping data between test and forecast for metrics. Exiting."
        )
        exit()


# Construct a filename that reflects the training and test periods accurately
train_period_str = (
    f"{int(train_period_value)}{train_period_unit[0]}"  # e.g., 10y or 365d
)
test_period_str = (
    f"{int(test_period_value)}{test_period_unit[0]}"  # e.g., 1y or 600d
)

csv_name = (
    f"{timestamp}_{TICKER}_{TIMEFREQ}_{args.method}_split_{args.split_date}_"
    f"train_{train_period_str}_test_{test_period_str}_horizon_{HORIZON}.csv"
)

mae = mean_absolute_error(test_data, forecasts_series)
rmse = root_mean_squared_error(test_data, forecasts_series)
mape = mean_absolute_percentage_error(test_data, forecasts_series)
mse = mean_squared_error(test_data, forecasts_series)
smape = smape(test_data, forecasts_series)
mda = mean_directional_accuracy(test_data, forecasts_series)

logging.info(f"Evaluation Metrics for {args.method} Forecast:")
logging.info(f"  MSE:  {mse:.4f}")
logging.info(f"  MAE:  {mae:.4f}")
logging.info(f"  RMSE: {rmse:.4f}")
logging.info(f"  MAPE: {mape * 100:.4f}%")  # Display MAPE as percentage
logging.info(f"  SMAPE: {smape :.4f}%")  # Display SMAPE as percentage
logging.info(f"  Mean Directional Accuracy: {mda * 100:.4f}%")  # Display as percentage

results_df = pd.DataFrame(
    {"Actual": test_data, f"{args.method}_Forecast": forecasts_series}
)  # Renamed forecast column

results_df.index.name = "Date"
results_path = RESULTS_DATA_DIR / TICKER / TIMEFREQ / args.method / EXP_NAME
results_path.mkdir(parents=True, exist_ok=True)

results_file_path = results_path / csv_name
results_df.to_csv(results_file_path)
logging.info(f"Results saved to: {results_file_path}")

# Create experiment name for config file (can be same as csv_name without extension)
experiment_config_name = Path(csv_name).stem

experiment_settings = {
    "experiment_config_name": experiment_config_name,
    "run_timestamp": timestamp,
    "ticker": TICKER,
    "timefreq": TIMEFREQ,
    "target_column": TARGET_COLUMN,
    "split_date": args.split_date,
    "training_period_value": train_period_value,
    "training_period_unit": train_period_unit,
    "test_period_value": test_period_value,
    "test_period_unit": test_period_unit,
    "forecasting_method": args.method,
    "horizon_length": HORIZON,
    "differencing_applied": args.diff,
    "percentage_change_applied": args.pct_change,
    "results_file_path": str(results_file_path.resolve()),
    "arima_order": arima_order_info,
    "evaluation_metrics": {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape * 100,
        "smape": float(smape),
        "mean_directional_accuracy": float(mda * 100),
    },
}

save_experiment_config(
    CONFIG_DIR,
    experiment_config_name,
    experiment_settings,
    args.method,
    TICKER,
    TIMEFREQ,
    EXP_NAME,
)

logging.info(f"Process for {TICKER} completed successfully.")

if __name__ == "__main__":
    # testing purposes
    pass
