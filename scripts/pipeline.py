import datetime
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Callable, Tuple

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from pmdarima.metrics import smape as pm_smape
import numpy as np

# Fallback for scikit-learn < 1.4 which lacks root_mean_squared_error.
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

from utils.download_data import download_data
from utils.model_data_prep import prepare_data_for_modeling
from utils.argfile import get_pipeline_arguments


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


BASE_DATA_DIR = Path("data")
RESULTS_DATA_DIR = Path("results")
CONFIG_DIR = Path("configs")


def mean_directional_accuracy(y_true, y_pred):
    """Fraction of test points where forecast and actual share the same sign."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.sign(y_true) == np.sign(y_pred))


def save_experiment_config(
    config_dir, experiment_name, settings, method, ticker, timefreq, exp_name
):
    """Saves the experiment settings to a YAML file."""
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


def _resolve_pipeline_period(
    n_days, n_years, days_unit_label="days", years_unit_label="years"
) -> Tuple[int, str]:
    """Mirrors prepare_data_for_modeling's precedence: positive days override years."""
    if n_days is not None and n_days > 0:
        return n_days, days_unit_label
    return n_years, years_unit_label


# ---------- forecaster registry ----------
#
# Each entry is keyed by --method and yields (forecasts_series, arima_order_info).
# Imports are inside the factories so a missing optional dependency only kills
# the run when that method is actually requested.

def _run_naive(train, test, horizon):
    from methods.naive_forecast import NaiveForecaster

    forecaster = NaiveForecaster(train, test)
    series = forecaster.forecast(horizon=horizon)
    series.name = "Naive Forecast"
    return series, "N/A"


def _run_arima(train, test, horizon):
    from methods.arima import ArimaForecaster

    arima = ArimaForecaster(train, test)
    arima.fit()
    series = arima.forecast(horizon=horizon)
    series.name = "ARIMA Forecast"
    return series, str(arima.order)


def _run_timesfm(train, test, horizon):
    try:
        from methods.times import TimesFMForecaster
    except ImportError:
        logging.error("TimesFM method requires the 'timesfm' package. Please install it.")
        sys.exit(1)
    series = TimesFMForecaster(train, test, horizon_len=horizon).forecast()
    series.name = "TimesFM Forecast"
    return series, "N/A"


def _run_sundial(train, test, horizon):
    try:
        from methods.sundial import SundialForecaster
    except ImportError:
        logging.error("Sundial method requires the 'transformers' package. Please ensure it is installed.")
        sys.exit(1)
    series = SundialForecaster(train, test, horizon_len=horizon).forecast()
    series.name = "Sundial Forecast"
    return series, "N/A"


def _run_chronos(train, test, horizon):
    try:
        from methods.chronos_forcast import ChronosForecaster
    except ImportError:
        if os.uname().sysname == "Darwin":
            logging.error("Chronos-MAC method requires the 'chronos_mlx' package. Please ensure it is installed.")
        sys.exit(1)
    series = ChronosForecaster(train, test, horizon_len=horizon).forecast()
    series.name = "Chronos-MAC Forecast"
    return series, "N/A"


_FORECASTERS: dict[str, Callable] = {
    "naive": _run_naive,
    "arima": _run_arima,
    "fm": _run_timesfm,
    "sundial": _run_sundial,
    "chronos_base": _run_chronos,
}


def _compute_metrics(test_data, forecasts):
    """Return all evaluation metrics as a plain dict."""
    return {
        "mse": mean_squared_error(test_data, forecasts),
        "mae": mean_absolute_error(test_data, forecasts),
        "rmse": root_mean_squared_error(test_data, forecasts),
        "mape": mean_absolute_percentage_error(test_data, forecasts),
        "smape": float(pm_smape(test_data, forecasts)),
        "mda": float(mean_directional_accuracy(test_data, forecasts)),
    }


def _log_metrics(method, metrics):
    logging.info(f"Evaluation Metrics for {method} Forecast:")
    logging.info(f"  MSE:  {metrics['mse']:.4f}")
    logging.info(f"  MAE:  {metrics['mae']:.4f}")
    logging.info(f"  RMSE: {metrics['rmse']:.4f}")
    logging.info(f"  MAPE: {metrics['mape'] * 100:.4f}%")
    logging.info(f"  SMAPE: {metrics['smape']:.4f}%")
    logging.info(f"  Mean Directional Accuracy: {metrics['mda'] * 100:.4f}%")


def main():
    args = get_pipeline_arguments()

    exp_name = args.exp_name or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_experiment")

    logging.info(f"Starting forecasting pipeline with parameters: {args}")

    ticker = args.ticker
    timefreq = args.timefreq
    horizon = args.horizon_len

    if args.method not in _FORECASTERS:
        logging.error(f"Unsupported method: {args.method}")
        sys.exit(1)

    logging.info(f"Starting process for {ticker} ({timefreq})")

    download_data(ticker_input=ticker, timefreq=timefreq, base_dir=BASE_DATA_DIR)

    logging.info("Preparing data for modeling...")

    train_period_value, train_period_unit = _resolve_pipeline_period(
        args.train_last_n_days, args.train_last_n_years
    )
    test_period_value, test_period_unit = _resolve_pipeline_period(
        args.test_n_days, args.test_years
    )

    prepared_data = prepare_data_for_modeling(
        ticker=ticker,
        timefreq=timefreq,
        rel_date=args.split_date,
        n_train_years=args.train_last_n_years,
        n_train_days=args.train_last_n_days,
        n_test_years=args.test_years,
        n_test_days=args.test_n_days,
        target_column=args.target_column,
        base_dir=BASE_DATA_DIR,
        diff=args.diff,
        pct_change=args.pct_change,
    )

    if not prepared_data:
        logging.error("Data preparation failed or returned None. Exiting.")
        sys.exit(1)

    train_data, test_data = prepared_data
    if train_data is None or train_data.empty or test_data is None or test_data.empty:
        logging.error("Data preparation returned empty train or test set. Exiting.")
        sys.exit(1)

    logging.info(f"Data prepared: Train size={len(train_data)}, Test size={len(test_data)}")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    forecasts_series, arima_order_info = _FORECASTERS[args.method](
        train_data, test_data, horizon
    )

    if forecasts_series is None:
        logging.error(f"Forecast generation failed for method {args.method}. Exiting.")
        sys.exit(1)

    if len(test_data) != len(forecasts_series):
        logging.warning(
            f"Test data length ({len(test_data)}) and forecast length ({len(forecasts_series)}) "
            f"mismatch. Adjusting forecast to match test data length for evaluation."
        )
        min_len = min(len(test_data), len(forecasts_series))
        if min_len == 0:
            logging.error("No overlapping data between test and forecast for metrics. Exiting.")
            sys.exit(1)
        test_data = test_data.iloc[:min_len]
        forecasts_series = forecasts_series.iloc[:min_len]

    train_period_str = f"{int(train_period_value)}{train_period_unit[0]}"
    test_period_str = f"{int(test_period_value)}{test_period_unit[0]}"

    csv_name = (
        f"{timestamp}_{ticker}_{timefreq}_{args.method}_split_{args.split_date}_"
        f"train_{train_period_str}_test_{test_period_str}_horizon_{horizon}.csv"
    )

    metrics = _compute_metrics(test_data, forecasts_series)
    _log_metrics(args.method, metrics)

    results_df = pd.DataFrame(
        {"Actual": test_data, f"{args.method}_Forecast": forecasts_series}
    )
    results_df.index.name = "Date"
    results_path = RESULTS_DATA_DIR / ticker / timefreq / args.method / exp_name
    results_path.mkdir(parents=True, exist_ok=True)

    results_file_path = results_path / csv_name
    results_df.to_csv(results_file_path)
    logging.info(f"Results saved to: {results_file_path}")

    experiment_config_name = Path(csv_name).stem

    experiment_settings = {
        "experiment_config_name": experiment_config_name,
        "run_timestamp": timestamp,
        "ticker": ticker,
        "timefreq": timefreq,
        "target_column": args.target_column,
        "split_date": args.split_date,
        "training_period_value": train_period_value,
        "training_period_unit": train_period_unit,
        "test_period_value": test_period_value,
        "test_period_unit": test_period_unit,
        "forecasting_method": args.method,
        "horizon_length": horizon,
        "differencing_applied": args.diff,
        "percentage_change_applied": args.pct_change,
        "results_file_path": str(results_file_path.resolve()),
        "arima_order": arima_order_info,
        "evaluation_metrics": {
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape"] * 100,
            "smape": metrics["smape"],
            "mean_directional_accuracy": metrics["mda"] * 100,
        },
    }

    save_experiment_config(
        CONFIG_DIR,
        experiment_config_name,
        experiment_settings,
        args.method,
        ticker,
        timefreq,
        exp_name,
    )

    logging.info(f"Process for {ticker} completed successfully.")


if __name__ == "__main__":
    main()
