import datetime
from pathlib import Path
import logging

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    root_mean_squared_error,
    mean_absolute_percentage_error)


from utils.download_data import download_data
from utils.model_data_prep import prepare_data_for_modeling
# except ImportError as e:
#     print("="*50)
#     print("ERROR: Could not import functions from 'utils' subfolder.")
#     print("Please ensure:")
#     print("1. You have a subfolder named 'utils'.")
#     print("2. It contains an empty file named '__init__.py'.")
#     print("3. It contains 'data_download.py' with 'download_data'.")
#     print("4. It contains 'model_data_preparation.py' with 'prepare_data_for_modeling'.")
#     print(f"Original error: {e}")
#     print("="*50)
#     exit() # Stop execution if imports fail

# exit()

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Parameters ---
TICKER = "MSFT"          # Stock ticker
TIMEFREQ = "1d"          # Data time frequency (e.g., "1d", "1h")
TARGET_COLUMN = 'Open'  # Column to forecast
TEST_SIZE = 0.2          # Proportion of data for the test set (e.g., 0.2 for 80/20 split)
TRAIN_LAST_N = 0.8       # Keep last 80% of the initial train data (use int for absolute count)
PLOT_RESULTS = False      # Whether to display the plot

# Define the base directory (consistent with the utils functions)
BASE_DATA_DIR = Path("data")
RESULTS_DATA_DIR = Path("results")

# --- Main Execution ---
if __name__ == "__main__":
    logging.info(f"Starting Naive Forecast process for {TICKER} ({TIMEFREQ})")

    # 1. (Optional) Download data if it doesn't exist
    # You can comment this out if you are sure the data is already downloaded
    try:
        logging.info("Checking/Downloading data...")
        # Assuming download_data uses the same BASE_DATA_DIR
        download_data(ticker_input=TICKER, timefreq=TIMEFREQ, base_dir=BASE_DATA_DIR)
    except Exception as e:
        logging.error(f"Error during data download step: {e}")
        # Decide if you want to stop or continue if download fails
        # exit()
    
    # 2. Prepare Data (Load, Split, Truncate)
    logging.info("Preparing data for modeling...")
    train_data = None
    test_data = None
    try:
        prepared_data = prepare_data_for_modeling(
            ticker=TICKER,
            timefreq=TIMEFREQ,
            train_last_n=TRAIN_LAST_N,
            target_column=TARGET_COLUMN,
            test_size=TEST_SIZE,
            base_dir=BASE_DATA_DIR
        )
        if prepared_data:
            train_data, test_data = prepared_data
            logging.info(f"Data prepared: Train size={len(train_data)}, Test size={len(test_data)}")
        else:
            logging.error("Data preparation failed or returned None.")
            exit() # Stop if data preparation fails

    except FileNotFoundError:
        logging.error(f"Required data file not found for {TICKER} ({TIMEFREQ}). Please run download first.")
        exit()
    except ValueError as e:
        logging.error(f"Invalid parameters or data issue during preparation: {e}")
        exit()
    except Exception as e:
        logging.error(f"An unexpected error occurred during data preparation: {e}")
        exit()

    # 3. Perform Naive Forecast
    logging.info("Performing Naive Forecast...")
    if train_data is None or train_data.empty:
        logging.error("Training data is empty, cannot perform Naive forecast.")
        exit()

    # exit()
    forecasts = [train_data.iloc[-1]]
    forecasts.extend(test_data.iloc[:-1])
    naive_forecast = pd.Series(forecasts, index=test_data.index)
    # naive_forecast = pd.Series(last_train_value, index=test_data.index)
    naive_forecast.name = "Naive Forecast"

    # logging.info(f"Naive forecast value (last train value): {last_train_value:.4f}")
    # 4. Evaluate Forecast
    logging.info("Evaluating Naive Forecast...")
    try:
        mae = mean_absolute_error(test_data, naive_forecast)
        rmse = root_mean_squared_error(test_data, naive_forecast)
        mape = mean_absolute_percentage_error(test_data, naive_forecast)
        mse = mean_squared_error(test_data, naive_forecast)
        
        
        logging.info(f"Evaluation Metrics for Naive Forecast:")
        logging.info(f"  MAE:  {mae:.4f}")
        logging.info(f"  RMSE: {rmse:.4f}")
        logging.info(f"  MAPE: {mape:.4f}")
        logging.info(f"  MSE:  {mse:.4f}")

    except Exception as e:
        logging.error(f"Error during forecast evaluation: {e}")

    # exit()
    # 5. Plot Results (Optional)
    if PLOT_RESULTS:
        logging.info("Plotting results...")
        try:
            plt.figure(figsize=(12, 6))
            # plt.plot(train_data.index, train_data, label=f'Train ({TARGET_COLUMN})', color='blue')
            plt.plot(test_data.index, test_data, label=f'Test ({TARGET_COLUMN})', color='green')
            plt.plot(naive_forecast.index, naive_forecast, label='Naive Forecast', color='red', linestyle='--')

            plt.title(f'Naive Forecast vs Actual Data for {TICKER} ({TARGET_COLUMN})')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error during plotting: {e}")
    # 6. Save Results to CSV
    logging.info("Saving results to CSV...")
    try:
        results_df = pd.DataFrame({
            'Actual': test_data,
            'Naive Forecast': naive_forecast
        })

        results_df.index.name = 'Date'  # Ensure index has a name
        results_dir = RESULTS_DATA_DIR / "Naive" / TICKER
        results_dir.mkdir(parents=True, exist_ok=True)

        # calculate a date time to me the results indentifiable YYYY-MM-DD-HH-MM
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M")

        results_file_path = results_dir / f"{TICKER}_{TIMEFREQ}_naive_forecast_results_{timestamp}.csv"
        results_df.to_csv(results_file_path)
        logging.info(f"Results saved to: {results_file_path}")
    except Exception as e:
        logging.error(f"Error during saving results to CSV: {e}")

    logging.info(f"Naive Forecast process for {TICKER} completed.")
