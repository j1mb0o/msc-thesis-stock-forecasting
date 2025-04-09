import pandas as pd
import pmdarima as pm
from pmdarima import model_selection
from pathlib import Path
import logging
from typing import Union, Tuple, Optional

# --- Configuration (reuse from previous function if desired) ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the base directory where data is stored
BASE_DATA_DIR = Path("data")

# --- Data Preparation Function ---
def prepare_data_for_modeling(
    ticker: str,
    timefreq: str,
    train_last_n: Union[float, int]=1.,
    target_column: str = 'Close',
    test_size: float = 0.2,
    base_dir: Path = BASE_DATA_DIR
) -> Optional[Tuple[pd.Series, pd.Series]]:
    """
    Loads pre-downloaded stock data, performs a train-test split,
    and keeps only the last specified portion of the training data.

    Args:
        ticker: The stock ticker string (e.g., "MSFT").
        timefreq: The time frequency string (e.g., "1d"). Must match
                  the saved file's naming convention.
        train_last_n: Specifies how much of the initial training data
                      to keep.
                      - If float (0.0 < value <= 1.0): Percentage of the
                        *initial* training set size (e.g., 0.5 for last 50%).
                      - If int (>= 1): The absolute number of *last* data
                        points to keep from the initial training set.
        target_column: The column name in the CSV to use as the time series
                       target (defaults to 'Close').
        test_size: The proportion of the data to use for the initial test
                   set (defaults to 0.2 for an 80-20 split).
        base_dir: The base directory Path object where data is stored
                  (defaults to BASE_DATA_DIR).

    Returns:
        A tuple containing (final_train_series, test_series) if successful.
        Returns None if data loading or processing fails.

    Raises:
        FileNotFoundError: If the specified data file does not exist.
        ValueError: If train_last_n has an invalid value or type, or if
                    target_column is not found.
        # Other exceptions might be raised by pandas or pmdarima.
    """
    ticker = ticker.upper() # Ensure consistency
    file_path = base_dir / ticker / f"{ticker}_{timefreq}.csv"

    # 1. Load Data
    try:
        data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        logging.info(f"Successfully loaded data from {file_path}")
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in {file_path}. Available columns: {data.columns.tolist()}")
        # Select only the target series
        series = data[target_column].dropna() # Drop NaNs from target
        if series.empty:
            logging.warning(f"No non-NaN data found for target '{target_column}' in {file_path}. Cannot proceed.")
            return None

    except FileNotFoundError:
        logging.error(f"Data file not found at: {file_path}")
        raise # Re-raise the error for the caller to handle
    except ValueError as e:
        logging.error(f"Error during data loading or validation: {e}")
        raise # Re-raise the error
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}")
        raise # Re-raise unexpected errors

    # 2. Initial Train-Test Split
    try:
        # Ensure test_size is valid
        if not 0 < test_size < 1:
             raise ValueError("test_size must be between 0 and 1 (exclusive).")

        initial_train, test = model_selection.train_test_split(series, test_size=test_size)
        logging.info(f"Initial split: Train size={len(initial_train)}, Test size={len(test)}")

        if len(initial_train) == 0 or len(test) == 0:
            logging.warning("Initial train or test set is empty after split. Check data length and test_size.")
            return None

    except Exception as e:
        logging.error(f"Error during train-test split: {e}")
        raise # Re-raise unexpected errors

    # 3. Calculate the number of training points (n) to keep
    n: Optional[int] = None
    train_len = len(initial_train)

    if isinstance(train_last_n, float):
        if 0.0 < train_last_n <= 1.0:
            n = int(train_len * train_last_n)
            # Ensure at least one data point if percentage is very small but > 0
            if n == 0 and train_last_n > 0.0:
                n = 1
            logging.info(f"train_last_n is a float ({train_last_n}). Keeping last {n} points ({train_last_n*100:.2f}%) of initial train set.")
        else:
            raise ValueError("If train_last_n is a float, it must be in the range (0.0, 1.0].")
    elif isinstance(train_last_n, int):
        if train_last_n >= 1:
            n = train_last_n
            logging.info(f"train_last_n is an int ({train_last_n}). Keeping last {n} points of initial train set.")
        else:
            raise ValueError("If train_last_n is an int, it must be >= 1.")
    else:
        raise ValueError("train_last_n must be a float (0.0 < value <= 1.0) or an int (>= 1).")

    # Validate calculated n
    if n is None or n < 1:
         logging.warning(f"Calculated number of training points 'n' ({n}) is invalid. Cannot proceed.")
         return None # Or raise error
    if n > train_len:
        logging.warning(f"Requested n ({n}) is greater than initial train size ({train_len}). Using entire initial train set.")
        n = train_len # Use the full initial training set

    # 4. Select the final training data (last n points)
    final_train = initial_train.iloc[-n:]
    logging.info(f"Final train set size after selecting last {n} points: {len(final_train)}")

    if final_train.empty:
         logging.warning("Final training set is empty after slicing. Check 'n' calculation and initial train data.")
         return None

    return final_train, test

# --- Example Usage ---
if __name__ == "__main__":
    # Assume "MSFT_1d.csv" exists in "data/MSFT/"
    TICKER = "MSFT" # Example Ticker
    TIMEFREQ = "1d"  # Example Time Frequency

    try:
        # Example 1: Keep last 50% of the initial 80% training data
        print("\n--- Example 1: Keep last 50% of train ---")
        train_data_perc, test_data_perc = prepare_data_for_modeling(
            ticker=TICKER,
            timefreq=TIMEFREQ,
            train_last_n=0.5, # Keep last 50%
            test_size=0.2      # Initial 80/20 split
        )
        if train_data_perc is not None:
            print(f"Final Train Head (Percentage):\n{train_data_perc.head()}")
            print(f"Final Train Tail (Percentage):\n{train_data_perc.tail()}")
            print(f"Test Head (Percentage):\n{test_data_perc.head()}")
            print(f"Train length: {len(train_data_perc)}, Test length: {len(test_data_perc)}")

        # Example 2: Keep last 200 data points of the initial 80% training data
        print("\n--- Example 2: Keep last 200 points of train ---")
        train_data_abs, test_data_abs = prepare_data_for_modeling(
            ticker=TICKER,
            timefreq=TIMEFREQ,
            train_last_n=200, # Keep last 200 points
            test_size=0.2     # Initial 80/20 split
        )
        if train_data_abs is not None:
            print(f"Final Train Head (Absolute):\n{train_data_abs.head()}")
            print(f"Final Train Tail (Absolute):\n{train_data_abs.tail()}")
            print(f"Test Head (Absolute):\n{test_data_abs.head()}")
            print(f"Train length: {len(train_data_abs)}, Test length: {len(test_data_abs)}")

        # Example 3: Use 'Open' column, keep last 75% of train
        print("\n--- Example 3: Use 'Open', keep last 75% ---")
        train_open, test_open = prepare_data_for_modeling(
            ticker=TICKER,
            timefreq=TIMEFREQ,
            train_last_n=0.75,
            target_column='Open', # Use the 'Open' price
            test_size=0.2
        )
        if train_open is not None:
             print(f"Final Train Head (Open):\n{train_open.head()}")
             print(f"Test Head (Open):\n{test_open.head()}")
             print(f"Train length: {len(train_open)}, Test length: {len(test_open)}")


    except FileNotFoundError:
        print(f"\nERROR: Ensure the data file exists for {TICKER} at {BASE_DATA_DIR / TICKER / f'{TICKER}_{TIMEFREQ}.csv'}")
        print("You might need to run the download function first.")
    except ValueError as e:
        print(f"\nERROR: Invalid input - {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

