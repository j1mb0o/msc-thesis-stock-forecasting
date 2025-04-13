import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Optional
from dateutil.relativedelta import relativedelta # Correctly imported
import datetime # Needed for Timestamp

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the default base directory where data is stored
BASE_DATA_DIR = Path("data")


def prepare_data_for_modeling(
    ticker: str = "MSFT",
    timefreq: str = "1d",
    target_column: str = "Open",
    rel_date: str = "2020-01-01", # The date *before* which training data ends, and *at* which test data begins
    n_train_years: int = 10,
    n_test_years: int = 2,
    base_dir: Path = BASE_DATA_DIR,
    diff: bool = False
) -> Optional[Tuple[pd.Series, pd.Series]]:
    """
    Loads time series data, selects a target column, optionally differences it,
    and splits it into training and testing sets based on a relative date and specified durations.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'MSFT'). Will be converted to uppercase.
        timefreq (str): The time frequency of the data file (e.g., '1d').
                        Used to construct the filename: {ticker}_{timefreq}.csv.
        target_column (str): The name of the column in the CSV to use as the target series.
                             Defaults to "Open".
        rel_date (str): The reference date string (YYYY-MM-DD format expected) used as the split point.
                        Training data will end *before* this date.
                        Testing data will start *at* this date. Defaults to "2020-01-01".
        n_train_years (int): The number of years of training data to retrieve *before* rel_date.
                             Defaults to 10.
        n_test_years (int): The number of years of testing data to retrieve *from* rel_date.
                            Defaults to 5.
        base_dir (Path): The base directory where ticker-specific subdirectories are located.
                         Defaults to BASE_DATA_DIR ('data/').
        diff (bool): If True, computes the first difference of the series before splitting.
                     Defaults to False.

    Returns:
        Optional[Tuple[pd.Series, pd.Series]]: A tuple containing the training series
                                               and the testing series (train, test).
                                               Returns None if the target column has no valid data
                                               after initial loading and NaN removal, or after
                                               differencing if applicable.

    Raises:
        FileNotFoundError: If the data file ({base_dir}/{ticker}/{ticker}_{timefreq}.csv) is not found.
        ValueError: If the target_column is not found in the loaded data, or if rel_date
                    is not a valid date string that can be parsed by pd.Timestamp.
        Exception: For other unexpected errors during file loading or processing.
    """
    ticker = ticker.upper()
    # Ensure base_dir is a Path object
    base_dir_path = Path(base_dir)
    file_path = base_dir_path / ticker / f"{ticker}_{timefreq}.csv"
    logging.info(f"Attempting to load data from: {file_path}")

    try:
        # --- Validate inputs ---
        try:
            split_date = pd.Timestamp(rel_date)
        except ValueError as e:
            msg = f"Invalid rel_date format: '{rel_date}'. Expected YYYY-MM-DD. Error: {e}"
            logging.error(msg)
            raise ValueError(msg) from e

        # --- Load data ---
        data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        logging.info(f"Successfully loaded data for {ticker} ({timefreq}). Columns: {data.columns.tolist()}")

        # --- Validate target column ---
        if target_column not in data.columns:
            msg = f"Target column '{target_column}' not found in {file_path}. Available: {data.columns.tolist()}"
            logging.error(msg)
            raise ValueError(msg)

        # --- Select and clean target series ---
        series = data[target_column].dropna()
        if series.empty:
            logging.warning(f"No non-NaN data for target '{target_column}' in {file_path} after dropna().")
            return None # Return None as per Optional type hint if no valid data

        # Sort index to ensure correct slicing
        series = series.sort_index()
        logging.info(f"Target series '{target_column}' selected. Length after dropna: {len(series)}. Date range: {series.index.min()} to {series.index.max()}")

    except FileNotFoundError:
        logging.error(f"Data file not found: {file_path}")
        raise # Re-raise the specific error
    except ValueError as e:
        # Catch specific ValueError from Timestamp conversion or column check
        logging.error(f"Data validation error: {e}")
        raise # Re-raise the specific error
    except Exception as e:
        logging.error(f"Unexpected error loading/validating data from {file_path}: {e}")
        raise # Re-raise other unexpected errors

    # --- Apply differencing if requested ---
    if diff:
        logging.info(f"Applying first difference to '{target_column}' series.")
        series = series.diff().dropna()
        if series.empty:
            return None # Return None if differencing results in no data
        logging.info(f"Series length after differencing and dropna: {len(series)}")

    # --- Calculate date ranges ---
    train_start_date = split_date - relativedelta(years=n_train_years)
    test_end_date = split_date + relativedelta(years=n_test_years)


    logging.info(f"Splitting data around reference date: {split_date.date()}")
    logging.info(f"Requested train period start: {train_start_date.date()}")
    logging.info(f"Requested test period end: {test_end_date.date()}")

    # --- Split the data ---
    train = series.loc[(series.index <= split_date) & (series.index > train_start_date)].copy()
    test = series.loc[(series.index >= split_date) & (series.index < test_end_date)].copy() 

    # --- Validate and log splits ---
    if train.empty:
        logging.warning(f"Training set is empty for the period >= {train_start_date.date()} and < {split_date.date()}. Check data availability and requested range.")
    else:
        logging.info(f"Training set created with {len(train)} points from {train.index.min().date()} to {train.index.max().date()}")

    if test.empty:
        logging.warning(f"Test set is empty for the period >= {split_date.date()} and < {test_end_date.date()}. Check data availability and requested range.")
    else:
         logging.info(f"Test set created with {len(test)} points from {test.index.min().date()} to {test.index.max().date()}")

    return train, test

# --- Example Usage ---
if __name__ == "__main__":
    # Assume "MSFT_1d.csv" exists in "data/MSFT/"
    TICKER = "MSFT" # Example Ticker
    TIMEFREQ = "1d"  # Example Time Frequency

    train, test = prepare_data_for_modeling()
