import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Optional
from dateutil.relativedelta import relativedelta
import datetime # Needed for Timestamp

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the default base directory where data is stored
BASE_DATA_DIR = Path("data")


def prepare_data_for_modeling(
    ticker: str = "MSFT",
    timefreq: str = "1d",
    target_column: str = "Open",
    rel_date: str = "2020-01-01",
    n_train_years: Optional[float] = 10, # Can be None if days are used
    n_train_days: Optional[int] = None,  # New parameter for days
    n_test_years: float = 1,
    n_test_days: Optional[int] = None,  # New parameter for test days
    base_dir: Path = BASE_DATA_DIR,
    diff: bool = False,
    pct_change: bool = False
) -> Optional[Tuple[pd.Series, pd.Series]]:
    """
    Loads time series data, selects a target column, optionally differences it,
    and splits it into training and testing sets based on a relative date and specified durations.
    Training and testing durations can be specified in years or days, with days taking precedence.

    Args:
        ticker (str): The stock ticker symbol.
        timefreq (str): The time frequency of the data file.
        target_column (str): The name of the column to use as the target series.
        rel_date (str): The reference date string (YYYY-MM-DD) used as the split point.
        n_train_years (Optional[float]): Number of years of training data before rel_date.
                                        Used if n_train_days is None. Defaults to 10.
        n_train_days (Optional[int]): Number of days of training data before rel_date.
                                     If provided, this overrides n_train_years. Defaults to None.
        n_test_years (float): Number of years of testing data from rel_date.
                             Used if n_test_days is None. Defaults to 1.
        n_test_days (Optional[int]): Number of days of testing data from rel_date.
                                    If provided, this overrides n_test_years. Defaults to None.
        base_dir (Path): The base directory for data.
        diff (bool): If True, computes the first difference of the series. Defaults to False.
        pct_change (bool): If True, computes the percentage change of the series. Defaults to False.
                           Mutually exclusive with diff parameter.

    Returns:
        Optional[Tuple[pd.Series, pd.Series]]: (train_series, test_series), or None on failure.

    Raises:
        FileNotFoundError: If the data file is not found.
        ValueError: For invalid inputs (e.g., date format, target column).
    """
    ticker = ticker.upper()
    base_dir_path = Path(base_dir)
    file_path = base_dir_path / ticker / f"{ticker}_{timefreq}.csv"
    logging.info(f"Attempting to load data from: {file_path}")

    try:
        try:
            split_date = pd.Timestamp(rel_date)
        except ValueError as e:
            msg = f"Invalid rel_date format: '{rel_date}'. Expected YYYY-MM-DD. Error: {e}"
            logging.error(msg)
            raise ValueError(msg) from e

        if timefreq == '1h':
            data = pd.read_csv(file_path, sep='\t', engine='python')
            data.columns = data.columns.str.replace('[<>]', '', regex=True)
            data.columns = data.columns.str.title()
            data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
            data = data.set_index('datetime')
            data = data.drop(columns=['Date', 'Time', 'Tickvol', 'Vol', 'Spread'])
        else:
            # Try reading as tab-separated with angle-bracket headers first (new format)
            try:
                data = pd.read_csv(file_path, sep='\t', engine='python')
                # Check if it has angle-bracket headers
                if any('<' in col or '>' in col for col in data.columns):
                    data.columns = data.columns.str.replace('[<>]', '', regex=True)
                    data.columns = data.columns.str.title()
                    data['Date'] = pd.to_datetime(data['Date'])
                    data = data.set_index('Date')
                    # Drop extra columns if they exist
                    cols_to_drop = [col for col in ['Tickvol', 'Vol', 'Spread'] if col in data.columns]
                    if cols_to_drop:
                        data = data.drop(columns=cols_to_drop)
                else:
                    # Tab-separated but without angle brackets, try standard format
                    raise ValueError("Not angle-bracket format")
            except (ValueError, KeyError):
                # Fall back to standard yfinance CSV format
                data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        logging.info(f"Successfully loaded data for {ticker} ({timefreq}). Columns: {data.columns.tolist()}")

        if target_column not in data.columns:
            msg = f"Target column '{target_column}' not found in {file_path}. Available: {data.columns.tolist()}"
            logging.error(msg)
            raise ValueError(msg)

        series = data[target_column].dropna()
        if series.empty:
            logging.warning(f"No non-NaN data for target '{target_column}' in {file_path} after dropna().")
            return None

        series = series.sort_index()
        logging.info(f"Target series '{target_column}' selected. Length: {len(series)}. Date range: {series.index.min()} to {series.index.max()}")

    except FileNotFoundError:
        logging.error(f"Data file not found: {file_path}")
        raise
    except ValueError as e:
        logging.error(f"Data validation error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading/validating data from {file_path}: {e}")
        raise

    # Validate that diff and pct_change are not both True
    if diff and pct_change:
        msg = "diff and pct_change parameters are mutually exclusive. Please set only one to True."
        logging.error(msg)
        raise ValueError(msg)

    if diff:
        logging.info(f"Applying first difference to '{target_column}' series.")
        series = series.diff().dropna()
        if series.empty:
            logging.warning(f"Series is empty after differencing and dropna for '{target_column}'.")
            return None
        logging.info(f"Series length after differencing and dropna: {len(series)}")
    elif pct_change:
        logging.info(f"Applying percentage change to '{target_column}' series.")
        series = series.pct_change().dropna()
        if series.empty:
            logging.warning(f"Series is empty after percentage change and dropna for '{target_column}'.")
            return None
        logging.info(f"Series length after percentage change and dropna: {len(series)}")

    # --- Calculate date ranges ---
    # Determine training period start date based on days or years
    train_period_unit = ""
    train_period_value = 0

    if n_train_days is not None and n_train_days > 0:
        train_start_date = split_date - relativedelta(days=n_train_days)
        train_period_unit = "days"
        train_period_value = n_train_days
        logging.info(f"Training period set to {n_train_days} days before {split_date.date()}.")
    elif n_train_years is not None and n_train_years > 0:
        # Ensure n_train_years is treated as int for relativedelta if it's a float (e.g. 10.0)
        train_start_date = split_date - relativedelta(years=int(n_train_years))
        train_period_unit = "years"
        train_period_value = int(n_train_years)
        logging.info(f"Training period set to {int(n_train_years)} years before {split_date.date()}.")
    else:
        if n_train_years is not None and n_train_years <=0 and (n_train_days is None or n_train_days <=0):
             logging.error("Training period (years or days) must be positive.")
             # Decide how to handle: return None, raise error, or use a default minimum.
             # For now, let's log and it will likely result in an empty train set.
             # A more robust solution would be to raise ValueError here.
        # If n_train_years is None (because days is also None, which shouldn't happen with defaults)
        # this indicates a logic flaw in how parameters are passed or defaults are set.
        # Given the defaults, n_train_years will usually have a value.
        # The primary goal is to give precedence to n_train_days.
        # If n_train_days is NOT set, then n_train_years (default 10) is used.
        # If n_train_years is explicitly set to 0 or negative, and days is not set, it's an issue.
        # Let's refine the condition for logging an error/warning:
        if not ((n_train_days is not None and n_train_days > 0) or \
                (n_train_years is not None and n_train_years > 0)):
            logging.error("Invalid training period: n_train_days or n_train_years must be positive. Using default 10 years if available.")
            # Fallback to a default if both are invalid, though args parsing should handle some of this.
            # This internal check is a safeguard.
            train_start_date = split_date - relativedelta(years=10) # Default fallback
            train_period_unit = "years"
            train_period_value = 10


    # Determine test period end date based on days or years
    test_period_unit = ""
    test_period_value = 0

    if n_test_days is not None and n_test_days > 0:
        test_end_date = split_date + relativedelta(days=n_test_days)
        test_period_unit = "days"
        test_period_value = n_test_days
        logging.info(f"Test period set to {n_test_days} days from {split_date.date()}.")
    elif n_test_years is not None and n_test_years > 0:
        test_end_date = split_date + relativedelta(years=int(n_test_years))
        test_period_unit = "years"
        test_period_value = int(n_test_years)
        logging.info(f"Test period set to {int(n_test_years)} years from {split_date.date()}.")
    else:
        logging.error("Invalid test period: n_test_days or n_test_years must be positive. Using default 1 year.")
        test_end_date = split_date + relativedelta(years=1)
        test_period_unit = "years"
        test_period_value = 1

    logging.info(f"Splitting data around reference date: {split_date.date()}")
    logging.info(f"Calculated train period start: {train_start_date.date()} (based on {train_period_value} {train_period_unit})")
    logging.info(f"Calculated test period end: {test_end_date.date()} (based on {test_period_value} {test_period_unit})")

    # --- Split the data ---
    # Ensure train data ends *before* or *at* split_date, and starts *after* train_start_date.
    # The original logic for train was: (series.index <= split_date) & (series.index > train_start_date)
    # This means data *on* split_date could be in train if it's the last point.
    # Typically, for forecasting, train data is strictly *before* the test period.
    # Let's adjust: train ends *before* split_date. Test starts *at* split_date.

    # Train data: from train_start_date (exclusive) up to split_date (exclusive)
    train = series.loc[(series.index > train_start_date) & (series.index < split_date)].copy()
    
    # Test data: from split_date (inclusive) up to test_end_date (exclusive)
    test = series.loc[(series.index >= split_date) & (series.index < test_end_date)].copy()

    # --- Validate and log splits ---
    if train.empty:
        logging.error(f"Training set is empty. Period: ({train_start_date.date()}, {split_date.date()}). Check data availability and requested range.")
        raise(ValueError("Training set is empty"))
    else:
        logging.info(f"Training set created with {len(train)} points from {train.index.min().date()} to {train.index.max().date()}")

    if test.empty:
        logging.warning(f"Test set is empty. Period: [{split_date.date()}, {test_end_date.date()}). Check data availability and requested range.")
        raise(ValueError("Test set is empty"))
    else:
         logging.info(f"Test set created with {len(test)} points from {test.index.min().date()} to {test.index.max().date()}")
    
    # For returning, it might be useful to also return the unit and value used for training period
    # but the function signature is fixed. This info will be in logs and pipeline.py can store it.
    return train, test

# --- Example Usage ---
if __name__ == "__main__":
    # Create dummy data dir and file for testing
    dummy_data_path = Path("data/DUMMY/DUMMY_1d.csv")
    dummy_data_path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(start="2010-01-01", end="2024-01-01", freq="B") # Business days
    dummy_df = pd.DataFrame({
        "Open": range(len(dates)),
        "Close": range(len(dates))
    }, index=dates)
    dummy_df.index.name = "Date"
    dummy_df.to_csv(dummy_data_path)
    
    logging.info("--- Testing with n_train_days ---")
    train_days, test_days = prepare_data_for_modeling(
        ticker="DUMMY",
        timefreq="1d",
        rel_date="2022-01-01",
        n_train_days=365, # Use 1 year of days
        n_train_years=5,   # This should be ignored
        n_test_years=1
    )
    if train_days is not None and test_days is not None:
        print(f"Train (days): {len(train_days)} obs, Test (days): {len(test_days)} obs")
        if not train_days.empty:
             print(f"Train (days) range: {train_days.index.min()} to {train_days.index.max()}")
        if not test_days.empty:
            print(f"Test (days) range: {test_days.index.min()} to {test_days.index.max()}")


    logging.info("\n--- Testing with n_train_years (days not specified) ---")
    train_years, test_years = prepare_data_for_modeling(
        ticker="DUMMY",
        timefreq="1d",
        rel_date="2022-01-01",
        n_train_years=2,
        n_train_days=None, # Explicitly None
        n_test_years=1
    )
    if train_years is not None and test_years is not None:
        print(f"Train (years): {len(train_years)} obs, Test (years): {len(test_years)} obs")
        if not train_years.empty:
            print(f"Train (years) range: {train_years.index.min()} to {train_years.index.max()}")
        if not test_years.empty:
            print(f"Test (years) range: {test_years.index.min()} to {test_years.index.max()}")

    logging.info("\n--- Testing with n_train_years (days is 0) ---")
    train_years_days_zero, test_years_days_zero = prepare_data_for_modeling(
        ticker="DUMMY",
        timefreq="1d",
        rel_date="2022-01-01",
        n_train_years=3,
        n_train_days=0, # Days is 0, so years should be used
        n_test_years=1
    )
    if train_years_days_zero is not None and test_years_days_zero is not None:
        print(f"Train (years, days=0): {len(train_years_days_zero)} obs, Test (years, days=0): {len(test_years_days_zero)} obs")
        if not train_years_days_zero.empty:
            print(f"Train (years, days=0) range: {train_years_days_zero.index.min()} to {train_years_days_zero.index.max()}")

    logging.info("\n--- Testing with pct_change=True ---")
    train_pct, test_pct = prepare_data_for_modeling(
        ticker="DUMMY",
        timefreq="1d",
        rel_date="2022-01-01",
        n_train_years=2,
        n_test_years=1,
        pct_change=True
    )
    if train_pct is not None and test_pct is not None:
        print(f"Train (pct_change): {len(train_pct)} obs, Test (pct_change): {len(test_pct)} obs")
        if not train_pct.empty:
            print(f"Train (pct_change) range: {train_pct.index.min()} to {train_pct.index.max()}")
            print(f"Train (pct_change) sample values: {train_pct.head().values}")

    logging.info("\n--- Testing mutual exclusivity (diff=True, pct_change=True) ---")
    try:
        prepare_data_for_modeling(
            ticker="DUMMY",
            timefreq="1d",
            rel_date="2022-01-01",
            n_train_years=2,
            n_test_years=1,
            diff=True,
            pct_change=True
        )
        print("ERROR: Should have raised ValueError for mutual exclusivity!")
    except ValueError as e:
        print(f"SUCCESS: Correctly raised ValueError: {e}")

    logging.info("\n--- Testing n_test_days parameter ---")
    train_test_days, test_test_days = prepare_data_for_modeling(
        ticker="DUMMY",
        timefreq="1d",
        rel_date="2022-01-01",
        n_train_days=365,
        n_test_days=180,  # Test with 180 days
        n_test_years=5    # This should be ignored
    )
    if train_test_days is not None and test_test_days is not None:
        print(f"Train: {len(train_test_days)} obs, Test: {len(test_test_days)} obs")
        print(f"Test range: {test_test_days.index.min()} to {test_test_days.index.max()}")
        # Verify test period is approximately 180 days (accounting for weekends/holidays)
        test_days_actual = (test_test_days.index.max() - test_test_days.index.min()).days
        print(f"Test period duration: {test_days_actual} calendar days (expected ~180)")
        if test_days_actual >= 170 and test_days_actual <= 270:  # Allow some margin for business days
            print("SUCCESS: Test period duration is reasonable for 180 business days")
        else:
            print(f"WARNING: Test period duration {test_days_actual} seems off for 180 days request")

    logging.info("\n--- Testing n_test_days precedence over n_test_years ---")
    train_precedence, test_precedence = prepare_data_for_modeling(
        ticker="DUMMY",
        timefreq="1d",
        rel_date="2022-01-01",
        n_train_years=2,
        n_test_days=90,    # Should use this
        n_test_years=2     # Should ignore this
    )
    if train_precedence is not None and test_precedence is not None:
        print(f"Test range: {test_precedence.index.min()} to {test_precedence.index.max()}")
        test_days_actual = (test_precedence.index.max() - test_precedence.index.min()).days
        print(f"Test period duration: {test_days_actual} calendar days (expected ~90)")
        if test_days_actual >= 85 and test_days_actual <= 135:  # Allow margin for business days
            print("SUCCESS: n_test_days took precedence over n_test_years")
        else:
            print(f"WARNING: Test period seems incorrect, got {test_days_actual} days")

    logging.info("\n--- Testing n_test_years when n_test_days is None ---")
    train_years_test, test_years_test = prepare_data_for_modeling(
        ticker="DUMMY",
        timefreq="1d",
        rel_date="2020-01-01",
        n_train_years=2,
        n_test_days=None,  # Explicitly None
        n_test_years=1     # Should use this (1 year)
    )
    if train_years_test is not None and test_years_test is not None:
        print(f"Test range: {test_years_test.index.min()} to {test_years_test.index.max()}")
        test_days_actual = (test_years_test.index.max() - test_years_test.index.min()).days
        print(f"Test period duration: {test_days_actual} calendar days (expected ~365)")
        if test_days_actual >= 350 and test_days_actual <= 380:  # Allow margin
            print("SUCCESS: n_test_years used when n_test_days is None")
        else:
            print(f"WARNING: Test period seems incorrect for 1 year, got {test_days_actual} days")

    logging.info("\n--- Testing large n_test_days (600 days for crash recovery) ---")
    train_crash, test_crash = prepare_data_for_modeling(
        ticker="DUMMY",
        timefreq="1d",
        rel_date="2018-01-01",
        n_train_days=250,
        n_test_days=600  # Test 600 days recovery period
    )
    if train_crash is not None and test_crash is not None:
        print(f"Train: {len(train_crash)} obs, Test: {len(test_crash)} obs")
        print(f"Test range: {test_crash.index.min()} to {test_crash.index.max()}")
        test_days_actual = (test_crash.index.max() - test_crash.index.min()).days
        print(f"Test period duration: {test_days_actual} calendar days (expected ~600)")
        if test_days_actual >= 550 and test_days_actual <= 850:  # Allow margin for business days
            print("SUCCESS: Can handle large test periods like 600 days for crash recovery")
        else:
            print(f"WARNING: Test period seems incorrect, got {test_days_actual} days")

    # Cleanup dummy file
    import os
    os.remove(dummy_data_path)
    os.rmdir(dummy_data_path.parent)
