import yfinance as yf
import pandas as pd
from pathlib import Path
import logging
from typing import Union, List, Optional

# --- Configuration ---
# Set up basic logging
# In a real application, you might configure this more extensively
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the base directory for saving data
BASE_DATA_DIR = Path("../data")

# --- Helper Function (Optional but Recommended) ---
def _validate_tickers(ticker_input: Union[str, List[str]]) -> List[str]:
    """Validates and standardizes the ticker input to a list of strings."""
    if isinstance(ticker_input, str):
        tickers = [ticker_input]
    elif isinstance(ticker_input, list):
        if not all(isinstance(t, str) for t in ticker_input):
            raise ValueError("If ticker is a list, all elements must be strings.")
        tickers = ticker_input
    else:
        raise ValueError("Ticker must be a string or a list of strings.")
    # Optional: Convert to uppercase or perform other cleaning
    return [t.upper() for t in tickers]

# --- Main Download Function ---
def download_data(
    ticker_input: Union[str, List[str]],
    timefreq: str,
    base_dir: Path = BASE_DATA_DIR,
    period: str = "max"
) -> None:
    """
    Downloads historical stock data using yfinance and saves it locally.

    Checks if data for a specific ticker and time frequency already exists
    before downloading. Downloads all required tickers efficiently in a
    single batch.

    Args:
        ticker_input: A single ticker string (e.g., "MSFT") or a list of
                      ticker strings (e.g., ["AAPL", "MSFT"]).
        timefreq: The time frequency/interval for the data
                  (e.g., "1d", "1h", "1wk", "1mo"). See yfinance docs
                  for valid intervals.
        base_dir: The base directory Path object where data will be stored
                  (defaults to BASE_DATA_DIR). Data will be saved in
                  subdirectories: base_dir / ticker / ticker_timefreq.csv.
        period: The period for which to download data (e.g., "1y", "max").
                Defaults to "max".

    Returns:
        None. Saves data to files.

    Raises:
        ValueError: If ticker_input format is invalid.
        # Other exceptions might be raised by yfinance or pandas operations.
    """
    try:
        tickers = _validate_tickers(ticker_input)
    except ValueError as e:
        logging.error(f"Invalid ticker input: {e}")
        return # Or re-raise the exception if preferred

    tickers_to_download = []
    ticker_file_paths = {} # Store expected file path for each ticker

    # 1. Check which tickers actually need downloading
    for tick in tickers:
        ticker_dir = base_dir / tick
        file_path = ticker_dir / f"{tick}_{timefreq}.csv"
        ticker_file_paths[tick] = file_path

        if file_path.exists():
            logging.info(f"Data for {tick} ({timefreq}) already exists at {file_path}. Skipping download.")
        else:
            logging.info(f"Data for {tick} ({timefreq}) does not exist. Will attempt download.")
            tickers_to_download.append(tick)
            # Ensure directory exists for saving later
            ticker_dir.mkdir(parents=True, exist_ok=True)

    # 2. Download data for missing tickers (if any) in one batch
    if not tickers_to_download:
        logging.info("All requested ticker data already exists locally.")
        return

    logging.info(f"Attempting to download data for: {', '.join(tickers_to_download)}")
    try:
        # yfinance downloads multiple tickers efficiently in one call
        # It returns a multi-index DataFrame if multiple tickers are successful
        # and group_by='ticker' (default). If only one ticker is requested
        # or only one succeeds, it might return a single-index DataFrame.
        downloaded_data = yf.download(
            tickers=tickers_to_download,
            interval=timefreq,
            period=period,
            group_by='ticker' # Keep default grouping for multi-index columns
        )

        if downloaded_data.empty:
             logging.warning("Download attempt returned no data.")
             return

    except Exception as e:
        # Catch potential errors during download (network, invalid ticker, etc.)
        logging.error(f"An error occurred during download: {e}")
        # Depending on requirements, you might want to retry or handle partially successful downloads
        return # Stop processing if the batch download fails

    # 3. Process and save the downloaded data
    for tick in tickers_to_download:
        file_path = ticker_file_paths[tick]
        ticker_data: Optional[pd.DataFrame] = None

        try:
            # Extract data for the current ticker
            # Check if downloaded_data has multi-index columns
            if isinstance(downloaded_data.columns, pd.MultiIndex):
                 # Check if the specific ticker was successfully downloaded (might have failed partially)
                if tick in downloaded_data.columns.get_level_values(0):
                     # Use .xs only if multi-index and ticker exists
                    ticker_data = downloaded_data.xs(tick, level=0, axis=1)
                else:
                    logging.warning(f"Ticker {tick} was requested but not found in the downloaded multi-index data. Skipping save.")
                    continue # Skip to next ticker
            # Handle case where only one ticker was downloaded (returns simple columns)
            elif len(tickers_to_download) == 1 and tick == tickers_to_download[0]:
                 # Assume the entire frame is for this single ticker
                 ticker_data = downloaded_data
            else:
                # This case might occur if download failed partially or structure is unexpected
                 logging.warning(f"Could not reliably extract data for {tick} from download result. Structure: {type(downloaded_data.columns)}. Skipping save.")
                 continue # Skip to next ticker

            # Clean and save the extracted data
            if ticker_data is not None and not ticker_data.empty:
                ticker_data = ticker_data.dropna(how="all")
                if not ticker_data.empty:
                    ticker_data.to_csv(file_path)
                    logging.info(f"Data for {tick} ({timefreq}) saved to {file_path}")
                else:
                    logging.info(f"No valid data rows remained for {tick} ({timefreq}) after dropna. File not saved.")
            elif ticker_data is not None and ticker_data.empty:
                 logging.info(f"Downloaded data for {tick} ({timefreq}) was initially empty. File not saved.")


        except KeyError:
             # This might happen if a ticker was in tickers_to_download but yfinance failed to retrieve it
             # within the multi-index structure (less likely with the check above, but good practice)
            logging.warning(f"KeyError extracting data for {tick}. It might not have been downloaded successfully. Skipping save.")
        except Exception as e:
            logging.error(f"An error occurred processing or saving data for {tick}: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    download_data(["MSFT", "AAPL", "GOOGL"], "1d")
