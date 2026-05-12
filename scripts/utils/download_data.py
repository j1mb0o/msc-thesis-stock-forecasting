import yfinance as yf
import pandas as pd
from pathlib import Path
import logging
from typing import Union, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DATA_DIR = Path("data")


def _validate_tickers(ticker_input: Union[str, List[str]]) -> List[str]:
    """Validates and standardizes the ticker input to a list of uppercase strings."""
    if isinstance(ticker_input, str):
        tickers = [ticker_input]
    elif isinstance(ticker_input, list):
        if not all(isinstance(t, str) for t in ticker_input):
            raise ValueError("If ticker is a list, all elements must be strings.")
        tickers = ticker_input
    else:
        raise ValueError("Ticker must be a string or a list of strings.")
    return [t.upper() for t in tickers]


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
    """
    try:
        tickers = _validate_tickers(ticker_input)
    except ValueError as e:
        logging.error(f"Invalid ticker input: {e}")
        return

    tickers_to_download = []
    ticker_file_paths = {}

    for tick in tickers:
        ticker_dir = base_dir / tick
        file_path = ticker_dir / f"{tick}_{timefreq}.csv"
        ticker_file_paths[tick] = file_path

        if file_path.exists():
            logging.info(f"Data for {tick} ({timefreq}) already exists at {file_path}. Skipping download.")
        else:
            logging.info(f"Data for {tick} ({timefreq}) does not exist. Will attempt download.")
            tickers_to_download.append(tick)
            ticker_dir.mkdir(parents=True, exist_ok=True)

    if not tickers_to_download:
        logging.info("All requested ticker data already exists locally.")
        return

    logging.info(f"Attempting to download data for: {', '.join(tickers_to_download)}")
    try:
        # yfinance returns a multi-index DataFrame for multiple tickers under group_by='ticker',
        # but a single-index DataFrame when only one ticker comes back.
        downloaded_data = yf.download(
            tickers=tickers_to_download,
            interval=timefreq,
            period=period,
            group_by='ticker'
        )

        if downloaded_data.empty:
             logging.warning("Download attempt returned no data.")
             return

    except Exception as e:
        logging.error(f"An error occurred during download: {e}")
        return

    for tick in tickers_to_download:
        file_path = ticker_file_paths[tick]
        ticker_data: Optional[pd.DataFrame] = None

        try:
            if isinstance(downloaded_data.columns, pd.MultiIndex):
                if tick in downloaded_data.columns.get_level_values(0):
                    ticker_data = downloaded_data.xs(tick, level=0, axis=1)
                else:
                    logging.warning(f"Ticker {tick} was requested but not found in the downloaded multi-index data. Skipping save.")
                    continue
            elif len(tickers_to_download) == 1 and tick == tickers_to_download[0]:
                ticker_data = downloaded_data
            else:
                logging.warning(f"Could not reliably extract data for {tick} from download result. Structure: {type(downloaded_data.columns)}. Skipping save.")
                continue

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
            logging.warning(f"KeyError extracting data for {tick}. It might not have been downloaded successfully. Skipping save.")
        except Exception as e:
            logging.error(f"An error occurred processing or saving data for {tick}: {e}")


if __name__ == "__main__":
    download_data(["MSFT", "AAPL", "GOOGL"], "1d")
