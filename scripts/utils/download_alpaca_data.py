import os
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except ImportError:
    raise ImportError(
        "alpaca-py is not installed. Install it with: pip install alpaca-py"
    )

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DATA_DIR = Path("data")


def download_alpaca_data(
    ticker: str = "MSFT",
    timeframe: str = "15min",
    base_dir: Path = BASE_DATA_DIR,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Downloads historical stock data using Alpaca API and saves it locally.

    Alpaca provides up to several years of historical intraday data depending
    on your subscription level. Free tier typically provides recent data.

    Args:
        ticker: Stock ticker symbol (e.g., "MSFT").
        timeframe: Time interval for bars. Supported values:
                   "1min", "5min", "15min", "30min", "1hour", "1day".
        base_dir: Base directory Path where data will be stored.
                  Data saved as: base_dir / ticker / ticker_timeframe.csv
        api_key: Alpaca API key. If None, reads from ALPACA_API_KEY env variable.
        secret_key: Alpaca secret key. If None, reads from ALPACA_SECRET_KEY env variable.
        start_date: Start date for data in 'YYYY-MM-DD' format.
                    If None, downloads maximum available history (typically 5-7 years for intraday).
        end_date: End date for data in 'YYYY-MM-DD' format.
                  If None, uses current date.

    Returns:
        pd.DataFrame: Downloaded data with columns [open, high, low, close, volume]
                      and DatetimeIndex. Returns None if download fails.

    Raises:
        ValueError: If API credentials are missing or timeframe is invalid.

    Example:
        # Set environment variables first:
        # export ALPACA_API_KEY="your_api_key"
        # export ALPACA_SECRET_KEY="your_secret_key"

        df = download_alpaca_data("MSFT", "15min")
    """
    # Validate and get API credentials
    if api_key is None:
        api_key = os.getenv("ALPACA_API_KEY")
    if secret_key is None:
        secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca API credentials not provided. Set ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY environment variables or pass them as arguments."
        )

    ticker = ticker.upper()

    # Map timeframe string to Alpaca TimeFrame enum
    timeframe_map = {
        "1min": TimeFrame.Minute,
        "5min": TimeFrame(5, TimeFrameUnit.Minute),
        "15min": TimeFrame(15, TimeFrameUnit.Minute),
        "30min": TimeFrame(30, TimeFrameUnit.Minute),
        "1hour": TimeFrame.Hour,
        "1day": TimeFrame.Day,
    }

    if timeframe not in timeframe_map:
        raise ValueError(
            f"Invalid timeframe '{timeframe}'. Supported values: {list(timeframe_map.keys())}"
        )

    alpaca_timeframe = timeframe_map[timeframe]

    # Set up file path
    ticker_dir = base_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    file_path = ticker_dir / f"{ticker}_{timeframe}.csv"

    # Check if data already exists
    if file_path.exists():
        logging.info(
            f"Data for {ticker} ({timeframe}) already exists at {file_path}. "
            "Delete the file if you want to re-download."
        )
        return pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Set date range - get maximum available history by default
    if end_date is None:
        end_dt = datetime.now()
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if start_date is None:
        # Alpaca provides different history lengths based on subscription
        # Free tier: typically 5 years for intraday data
        # To get maximum data, go back far enough
        start_dt = end_dt - timedelta(days=365 * 10)  # Request 10 years back
        logging.info(
            f"No start date specified. Requesting maximum available history "
            f"(from {start_dt.date()} to {end_dt.date()})"
        )
    else:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")

    # Initialize Alpaca client
    logging.info(f"Initializing Alpaca client for {ticker}")
    client = StockHistoricalDataClient(api_key, secret_key)

    try:
        # Create request
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=alpaca_timeframe,
            start=start_dt,
            end=end_dt,
        )

        logging.info(
            f"Downloading {ticker} data from {start_dt.date()} to {end_dt.date()} "
            f"with {timeframe} bars..."
        )

        # Fetch data
        bars = client.get_stock_bars(request_params)

        # Convert to DataFrame
        df = bars.df

        if df.empty:
            logging.warning(f"No data returned for {ticker}")
            return None

        # If multi-index (symbol level), extract the specific ticker
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(ticker, level="symbol")

        # Reset index to have timestamp as a column, then set it back
        # This ensures proper datetime index handling
        df = df.reset_index()
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        # Rename columns to match project convention (lowercase)
        df.columns = df.columns.str.lower()

        # Log data range actually received
        logging.info(
            f"Downloaded {len(df)} bars from {df.index[0]} to {df.index[-1]}"
        )

        # Save to CSV
        df.to_csv(file_path)
        logging.info(f"Data for {ticker} ({timeframe}) saved to {file_path}")

        return df

    except Exception as e:
        logging.error(f"Error downloading data for {ticker}: {e}")
        return None


if __name__ == "__main__":
    # Example usage - downloads maximum available 15-minute data for MSFT
    # Make sure to set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables

    # For free tier, request data ending at least 15 minutes ago
    # Or specify a date range that ends before today
    end_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    df = download_alpaca_data(
        ticker="MSFT",
        timeframe="15min",
        end_date=end_date  # End yesterday to avoid real-time data restrictions
    )

    if df is not None:
        print(f"\nDownloaded {len(df)} bars")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nLast few rows:")
        print(df.tail())
        print(f"\nData info:")
        print(df.info())
