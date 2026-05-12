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

    ticker_dir = base_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    file_path = ticker_dir / f"{ticker}_{timeframe}_v2.csv"

    if file_path.exists():
        logging.info(
            f"Data for {ticker} ({timeframe}) already exists at {file_path}. "
            "Delete the file if you want to re-download."
        )
        return pd.read_csv(file_path, index_col=0, parse_dates=True)

    if end_date is None:
        end_dt = datetime.now()
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if start_date is None:
        # Alpaca free tier serves ~5y of intraday history; go back 10y to take whatever is available.
        start_dt = end_dt - timedelta(days=365 * 10)
        logging.info(
            f"No start date specified. Requesting maximum available history "
            f"(from {start_dt.date()} to {end_dt.date()})"
        )
    else:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")

    logging.info(f"Initializing Alpaca client for {ticker}")
    client = StockHistoricalDataClient(api_key, secret_key)

    try:
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

        bars = client.get_stock_bars(request_params)
        df = bars.df

        if df.empty:
            logging.warning(f"No data returned for {ticker}")
            return None

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(ticker, level="symbol")

        df = df.reset_index()
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        df.columns = df.columns.str.lower()

        logging.info(
            f"Downloaded {len(df)} bars from {df.index[0]} to {df.index[-1]}"
        )

        df.to_csv(file_path)
        logging.info(f"Data for {ticker} ({timeframe}) saved to {file_path}")

        return df

    except Exception as e:
        logging.error(f"Error downloading data for {ticker}: {e}")
        return None


if __name__ == "__main__":
    start_date = "2014-01-01"
    end_date = "2024-01-01"

    df = download_alpaca_data(
        ticker="MSFT",
        timeframe="15min",
        start_date="2023-01-01",
        end_date=end_date,
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
