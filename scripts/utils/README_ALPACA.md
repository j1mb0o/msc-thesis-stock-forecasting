# Alpaca Data Download

This guide explains how to use the `download_alpaca_data.py` script to download 15-minute (or other intraday) stock data using the Alpaca API.

## Prerequisites

### 1. Install Dependencies

The `alpaca-py` package has been added to `pyproject.toml`. Install it with:

```bash
pip install -e "."
```

Or install just the Alpaca package:

```bash
pip install alpaca-py
```

### 2. Get Alpaca API Credentials

1. Sign up for a free Alpaca account at https://alpaca.markets/
2. Navigate to your dashboard and generate API keys (paper trading keys work fine for historical data)
3. You'll receive:
   - API Key ID
   - Secret Key

### 3. Set Environment Variables

Export your API credentials as environment variables:

```bash
export ALPACA_API_KEY="your_api_key_here"
export ALPACA_SECRET_KEY="your_secret_key_here"
```

To make these permanent, add them to your `~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`:

```bash
echo 'export ALPACA_API_KEY="your_api_key_here"' >> ~/.zshrc
echo 'export ALPACA_SECRET_KEY="your_secret_key_here"' >> ~/.zshrc
source ~/.zshrc
```

## Usage

### Basic Usage (Download Maximum Available Data)

```python
from scripts.utils.download_alpaca_data import download_alpaca_data

# Downloads maximum available 15-minute data for MSFT
df = download_alpaca_data(ticker="MSFT", timeframe="15min")
```

### Command Line Usage

```bash
cd /Users/dimitris/LU/Thesis/thesis-code-new
python scripts/utils/download_alpaca_data.py
```

This will download maximum available 15-minute data for MSFT by default.

### Custom Parameters

```python
from scripts.utils.download_alpaca_data import download_alpaca_data
from datetime import datetime, timedelta

# Download specific date range
df = download_alpaca_data(
    ticker="AAPL",
    timeframe="15min",
    start_date="2023-01-01",
    end_date="2024-01-01"
)

# Download different timeframes
df_5min = download_alpaca_data("MSFT", timeframe="5min")
df_1hour = download_alpaca_data("MSFT", timeframe="1hour")
```

### Supported Timeframes

- `"1min"` - 1-minute bars
- `"5min"` - 5-minute bars
- `"15min"` - 15-minute bars (recommended for this project)
- `"30min"` - 30-minute bars
- `"1hour"` - 1-hour bars
- `"1day"` - Daily bars

## Data Limits

Alpaca provides different amounts of historical data based on your subscription:

- **Free tier**: Typically 5+ years of intraday data
- **Unlimited plan**: Full historical intraday data (varies by symbol)

The script requests 10 years of history by default to ensure you get the maximum available data.

## Data Storage

Downloaded data is saved to:
```
data/{TICKER}/{TICKER}_{timeframe}.csv
```

For example:
```
data/MSFT/MSFT_15min.csv
```

The CSV file contains:
- `timestamp` (index): DateTime of the bar
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume
- `trade_count`: Number of trades (if available)
- `vwap`: Volume-weighted average price (if available)

## Integration with Existing Pipeline

The downloaded data follows the same format as the existing `download_data.py` (yfinance) output, so it can be used with the existing pipeline:

```python
from scripts.utils.download_alpaca_data import download_alpaca_data
from scripts.utils.model_data_prep import prepare_data_for_modeling

# Download 15-minute data
download_alpaca_data("MSFT", "15min")

# Use with existing pipeline
train_data, test_data, split_date = prepare_data_for_modeling(
    ticker="MSFT",
    timefreq="15min",
    split_strategy="days",
    train_days=250,
    test_days=30
)
```

## Troubleshooting

### Error: "Alpaca API credentials not provided"

Make sure you've set the environment variables correctly:
```bash
echo $ALPACA_API_KEY
echo $ALPACA_SECRET_KEY
```

### Error: "alpaca-py is not installed"

Install the package:
```bash
pip install alpaca-py
```

### No data returned

- Check that the ticker symbol is valid
- Verify your date range is reasonable
- Ensure your API keys are active and have data permissions

### Rate Limits

Alpaca has rate limits (200 requests/minute for free tier). The script downloads all data in a single request, so this shouldn't be an issue for normal use.

## Example Output

```
2024-01-15 10:30:00 - INFO - No start date specified. Requesting maximum available history (from 2014-01-15 to 2024-01-15)
2024-01-15 10:30:00 - INFO - Initializing Alpaca client for MSFT
2024-01-15 10:30:00 - INFO - Downloading MSFT data from 2014-01-15 to 2024-01-15 with 15min bars...
2024-01-15 10:30:05 - INFO - Downloaded 45678 bars from 2019-01-02 09:30:00 to 2024-01-15 16:00:00
2024-01-15 10:30:05 - INFO - Data for MSFT (15min) saved to data/MSFT/MSFT_15min.csv

Downloaded 45678 bars
Date range: 2019-01-02 09:30:00 to 2024-01-15 16:00:00
```
