"""
Script to identify and visualize significant market crashes using hourly stock data.

This script loads hourly stock data, identifies major drawdown periods, and plots
them for analysis in the market disruption resilience research question (RQ3).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_hourly_data(ticker: str) -> pd.DataFrame:
    """
    Load hourly stock data from the data directory.

    Args:
        ticker: Stock ticker symbol

    Returns:
        DataFrame with hourly price data
    """
    data_path = Path(f"data/{ticker}/{ticker}_1h.csv")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logging.info(f"Loading data from {data_path}")
    # Read tab-delimited file
    df = pd.read_csv(data_path, sep='\t')

    # Combine DATE and TIME columns if they exist
    if '<DATE>' in df.columns and '<TIME>' in df.columns:
        df['Datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
        df.set_index('Datetime', inplace=True)
        # Rename columns to standard format
        df.rename(columns={
            '<OPEN>': 'Open',
            '<HIGH>': 'High',
            '<LOW>': 'Low',
            '<CLOSE>': 'Close',
            '<VOL>': 'Volume'
        }, inplace=True)
    elif 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        # First column is likely the datetime
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)

    return df


def calculate_drawdown(prices: pd.Series) -> pd.Series:
    """
    Calculate drawdown from running maximum.

    Args:
        prices: Price series

    Returns:
        Drawdown series (negative values indicate drawdown percentage)
    """
    running_max = prices.expanding().max()
    drawdown = (prices - running_max) / running_max * 100
    return drawdown


def identify_crash_periods(drawdown: pd.Series, threshold: float = -10.0) -> pd.DataFrame:
    """
    Identify crash periods where drawdown exceeds threshold.

    Args:
        drawdown: Drawdown series
        threshold: Drawdown threshold to identify crashes (negative percentage)

    Returns:
        DataFrame with crash periods (start, end, max_drawdown)
    """
    in_crash = drawdown < threshold
    crash_periods = []

    start = None
    for i, (timestamp, is_crash) in enumerate(in_crash.items()):
        if is_crash and start is None:
            start = timestamp
        elif not is_crash and start is not None:
            end = drawdown.index[i-1]
            max_dd = drawdown[start:end].min()
            crash_periods.append({
                'start': start,
                'end': end,
                'max_drawdown': max_dd,
                'duration_days': (end - start).total_seconds() / 86400
            })
            start = None

    # Handle case where crash extends to end of data
    if start is not None:
        end = drawdown.index[-1]
        max_dd = drawdown[start:end].min()
        crash_periods.append({
            'start': start,
            'end': end,
            'max_drawdown': max_dd,
            'duration_days': (end - start).total_seconds() / 86400
        })

    return pd.DataFrame(crash_periods)


def highlight_crash_periods(ax, crash_periods: pd.DataFrame, show_labels: bool = True):
    """
    Highlight crash periods on the plot.

    Args:
        ax: Matplotlib axis
        crash_periods: DataFrame with crash periods
        show_labels: Whether to show labels for each crash period
    """
    # Use different colors for different severity levels
    for idx, crash in crash_periods.iterrows():
        # Determine color based on severity
        if crash['max_drawdown'] < -30:
            color = 'darkred'
            alpha = 0.25
            severity = 'Severe'
        elif crash['max_drawdown'] < -20:
            color = 'red'
            alpha = 0.2
            severity = 'Major'
        else:
            color = 'orange'
            alpha = 0.15
            severity = 'Moderate'

        ax.axvspan(crash['start'], crash['end'],
                  color=color, alpha=alpha, linewidth=0)

        # Add label for major crashes (optional)
        if show_labels and crash['max_drawdown'] < -20:
            mid_point = crash['start'] + (crash['end'] - crash['start']) / 2
            ax.text(mid_point, ax.get_ylim()[1] * 0.95,
                   f"{crash['max_drawdown']:.1f}%",
                   ha='center', va='top', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))


def plot_crashes(ticker: str, prices: pd.Series, drawdown: pd.Series,
                 crash_periods: pd.DataFrame, output_path: str = None,
                 start_year: int = None):
    """
    Create visualization of prices, drawdown, and crash periods.

    Args:
        ticker: Stock ticker symbol
        prices: Price series
        drawdown: Drawdown series
        crash_periods: DataFrame with crash periods
        output_path: Path to save figure (optional)
        start_year: Year to start the plot from (optional, filters data)
    """
    # Filter data if start_year is provided
    if start_year is not None:
        start_date = pd.Timestamp(f'{start_year}-01-01')
        prices = prices[prices.index >= start_date]
        drawdown = drawdown[drawdown.index >= start_date]
        crash_periods = crash_periods[crash_periods['start'] >= start_date].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 11), sharex=True)

    # Plot 1: Price with crash periods highlighted
    ax1.plot(prices.index, prices.values, linewidth=0.8, color='black',
             label='MSFT Price', alpha=0.8)

    # Highlight crash periods on price plot
    highlight_crash_periods(ax1, crash_periods, show_labels=True)

    ax1.set_ylabel('Price ($)', fontsize=13, fontweight='bold')
    ax1.set_title(f'{ticker} Price with Drawdown-Based Crash Periods (>10% decline from peak)\n' +
                  'Orange = Moderate (10-20%), Red = Major (20-30%), Dark Red = Severe (>30%)',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # Plot 2: Drawdown
    ax2.plot(drawdown.index, drawdown.values, linewidth=0.8, color='darkred',
             label='Drawdown from Peak', alpha=0.9)
    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     where=(drawdown.values < 0), color='red', alpha=0.2)

    # Highlight crash periods on drawdown plot
    highlight_crash_periods(ax2, crash_periods, show_labels=False)

    ax2.axhline(y=-10, color='orange', linestyle='--', linewidth=1.5,
               label='-10% Threshold (Moderate)', alpha=0.7)
    ax2.axhline(y=-20, color='red', linestyle='--', linewidth=1.5,
               label='-20% Threshold (Major)', alpha=0.7)
    ax2.axhline(y=-30, color='darkred', linestyle='--', linewidth=1.5,
               label='-30% Threshold (Severe)', alpha=0.7)

    ax2.set_ylabel('Drawdown (%)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax2.set_title('Drawdown from Running Maximum\n' +
                  '(0% = at all-time high, negative values = % below previous peak)',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower left', fontsize=11, framealpha=0.9, ncol=2)

    # Format x-axis with sparse, readable date labels.
    ax2.set_xlim(prices.index.min(), prices.index.max())
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Figure saved to {output_path}")



def main():
    parser = argparse.ArgumentParser(
        description='Identify and visualize market crashes from hourly stock data'
    )
    parser.add_argument(
        '--ticker',
        type=str,
        default='MSFT',
        help='Stock ticker to analyze (default: MSFT)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=-10.0,
        help='Drawdown threshold for crash identification (negative percentage)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='figures/market_crashes_hourly.pdf',
        help='Output path for figure'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top crashes to display in summary'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=2018,
        help='Year to start the plot from (default: 2018 to show context before 2019)'
    )

    args = parser.parse_args()

    # Load hourly data
    logging.info(f"Loading hourly data for {args.ticker}")
    try:
        data = load_hourly_data(args.ticker)
    except FileNotFoundError as e:
        logging.error(str(e))
        return

    # Use close price
    prices = data['Close']
    logging.info(f"Loaded {len(prices)} hourly price points from {prices.index[0]} to {prices.index[-1]}")

    # Calculate drawdown
    logging.info("Calculating drawdown...")
    drawdown = calculate_drawdown(prices)
    max_drawdown = drawdown.min()
    max_dd_date = drawdown.idxmin()
    logging.info(f"Maximum drawdown: {max_drawdown:.2f}% on {max_dd_date}")

    # Identify crash periods
    logging.info(f"Identifying crash periods (threshold: {args.threshold}%)...")
    crash_periods = identify_crash_periods(drawdown, threshold=args.threshold)

    if crash_periods.empty:
        logging.warning(f"No crashes found with threshold {args.threshold}%")
    else:
        # Sort by max drawdown
        crash_periods = crash_periods.sort_values('max_drawdown')

        logging.info(f"\n{'='*80}")
        logging.info(f"TOP {min(args.top_n, len(crash_periods))} MOST SIGNIFICANT CRASHES")
        logging.info(f"{'='*80}\n")

        for idx, crash in crash_periods.head(args.top_n).iterrows():
            logging.info(f"Crash #{idx + 1}:")
            logging.info(f"  Period: {crash['start'].strftime('%Y-%m-%d %H:%M')} to {crash['end'].strftime('%Y-%m-%d %H:%M')}")
            logging.info(f"  Max Drawdown: {crash['max_drawdown']:.2f}%")
            logging.info(f"  Duration: {crash['duration_days']:.1f} days")
            logging.info("")

    # Create visualization
    logging.info("Creating visualization...")
    plot_crashes(args.ticker, prices, drawdown, crash_periods,
                output_path=args.output, start_year=args.start_year)

    # Print summary statistics
    logging.info(f"\n{'='*80}")
    logging.info("SUMMARY STATISTICS")
    logging.info(f"{'='*80}")
    logging.info(f"Ticker: {args.ticker}")
    logging.info(f"Time Period: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    logging.info(f"Total Data Points: {len(prices):,}")
    logging.info(f"Max Drawdown: {max_drawdown:.2f}% on {max_dd_date.strftime('%Y-%m-%d %H:%M')}")
    logging.info(f"Number of Crashes (>{args.threshold}%): {len(crash_periods)}")
    if not crash_periods.empty:
        logging.info(f"Average Crash Duration: {crash_periods['duration_days'].mean():.1f} days")
        logging.info(f"Average Max Drawdown: {crash_periods['max_drawdown'].mean():.2f}%")


if __name__ == '__main__':
    main()
