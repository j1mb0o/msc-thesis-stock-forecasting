import argparse

def get_pipeline_arguments():
    parser = argparse.ArgumentParser(description="Run forecasting pipeline.")
    parser.add_argument(
        "--ticker",
        type=str,
        default="MSFT",
        help="Stock ticker symbol (e.g., MSFT)",
    )
    parser.add_argument(
        "--timefreq", type=str, default="1d", help="Data time frequency (e.g., 1d, 1h)"
    )
    parser.add_argument(
        "--test_years",
        type=float,
        default=1,
        help="Number of years to forecast for the test set.",
    )
    parser.add_argument(
        "--train_last_n_years", # Renamed for clarity
        type=float,
        default=10,
        help="Number of past years to use for training. Default is 10 years. This is overridden if --train_last_n_days is specified.",
    )
    parser.add_argument(
        "--train_last_n_days", # New argument for days
        type=int,
        default=None, # Default to None, so it's not used unless specified
        help="Number of past days to use for training. If specified, this overrides --train_last_n_years.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["naive", "arima", "fm", "sundial"],
        default="naive",
        help="Forecasting method (naive, arima, fm, sundial)",
    )
    parser.add_argument(
        "--target_column", type=str, default="Open", help="Target column to forecast"
    )
    parser.add_argument(
        "--horizon_len", type=int, default=1, help="Horizon length"
    )
    parser.add_argument(
        "--diff", action="store_true", help="Apply differencing to the target series."
    )
    parser.add_argument(
        "--split_date", type=str, default="2023-01-01", help="Split date (YYYY-MM-DD) for train/test separation."
    )
    parser.add_argument(
        "--exp_name", type=str, default=None, help="Experiment name for organizing results and configs."
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Example of how to access the new arguments
    args = get_pipeline_arguments()
    print(f"Ticker: {args.ticker}")
    print(f"Time Frequency: {args.timefreq}")
    print(f"Test Years: {args.test_years}")
    print(f"Train Last N Years: {args.train_last_n_years}")
    print(f"Train Last N Days: {args.train_last_n_days}") # New argument
    print(f"Method: {args.method}")
    print(f"Target Column: {args.target_column}")
    print(f"Horizon Length: {args.horizon_len}")
    print(f"Differencing: {args.diff}")
    print(f"Split Date: {args.split_date}")
    print(f"Experiment Name: {args.exp_name}")
