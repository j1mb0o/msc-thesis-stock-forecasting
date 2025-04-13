
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
        help="Numm of years to forecast",
    )
    parser.add_argument(
        "--train_last_n",
        type=float,
        default=10,
        help="Proportion of train data to keep",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["naive", "arima", "fm"],
        default="naive",
        help="Forecasting method (naive, arima)",
    )
    parser.add_argument(
        "--target_column", type=str, default="Open", help="Target column to forecast"
    )
    parser.add_argument(
        "--horizon_len", type=int, default=1, help="Horizon length"
    )

    parser.add_argument(
        "--diff", action="store_true", help="Number of times to diff the dataset"
    )

    parser.add_argument(
        "--split_date", type=str, default="2023-01-01", help="Split date"
    )
    args = parser.parse_args()
    return args
