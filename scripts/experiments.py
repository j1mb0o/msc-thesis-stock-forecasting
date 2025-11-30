import argparse
import subprocess
import time
from itertools import product

import numpy as np


def run_pipeline(
    ticker,
    timefreq,
    method,
    horizon_len,
    train_last_n,
    exp_name,
    days_flag=False,
    pct_change=False,
    split_date=None,
    test_n_days=None,
):
    """Runs the pipeline with the given parameters."""
    command = [
        "python",
        "scripts/pipeline.py",
        "--ticker",
        ticker,
        "--timefreq",
        timefreq,
        "--method",
        method,
        "--horizon_len",
        str(horizon_len),
        "--train_last_n_years" if not days_flag else "--train_last_n_days",
        str(train_last_n),
        "--exp_name",
        exp_name,
    ]
    if pct_change:
        command.append("--pct_change")
    if split_date:
        command.extend(["--split_date", split_date])
    if test_n_days:
        command.extend(["--test_n_days", str(test_n_days)])
    subprocess.run(command)
    # return command


parser = argparse.ArgumentParser(
    description="Run experiments for time series forecasting."
)
parser.add_argument(
    "--method", type=str, required=True, help="The forecasting method to use."
)
parser.add_argument(
    "--exp_name", type=str, required=False, help="The name of the experiment to run."
)
parser.add_argument(
    "--timefreq",
    default="1d",
    type=str,
    required=False,
    help="The time frequency of the data.",
)
parser.add_argument(
    "--pct_change",
    action="store_true",
    help="Apply percentage change transformation to the data.",
)
args = parser.parse_args()

# TODO: remove tickers and have just a single variable, also remove tickers and timefreqs from for loop
method = args.method
exp_name = args.exp_name
tickers = ["MSFT"]
# timefreqs = [args.timefreq]


# RQ 1
def RQ1():
    timefreqs = ["1d"]
    for exp_name in [
        "train-restricted-years",
        "train-less-year-log",
        "train-less-year-linear",
    ]:
        print(exp_name)

        if exp_name == "train-restricted-years":
            horizon_lens = [1, 5, 21, 63]
            # horizon_lens = [1]
            train_last_ns = [int(value) for value in np.linspace(1, 10, 10)]
            days = False
        elif exp_name == "train-less-year-log":
            horizon_lens = [1, 5, 21, 63]
            # horizon_lens = [1]
            log_spaces_values = np.logspace(np.log10(25), np.log10(250), 10)
            train_last_ns = [int(value) for value in log_spaces_values]
            days = True
        elif exp_name == "train-less-year-linear":
            horizon_lens = [1, 5, 21, 63]
            # horizon_lens = [1]
            lineal_spaced_values = np.linspace(25, 250, 10)
            train_last_ns = [int(value) for value in lineal_spaced_values]
            days = True
        else:
            raise NameError("The experiment name is not supported")

        if args.pct_change:
            exp_name = exp_name + "-pct"

        if method not in ["naive", "arima", "fm", "sundial", "chronos_base"]:
            raise NameError("The method is not supported")

        total_exp = 0

        for ticker, timefreq, horizon_len, train_last_n in product(
            tickers, timefreqs, horizon_lens, train_last_ns
        ):
            try:
                run_pipeline(
                    ticker,
                    timefreq,
                    method,
                    horizon_len,
                    train_last_n,
                    exp_name,
                    days_flag=days,
                    pct_change=args.pct_change,
                )
                time.sleep(15)
                total_exp += 1
            except Exception as e:
                print(f"Exception occured {e}")
                exit()
        print(total_exp)


# RQ 2
def RQ2():
    print(args.method)
    timefreqs = ["1h"]
    for exp_name in [
        "train-restricted-years",
        "train-less-year-log",
        "train-less-year-linear",
    ]:
        print(exp_name)

        if exp_name == "train-restricted-years":
            # horizon_lens = [1, 5, 21, 63]
            horizon_lens = [1]
            train_last_ns = [int(value) for value in np.linspace(1, 10, 10)]
            days = False
        elif exp_name == "train-less-year-log":
            # horizon_lens = [1, 5, 21, 63]
            horizon_lens = [1]
            log_spaces_values = np.logspace(np.log10(25), np.log10(250), 10)
            train_last_ns = [int(value) for value in log_spaces_values]
            days = True
        elif exp_name == "train-less-year-linear":
            # horizon_lens = [1, 5, 21, 63]
            horizon_lens = [1]
            lineal_spaced_values = np.linspace(25, 250, 10)
            train_last_ns = [int(value) for value in lineal_spaced_values]
            days = True
        else:
            raise NameError(f"The experiment {exp_name} is not supported")

        if args.pct_change:
            exp_name = exp_name + "-pct"

        if method not in ["naive", "arima", "fm", "sundial", "chronos_base"]:
            raise NameError("The method is not supported")

        total_exp = 0

        for ticker, timefreq, horizon_len, train_last_n in product(
            tickers, timefreqs, horizon_lens, train_last_ns
        ):
            try:
                run_pipeline(
                    ticker,
                    timefreq,
                    method,
                    horizon_len,
                    train_last_n,
                    exp_name,
                    days_flag=days,
                    pct_change=args.pct_change,
                )
                time.sleep(15)
                total_exp += 1
            except Exception as e:
                print(f"Exception occured {e}")
                exit()
        print(total_exp)


# RQ 3
def RQ3():
    print(args.method)
    timefreqs = ["1h"]

    # Split dates with crisis event labels
    crisis_events = {
        "2008-01-20": "financial-crisis-2008",  # Five days before 2008-01-25
        "2022-03-31": "market-downturn-2022",  # Five days before 2022-04-05
        "2020-02-28": "covid-crash-2020",  # Five days before 2020-03-04
    }

    # Test period: 2 months (approximately 60 days)
    test_n_days = 60

    # Horizon length
    horizon_lens = [1]

    if method not in ["naive", "arima", "fm", "sundial", "chronos_base"]:
        raise NameError("The method is not supported")

    train_last_ns = []
    # Iterate over different spacing configurations
    for spacing_type in ["log", "linear"]:
        print(f"Running experiments with {spacing_type} spacing")

        if spacing_type == "log":
            log_spaces_values = np.logspace(np.log10(25), np.log10(250), 10)
            train_last_ns = [int(value) for value in log_spaces_values]
        elif spacing_type == "linear":
            lineal_spaced_values = np.linspace(25, 250, 10)
            train_last_ns = [int(value) for value in lineal_spaced_values]

        total_exp = 0

        for ticker, timefreq, split_date, horizon_len, train_last_n in product(
            tickers, timefreqs, crisis_events.keys(), horizon_lens, train_last_ns
        ):
            # Create unique experiment name for each crisis event and spacing type
            exp_name = f"rq3-{crisis_events[split_date]}-{spacing_type}"

            if args.pct_change:
                exp_name = exp_name + "-pct"

            try:
                run_pipeline(
                    ticker,
                    timefreq,
                    method,
                    horizon_len,
                    train_last_n,
                    exp_name,
                    days_flag=True,  # Use days for training
                    pct_change=args.pct_change,
                    split_date=split_date,
                    test_n_days=test_n_days,
                )
                time.sleep(1)
                total_exp += 1
            except Exception as e:
                print(f"Exception occurred {e}")
                exit()

        print(f"Total experiments run for {spacing_type} spacing: {total_exp}")


RQ3()
