import argparse
import subprocess
import time
from itertools import product

import numpy as np


SUPPORTED_METHODS = ["naive", "arima", "fm", "sundial", "chronos_base"]
TICKERS = ["MSFT"]


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
        "--ticker", ticker,
        "--timefreq", timefreq,
        "--method", method,
        "--horizon_len", str(horizon_len),
        "--train_last_n_days" if days_flag else "--train_last_n_years",
        str(train_last_n),
        "--exp_name", exp_name,
    ]
    if pct_change:
        command.append("--pct_change")
    if split_date:
        command.extend(["--split_date", split_date])
    if test_n_days:
        command.extend(["--test_n_days", str(test_n_days)])
    subprocess.run(command)


def _train_sizes_for(experiment_name):
    """Return ``(train_last_ns, days_flag)`` for the standard RQ1/RQ2 experiments."""
    if experiment_name == "train-restricted-years":
        return [int(v) for v in np.linspace(1, 10, 10)], False
    if experiment_name == "train-less-year-log":
        return [int(v) for v in np.logspace(np.log10(25), np.log10(250), 10)], True
    if experiment_name == "train-less-year-linear":
        return [int(v) for v in np.linspace(25, 250, 10)], True
    raise NameError(f"The experiment {experiment_name} is not supported")


def _run_grid(*, method, timefreqs, horizon_lens, pct_change, sleep_seconds):
    """Run RQ1/RQ2-style sweep across the three standard training-size schedules."""
    if method not in SUPPORTED_METHODS:
        raise NameError("The method is not supported")

    for base_exp_name in [
        "train-restricted-years",
        "train-less-year-log",
        "train-less-year-linear",
    ]:
        print(base_exp_name)
        train_last_ns, days_flag = _train_sizes_for(base_exp_name)
        exp_name = f"{base_exp_name}-pct" if pct_change else base_exp_name

        total_exp = 0
        for ticker, timefreq, horizon_len, train_last_n in product(
            TICKERS, timefreqs, horizon_lens, train_last_ns
        ):
            try:
                run_pipeline(
                    ticker,
                    timefreq,
                    method,
                    horizon_len,
                    train_last_n,
                    exp_name,
                    days_flag=days_flag,
                    pct_change=pct_change,
                )
                time.sleep(sleep_seconds)
                total_exp += 1
            except Exception as e:
                print(f"Exception occured {e}")
                exit()
        print(total_exp)


def RQ1(method, pct_change):
    _run_grid(
        method=method,
        timefreqs=["1d"],
        horizon_lens=[1, 5, 21, 63],
        pct_change=pct_change,
        sleep_seconds=15,
    )


def RQ2(method, pct_change):
    print(method)
    _run_grid(
        method=method,
        timefreqs=["1h"],
        horizon_lens=[1],
        pct_change=pct_change,
        sleep_seconds=15,
    )


def RQ3(method, pct_change):
    """Crisis-period sweep: run each (event, spacing) combination."""
    print(method)
    if method not in SUPPORTED_METHODS:
        raise NameError("The method is not supported")

    timefreqs = ["1h"]
    test_n_days = 60
    horizon_lens = [1]

    # Each split date is 5 trading days before the labelled crisis onset.
    crisis_events = {
        "2008-01-20": "financial-crisis-2008",
        "2022-03-31": "market-downturn-2022",
        "2020-02-28": "covid-crash-2020",
    }

    spacing_train_sizes = {
        "log": [int(v) for v in np.logspace(np.log10(25), np.log10(250), 10)],
        "linear": [int(v) for v in np.linspace(25, 250, 10)],
    }

    for spacing_type, train_last_ns in spacing_train_sizes.items():
        print(f"Running experiments with {spacing_type} spacing")
        total_exp = 0

        for ticker, timefreq, split_date, horizon_len, train_last_n in product(
            TICKERS, timefreqs, crisis_events.keys(), horizon_lens, train_last_ns
        ):
            exp_name = f"rq3-{crisis_events[split_date]}-{spacing_type}"
            if pct_change:
                exp_name += "-pct"

            try:
                run_pipeline(
                    ticker,
                    timefreq,
                    method,
                    horizon_len,
                    train_last_n,
                    exp_name,
                    days_flag=True,
                    pct_change=pct_change,
                    split_date=split_date,
                    test_n_days=test_n_days,
                )
                time.sleep(1)
                total_exp += 1
            except Exception as e:
                print(f"Exception occurred {e}")
                exit()

        print(f"Total experiments run for {spacing_type} spacing: {total_exp}")


def _parse_args():
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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    RQ3(args.method, args.pct_change)
