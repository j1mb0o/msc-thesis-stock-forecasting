import subprocess
import time
from itertools import product
import numpy as np
import argparse


def run_pipeline(ticker, timefreq, method, horizon_len, train_last_n, exp_name, days_flag=False):
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
    subprocess.run(command)
    # return command

# def experiment_years_training():
#     tickers = ["MSFT"]
#     timefreqs = ["1d"]
#     horizon_lens = [1, 5, 21, 63]
#     train_last_ns = [1, 2, 4, 6, 8, 10]
#     exp_name = "train-resticted-years"
#     days = False
    
#     return horizon_lens, train_last_ns, exp_name, tickers, timefreqs, days


# def experiment_les_year_log():

#     tickers = ["MSFT"]
#     timefreqs = ["1d"]
#     start_log = np.log10(25)  
#     stop_log = np.log10(250)  
#     log_spaced_values = np.logspace(start_log, stop_log, 10)
#     horizon_lens = [1, 5, 21, 63]
#     train_last_ns = [int(value) for value in log_spaced_values]
#     exp_name = "train-less-year"
#     days = True
#     return horizon_lens, train_last_ns, exp_name, tickers, timefreqs, days

parser = argparse.ArgumentParser(description="Run experiments for time series forecasting.")
parser.add_argument("--method", type=str, required=True, help="The forecasting method to use.")
parser.add_argument("--exp_name", type=str, required=True, help="The name of the experiment to run.")
parser.add_argument("--timefreq", type=str, required=True, help="The time frequency of the data.")
args = parser.parse_args()

method = args.method
exp_name = args.exp_name
tickers = ["MSFT"]
timefreqs = ["1d"]

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
    raise NameError("The experiment name is not supported")


if method not in ["naive", "arima", "fm", "sundial", "chronos_base"]:
    raise NameError("The method is not supported")

total_exp = 0

for ticker, timefreq, horizon_len, train_last_n in product(tickers, timefreqs, horizon_lens, train_last_ns):
    try:
        run_pipeline(ticker, timefreq, method, horizon_len, train_last_n, exp_name, days_flag=days)
        time.sleep(15)
        total_exp += 1
    except Exception as e:
        print("Exception occured")
        exit()
print(total_exp)
