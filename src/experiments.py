import subprocess
from itertools import product
import numpy as np
import sys

def run_pipeline(ticker, timefreq, method, horizon_len, train_last_n, exp_name):
    """Runs the pipeline with the given parameters."""
    command = [
        "python",
        "src/pipeline.py",
        "--ticker",
        ticker,
        "--timefreq",
        timefreq,
        "--method",
        method,
        "--horizon_len",
        str(horizon_len),
        "----train_last_n_days",
        str(train_last_n),
        "--exp_name",
        exp_name,
    ]
    subprocess.run(command)
    # return command


tickers = ["MSFT"]
timefreqs = ["1d"]
method = sys.argv[1]

if method not in ["naive", "arima", "fm"]:
    raise NameError

start_log = np.log10(10)  
stop_log = np.log10(200)  
log_spaced_values = np.logspace(start_log, stop_log, 10)

horizon_lens = [1, 5, 21, 63]
train_last_ns = [int(value) for value in log_spaced_values]
print(train_last_ns)

exp_name = "train-less-year"
# exit()
total_exp = 0

for ticker, timefreq, horizon_len, train_last_n in product(tickers, timefreqs, horizon_lens, train_last_ns):
    run_pipeline(ticker, timefreq, method, horizon_len, train_last_n, exp_name)
    # print(f"{method=}, {horizon_len=}, {train_last_n=}")
    total_exp += 1

        
print(total_exp)