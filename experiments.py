from itertools import product

SPLIT_DATE = {
    "split": "2023-01-01"
}

METHODS = {
    "naive": "naive",
    "arima": "arima",
    "fm": "fm"
}

HORIZON_LEN = {
    "daily": 1,
    "weekly": 5,
    "monthly": 21,
    "quarterly": 63,
    # "yearly": 252
}

TRAIN_LAST_N = {
    "10_years": 10,
    "8_years": 8,
    "6_years": 6,
    "4_years": 4,
    "2_years": 2,
    "1_years": 1
}


DIFF = {
    "diff": True,
    "no_diff": False
}

# Generate here the necessary code 
import subprocess

def run_pipeline(ticker, timefreq, method, horizon_len, train_last_n, diff):
    """Runs the pipeline with the given parameters."""
    command = [
        "python",
        "pipeline.py",
        "--ticker",
        ticker,
        "--timefreq",
        timefreq,
        "--method",
        method,
        "--horizon_len",
        str(horizon_len),
        "--train_last_n",
        str(train_last_n),
    ]

    if diff:
        command.append("--diff")
    
    subprocess.run(command)
    # return command




# Define the parameters to iterate over
tickers = ["MSFT"]
timefreqs = ["1d"]
# methods = ["naive", "arima", "fm"]
methods = ["naive"]

horizon_lens = [1, 5, 21, 63]
train_last_ns = [1, 2, 4, 6, 8, 10]
diffs = [True, False]

total_exp = 0
# Iterate over all combinations of parameters
for ticker, timefreq, method, horizon_len, diff in product(tickers, timefreqs, methods, horizon_lens, diffs):
    if method == "naive":
        train_last_n = 1
        # print(f"{ticker=}, {timefreq=}, {method=}, {horizon_len=}, {diff=}, {train_last_n=}")
        print(run_pipeline(ticker, timefreq, method, horizon_len, train_last_n, diff))
        total_exp += 1

    else:
        # train_last_n_options = train_last_ns
        for train_last_n in train_last_ns:
            print(run_pipeline(ticker, timefreq, method, horizon_len, train_last_n, diff))
            # print(f"{method=}, {horizon_len=}, {diff=}, {train_last_n=}")
            total_exp += 1

    
            
print(total_exp)

    