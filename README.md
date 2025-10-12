# Thesis Project: When Old Meets New: A Comparative Study of Traditional and Foundation Models for Stock Prediction

## Project Description
This thesis explores stock price prediction by comparing time-series foundation models with traditional time-series machine learning methods. Using daily stock data the project benchmarks predictive performance across both approaches. The goal is to evaluate whether modern foundation models offer a meaningful advantage over established techniques in accuracy, generalizability, and robustness within financial forecasting.

## Setup Instructions

### Prerequisites
- Python 3.11 or higher
- Poetry or UV installed on your system

### Activating the Environment
To activate the environment, run the following commands in your terminal:

```bash
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### Installing Dependencies
This project uses `pyproject.toml` to manage dependencies with different groups for different models. You can install the base dependencies and the dependencies for a specific model group.

**Install base dependencies:**
```bash
uv pip install {target config}
```

**Install dependencies for a specific model:**
```bash
# For Sundial
pip install -e ".[sundial]"

# For Chronos
pip install -e ".[chronos]"
```

## Project Structure
- `data/`: Contains the raw stock data.
- `configs/`: Configuration files for the different models and experiments.
- `scripts/`: Main scripts for running experiments, generating plots, and data preparation.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and results visualization.
- `results/`: Stores the results of the experiments.
- `figures/`: Contains the plots generated from the results.
- `tables/`: Contains the tables generated from the results.

## Running Experiments
You can run the experiments using the `run_experiments.sh` script or by directly running the `pipeline.py` or `experiments.py` scripts.

**Using the shell script:**
```bash
./run_experiments.sh
```

**Using the Python scripts:**
```bash
python scripts/pipeline.py --config-path ../configs/MSFT/1h/arima/train-less-year-linear/config.yaml
python scripts/experiments.py --config-path ../configs/MSFT/1h/arima/train-less-year-linear/config.yaml
```

## Models
The following models are compared in this project:
- **ARIMA:** A classical statistical model for time series forecasting.
- **Chronos:** A family of pretrained time series forecasting models from Amazon.
- **Sundial:** A pretrained time series model for zero-shot forecasting.
- **Naive:** A simple baseline model that predicts the last observed value.

## Generating Plots and Tables
To generate plots and tables from the experiment results, you can use the `gen_plots.py` and `gen_tables.py` scripts.

```bash
python scripts/gen_plots.py
python scripts/gen_tables.py
```
