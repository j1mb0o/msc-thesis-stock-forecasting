# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a thesis project comparing time-series foundation models (Chronos, Sundial, TimesFM) with traditional methods (ARIMA, Naive) for stock price prediction. The project uses daily stock data to benchmark predictive performance across accuracy, generalizability, and robustness in financial forecasting.

## Environment Setup

**Python Version:** 3.11 (required)

**Environment Management:** Use `uv` for dependency management

```bash
# Create and activate environment
uv venv -p python3.11 .venv
source .venv/bin/activate

# Install base dependencies
pip install -e ".[dev]"

# Install model-specific dependencies
pip install -e ".[chronos]"  # For Chronos (uses transformers>=4.48.0)
pip install -e ".[sundial]"  # For Sundial (uses transformers==4.40.1)
pip install -e ".[times]"    # For TimesFM (Linux only)
```

**Important:** Chronos and Sundial have conflicting transformer versions. Use separate virtual environments or install only the model you're working with.

## Running Experiments

**Using the shell script (recommended for batch experiments):**
```bash
./run_experiments.sh [method]  # method: arima, chronos_base, sundial, naive
# Runs all experiment types with multiple horizons for the specified method
```

**Running single experiments:**
```bash
python scripts/pipeline.py \
  --ticker MSFT \
  --timefreq 1d \
  --method arima \
  --horizon_len 1 \
  --train_last_n_days 250 \
  --exp_name my-experiment

# Or using config files:
python scripts/pipeline.py --config-path configs/MSFT/1h/arima/train-less-year-linear/config.yaml
```

**Running batch experiments programmatically:**
```bash
python scripts/experiments.py --method naive --exp_name train-restricted-years
```

## Code Formatting and Testing

```bash
black scripts/  # Format Python code
pytest          # Run all tests
pytest tests/test_file.py::test_function  # Run single test
```

## Generating Results

```bash
python scripts/gen_plots.py   # Generate figures from experiment results
python scripts/gen_tables.py  # Generate tables from experiment results
```

## Architecture

### Core Pipeline Flow

1. **Data Download** (`utils/download_data.py`): Fetches stock data using yfinance
2. **Data Preparation** (`utils/model_data_prep.py`): Splits data into train/test sets, applies transformations (differencing, percentage change)
3. **Model Training & Forecasting** (`methods/*.py`): Each model implements fit/forecast pattern
4. **Evaluation** (`pipeline.py`): Calculates metrics (MAE, RMSE, MAPE, SMAPE, MDA)
5. **Results Storage**: Saves forecasts as CSV in `results/` and configs as YAML in `configs/`

### Key Scripts

- **`scripts/pipeline.py`**: Main entry point for running single experiments. Orchestrates the full pipeline from data download to evaluation.
- **`scripts/experiments.py`**: Batch experiment runner that generates multiple parameter combinations.
- **`run_experiments.sh`**: Shell script for running systematic experiments across different training periods.

### Model Implementation Pattern

All forecasters follow a consistent interface:
```python
class ModelForecaster:
    def __init__(self, train_data: pd.Series, test_data: pd.Series, horizon_len: int = 1):
        # Initialize with train/test data

    def fit(self):  # For ARIMA only
        # Train the model

    def forecast(self) -> pd.Series:
        # Generate rolling forecasts by:
        # 1. Predict next horizon steps
        # 2. Update context with actual observations
        # 3. Repeat until test set is covered
        return forecasts  # Same length as test_data
```

### Model-Specific Notes

- **ARIMA** (`methods/arima.py`): Uses `pmdarima.auto_arima` for automatic order selection. Requires explicit `fit()` before forecasting.
- **Chronos** (`methods/chronos_forcast.py`): Uses `amazon/chronos-bolt-base` model with PyTorch/MPS backend.
- **Sundial** (`methods/sundial.py`): Pretrained zero-shot forecasting model.
- **Naive** (`methods/naive_forecast.py`): Baseline that predicts the last observed value.

### Experiment Types

The codebase supports three experiment categories (defined in `run_experiments.sh` and `experiments.py`):

1. **train-restricted-years**: Trains on 1-10 years of data
2. **train-less-year-log**: Trains on 25-250 days (log-spaced intervals)
3. **train-less-year-linear**: Trains on 25-250 days (linear intervals)

Each can have `-pct` suffix to apply percentage change transformation to data.

### Configuration System

Experiment configs are automatically generated and stored as YAML files with naming pattern:
```
{timestamp}_{ticker}_{timefreq}_{method}_split_{split_date}_train_{period}_test_{test_years}y_horizon_{horizon}.yaml
```

Configs contain all experiment parameters and evaluation metrics for reproducibility.

### Data Transformations

The `prepare_data_for_modeling()` function in `utils/model_data_prep.py` supports:
- **Differencing** (`--diff` flag): First-order differencing for stationarity
- **Percentage Change** (`--pct_change` flag): Converts to percentage returns

These transformations are applied before train/test split.

### Metrics

All experiments calculate:
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error (reported as %)
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **MDA**: Mean Directional Accuracy (measures direction prediction correctness)

### Directory Structure

```
data/           # Raw stock data (downloaded via yfinance)
results/        # Experiment results CSVs
configs/        # Experiment configuration YAMLs
scripts/        # All Python scripts
  methods/      # Model implementations
  utils/        # Data utilities
figures/        # Generated plots
tables/         # Generated tables
notebooks/      # Jupyter notebooks for EDA
```

## Important Development Notes

- **Command-line arguments**: Use `utils/argfile.py` for argument parsing. The codebase supports both direct parameters and config file input.
- **Logging**: All scripts use `logging.basicConfig(level=logging.INFO, format='...')` consistently.
- **Error Handling**: Models validate inputs with `TypeError`/`ValueError` and use try/except for optional imports.
- **Type Hints**: Functions use type hints (e.g., `ticker: str = "MSFT"`).
- **Naming**: snake_case for variables/functions, PascalCase for classes.
- **Results are immutable**: Once experiments run, their configs and results are saved with timestamps. Don't modify existing result files.
- **Horizons**: Multi-step forecasts update the model with actual observations after each horizon window (rolling forecast approach).
