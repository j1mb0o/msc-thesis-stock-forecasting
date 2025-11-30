# Thesis Project: When Old Meets New: A Comparative Study of Traditional and Foundation Models for Stock Prediction

## Project Description
This thesis explores stock price prediction by comparing time-series foundation models (Chronos, Sundial, TimesFM) with traditional methods (ARIMA, Naive) for stock price prediction. Using daily stock data, the project benchmarks predictive performance across accuracy, generalizability, and robustness within financial forecasting.

## Setup Instructions

### Prerequisites
- Python 3.11 (required)
- `uv` for dependency management

### Environment Setup

```bash
# Create and activate virtual environment
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

## Project Structure

```
data/           # Raw stock data (downloaded via yfinance)
results/        # Experiment results CSVs
configs/        # Experiment configuration YAMLs
scripts/        # All Python scripts
  methods/      # Model implementations (arima.py, chronos_forcast.py, sundial.py, naive_forecast.py)
  utils/        # Data utilities (download_data.py, model_data_prep.py, argfile.py)
  pipeline.py   # Main entry point for single experiments
  experiments.py # Batch experiment runner
  gen_plots.py  # Generate figures from results
  gen_tables.py # Generate tables from results
figures/        # Generated plots
tables/         # Generated tables
notebooks/      # Jupyter notebooks for EDA
```

## Running Experiments

### Using the Shell Script (Recommended for Batch Experiments)

```bash
./run_experiments.sh [method]  # method: arima, chronos_base, sundial, naive
# Runs all experiment types with multiple horizons for the specified method
```

### Running Single Experiments

```bash
# Using command-line arguments
python scripts/pipeline.py \
  --ticker MSFT \
  --timefreq 1d \
  --method arima \
  --horizon_len 1 \
  --train_last_n_days 250 \
  --exp_name my-experiment

# Using config files
python scripts/pipeline.py --config-path configs/MSFT/1h/arima/train-less-year-linear/config.yaml
```

### Running Batch Experiments Programmatically

```bash
python scripts/experiments.py --method naive --exp_name train-restricted-years
```

### Experiment Types

The codebase supports three experiment categories:

1. **train-restricted-years**: Trains on 1-10 years of data
2. **train-less-year-log**: Trains on 25-250 days (log-spaced intervals)
3. **train-less-year-linear**: Trains on 25-250 days (linear intervals)

Each can have `-pct` suffix to apply percentage change transformation to data.

## Models

### Implemented Models

- **ARIMA** (`methods/arima.py`): Uses `pmdarima.auto_arima` for automatic order selection. Classical statistical model requiring explicit `fit()` before forecasting.
- **Chronos** (`methods/chronos_forcast.py`): Uses `amazon/chronos-bolt-base` pretrained foundation model with PyTorch/MPS backend.
- **Sundial** (`methods/sundial.py`): Pretrained time series model for zero-shot forecasting.
- **TimesFM** (`methods/times.py`): Google's time series foundation model (Linux only).
- **Naive** (`methods/naive_forecast.py`): Baseline that predicts the last observed value.

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

## Core Pipeline Flow

1. **Data Download** (`utils/download_data.py`): Fetches stock data using yfinance
2. **Data Preparation** (`utils/model_data_prep.py`): Splits data into train/test sets, applies transformations
3. **Model Training & Forecasting** (`methods/*.py`): Each model implements fit/forecast pattern
4. **Evaluation** (`pipeline.py`): Calculates metrics
5. **Results Storage**: Saves forecasts as CSV in `results/` and configs as YAML in `configs/`

### Data Transformations

The `prepare_data_for_modeling()` function supports:
- **Differencing** (`--diff` flag): First-order differencing for stationarity
- **Percentage Change** (`--pct_change` flag): Converts to percentage returns

### Evaluation Metrics

All experiments calculate:
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error (reported as %)
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **MDA**: Mean Directional Accuracy (measures direction prediction correctness)

## Generating Results

### Generate Plots

```bash
python scripts/gen_plots.py   # Generate figures from experiment results
```

### Generate Tables

```bash
python scripts/gen_tables.py  # Generate tables from experiment results
```

### Generate Comparison Tables (Fast)

```bash
python scripts/gen_comparison_tables.py  # Generate comparison tables efficiently
```

## Code Formatting and Testing

```bash
# Format Python code
black scripts/

# Run all tests
pytest

# Run specific test
pytest tests/test_file.py::test_function
```

## Configuration System

Experiment configs are automatically generated and stored as YAML files with naming pattern:

```
{timestamp}_{ticker}_{timefreq}_{method}_split_{split_date}_train_{period}_test_{test_years}y_horizon_{horizon}.yaml
```

Configs contain all experiment parameters and evaluation metrics for reproducibility.

## Key Development Notes

- **Results are immutable**: Once experiments run, their configs and results are saved with timestamps. Don't modify existing result files.
- **Horizons**: Multi-step forecasts update the model with actual observations after each horizon window (rolling forecast approach).
- **Logging**: All scripts use `logging.basicConfig(level=logging.INFO)` consistently.
- **Type Hints**: Functions use type hints for better code clarity.
- **Naming**: snake_case for variables/functions, PascalCase for classes.

## Additional Resources

For more detailed information about the project architecture, model implementations, and development guidelines, see [CLAUDE.md](CLAUDE.md).
