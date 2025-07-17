# Thesis Project: When Old Meets New: A Comparative Study of Traditional and Foundation Models for Stock Prediction

## Project Description
This thesis explores stock price prediction by comparing time-series foundation models with traditional time-series machine learning methods. Using daily stock data the project benchmarks predictive performance across both approaches. The goal is to evaluate whether modern foundation models offer a meaningful advantage over established techniques in accuracy, generalizability, and robustness within financial forecasting.

## Setup Instructions

### Prerequisites
- Python 3.11 or higher
- Poetry installed on your system

### Activating the Environment
To activate the Poetry environment, run the following commands in your terminal:

```bash
pyenv install 3.11
eval $(poetry env activate)
poetry install
```

## Running Scripts
Most important scripts: `pipeline.py`, `experiments.py`, `gen_plots.py`
```bash
python <script_name>.py
```

## Measuring Time

```bash
(time python {SCIPT_NAME}) &> {NAME_OF_OUTPUT_FILE}
```
