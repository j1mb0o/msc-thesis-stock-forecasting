#!/bin/bash

# This script runs RQ3 experiments with and without percentage change transformation.
# The forecasting method can be specified as the first argument to the script.
# Example: ./run_experiments.sh arima
# If no method is specified, it defaults to 'naive'.

# --- Configuration ---
METHOD=${1:-"naive"}

echo "===== Running RQ3 experiments for method: $METHOD ====="

# Run experiments without percentage change
echo "--- Running experiments WITHOUT percentage change ---"
python scripts/experiments.py --method "$METHOD"

# Run experiments with percentage change
echo "--- Running experiments WITH percentage change ---"
python scripts/experiments.py --method "$METHOD" --pct_change

echo "===== All RQ3 experiments for method $METHOD completed. ====="
