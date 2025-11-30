#!/bin/bash

# This script runs RQ3 experiments for all methods, automatically switching
# between virtual environments as needed due to conflicting dependencies.
#
# Usage: ./run_all_experiments.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "Running RQ3 experiments for all methods with environment switching"
echo "======================================================================"

# --- Run Naive and ARIMA (base environment) ---
echo ""
echo ">>> Activating BASE environment for Naive and ARIMA..."
source base_env/bin/activate

echo ""
echo ">>> Running NAIVE experiments..."
./run_experiments.sh naive
# echo "TEST TST"

echo ""
echo ">>> Running ARIMA experiments..."
./run_experiments.sh arima
# echo "TEST TST"


deactivate
echo ">>> Deactivated BASE environment"

# --- Run Sundial (sundial environment) ---
echo ""
echo ">>> Activating SUNDIAL environment..."
source env_sundial/bin/activate

echo ""
echo ">>> Running SUNDIAL experiments..."
./run_experiments.sh sundial
# echo "TEST TST"
# sleep 10

deactivate
echo ">>> Deactivated SUNDIAL environment"

# --- Run Chronos (chronos environment) ---
echo ""
echo ">>> Activating CHRONOS environment..."
source chronos_env/bin/activate

echo ""
echo ">>> Running CHRONOS experiments..."
./run_experiments.sh chronos_base
# echo "TEST TST"

deactivate
echo ">>> Deactivated CHRONOS environment"

# --- Completion ---
echo ""
echo "======================================================================"
echo "All RQ3 experiments completed successfully!"
echo "======================================================================"
