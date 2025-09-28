#!/bin/bash

# This script runs a series of forecasting experiments for the 'sundial' method,
# calling the main pipeline script directly with different parameter combinations.

# --- Configuration ---
TICKER="MSFT"
TIMEFREQ="1d"
METHOD="naive"
HORIZON_LEN=1

# For 1d the horizons are 1, 5, 21 and 63

# --- Experiment Loop ---
for exp_name in "train-restricted-years" "train-less-year-log" "train-less-year-linear"; do
  echo "===== Running Experiment: $exp_name for method $METHOD ====="

  # Determine training parameters based on experiment name
  if [ "$exp_name" == "train-restricted-years" ]; then
    TRAIN_LAST_NS=$(seq 1 10)
    DAYS_FLAG="--train_last_n_years"
  elif [ "$exp_name" == "train-less-year-log" ]; then
    TRAIN_LAST_NS=$(python -c "import numpy as np; print(' '.join([str(int(v)) for v in np.logspace(np.log10(25), np.log10(250), 10)]))")
    DAYS_FLAG="--train_last_n_days"
  elif [ "$exp_name" == "train-less-year-linear" ]; then
    TRAIN_LAST_NS=$(python -c "import numpy as np; print(' '.join([str(int(v)) for v in np.linspace(25, 250, 10)]))")
    DAYS_FLAG="--train_last_n_days"
  else
    echo "Unknown experiment name: $exp_name"
    continue
  fi

  # --- Parameter Combination Loop ---
  for train_last_n in $TRAIN_LAST_NS; do
    echo "--- Running: Ticker=$TICKER, Timefreq=$TIMEFREQ, Horizon=$HORIZON_LEN, Train=$train_last_n ---"
    
    python scripts/pipeline.py \
      --ticker "$TICKER" \
      --timefreq "$TIMEFREQ" \
      --method "$METHOD" \
      --horizon_len "$HORIZON_LEN" \
      "$DAYS_FLAG" "$train_last_n" \
      --exp_name "$exp_name"
    
  done
done

echo "===== All experiments for method $METHOD completed. =====
