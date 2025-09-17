#!/bin/bash

for exp_name in train-restricted-years train-less-year-log train-less-year-linear; do
  python scripts/experiments.py --method naive --exp_name "$exp_name" 
done
for exp_name in train-restricted-years train-less-year-log train-less-year-linear; do
  python scripts/experiments.py --method arima --exp_name "$exp_name" 
done
