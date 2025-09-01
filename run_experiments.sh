#!/bin/bash

for exp_name in train-restricted-years train-less-year-log train-less-year-linear; do
  python scripts/experiments.py --method chronos_base --exp_name "$exp_name" --timefreq 1h
done
