#!/bin/bash

for exp_name in train-restricted-years train-less-year-log train-less-year-linear; do
  python scripts/experiments.py --method sundial --exp_name "$exp_name" 
done
