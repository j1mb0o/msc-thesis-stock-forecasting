import matplotlib.pyplot as plt
import sys
import numpy as np
import yaml
import os
from pathlib import Path
from dataclasses import dataclass

try:
    EXPERIMENT_NAME = sys.argv[1]
except:
    raise SyntaxError("Provide experiment name as first argument")

CONFIG_PATH = Path('configs') / EXPERIMENT_NAME
RESULTS_PATH = Path('results') / EXPERIMENT_NAME

print(CONFIG_PATH)
print(RESULTS_PATH)

try:
    assert CONFIG_PATH.exists()
except:
    raise NotADirectoryError(CONFIG_PATH)

try:
    assert RESULTS_PATH.exists()
except:
    raise NotADirectoryError(RESULTS_PATH)


@dataclass
class ConfDataClass:    
    horizon_len: int = 0
    resutls_filepath: str = ''
    training_period_value: int = 0


def plot_varying_horizon(model:str) -> None:
    """Plots every configs based ont the horizon and changes"""

    # get the unique horizon lens
    #TODO: Change this when will add more stocks
    MODEL_PATH = CONFIG_PATH / model / "MSFT"
    
    unique_horizon_dicts = {}

    for conf in os.listdir(MODEL_PATH):
        with open(MODEL_PATH / conf, 'r') as f:
            config = yaml.safe_load(f)
        c = ConfDataClass(horizon_len=config['horizon_length'], 
                          resutls_filepath=config['results_file_path'], 
                          training_period_value=config['training_period_value'])
        unique_horizon_dicts[c.horizon_len] = [c]

    print(unique_horizon_dicts.keys())
    print(unique_horizon_dicts)


def plot_varying_train_size():
    pass

if __name__ == '__main__':
    print(os.listdir(CONFIG_PATH))

    for model in os.listdir(CONFIG_PATH):
        plot_varying_horizon(model)
        exit()
        # plot_varying_train_size()