import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import numpy as np
import yaml
import os
from pathlib import Path
import pandas as pd
from dataclasses import dataclass

try:
    EXPERIMENT_NAME = sys.argv[1]
except IndexError: # Changed from generic except to IndexError
    raise SyntaxError("Provide experiment name as first argument (e.g., python gen_plots.py first-exp)")

CONFIG_PATH = Path()
RESULTS_PATH = Path()
@dataclass
class ConfDataClass:    
    horizon_len: int = 0
    training_period_value: int = 0
    mse:  int = 0
    mae:  int = 0
    rmse: int = 0
    mape: int = 0


def setup_paths(exp_name: str):
    """Sets up global CONFIG_PATH and RESULTS_PATH."""
    global EXPERIMENT_NAME, CONFIG_PATH, RESULTS_PATH, TABLES_PATH
    EXPERIMENT_NAME = exp_name
    CONFIG_PATH = Path('configs') / EXPERIMENT_NAME
    RESULTS_PATH = Path('results') / EXPERIMENT_NAME
    TABLES_PATH = Path('tables') / EXPERIMENT_NAME


    print(f"Experiment Name: {EXPERIMENT_NAME}")
    print(f"Config Path: {CONFIG_PATH}")
    print(f"Results Path: {RESULTS_PATH}")

    if not CONFIG_PATH.exists():
        # This is a critical check, as no configs means nothing to plot.
        raise NotADirectoryError(f"CRITICAL: CONFIG_PATH does not exist: {CONFIG_PATH}")


def varying_horizon(model:str, stock:str) -> None:
    MODEL_PATH = CONFIG_PATH / model / stock
    
    unique_horizon_dicts = {}

    for conf_filename in os.listdir(MODEL_PATH):
        with open(MODEL_PATH / conf_filename, 'r') as f:
            config = yaml.safe_load(f)

        c = ConfDataClass(
            horizon_len=config['horizon_length'],
            training_period_value=config['training_period_value'],
            mse=config['evaluation_metrics']['mse'],
            mae=config['evaluation_metrics']['mae'],
            rmse=config['evaluation_metrics']['rmse'],
            mape=config['evaluation_metrics']['mape']
        )
        
        if c.horizon_len not in unique_horizon_dicts:
            unique_horizon_dicts[c.horizon_len] = []
        
        # here we create a dictionary with the horizon as key
        # and the config as value
        unique_horizon_dicts[c.horizon_len].append(c)

    
    if not unique_horizon_dicts:
        print(f"No configurations found for model {model} under {MODEL_PATH}")
        return


    for idx, (horizon_len, configs_for_this_horizon) in enumerate(sorted(unique_horizon_dicts.items())):
        if not configs_for_this_horizon:
            print(f"No configurations loaded for horizon: {horizon_len}")
            continue

        configs_for_this_horizon.sort(key=lambda x: x.training_period_value)

        # Create a DataFrame for the current horizon
        df_data = []
        for conf_data in configs_for_this_horizon:
            df_data.append({
                'Training Period (Days)': conf_data.training_period_value,
                'MSE': conf_data.mse,
                'MAE': conf_data.mae,
                'RMSE': conf_data.rmse,
                'MAPE (%)': conf_data.mape
            })
        
        df = pd.DataFrame(df_data)
        df = df.set_index('Training Period (Days)')
        
        # Save the DataFrame to a CSV file
        output_dir = TABLES_PATH / model / stock
        os.makedirs(output_dir, exist_ok=True)
        
        table_filename = output_dir / f"{model}_{stock}_horizon_{horizon_len}_metrics.csv"
        # df.to_csv(table_filename)
        # Save the DataFrame to a LaTeX file
        latex_filename = output_dir / f"{model}_{stock}_horizon_{horizon_len}_metrics.tex"
        df.to_latex(latex_filename, caption=f"Evaluation Metrics for {model.upper()} on {stock} with Horizon {horizon_len} {'days' if horizon_len > 1 else 'day'}", label=f"tab:{model}_{stock}_h{horizon_len}_metrics")
        
        print(f"Table for Horizon {horizon_len} saved to {table_filename}")

    # os.makedirs(TABLES_PATH / model / stock, exist_ok=True)
    # plt.savefig(TABLES_PATH / model / stock / f"{model}_{stock}_varying_horizon_tight.png", format='png', dpi=300, bbox_inches='tight')



if __name__ == '__main__':
    print(f"--- Starting Plot Generation for Experiment: {EXPERIMENT_NAME} ---")
    setup_paths(EXPERIMENT_NAME)
    # varying_horizon('fm', 'MSFT')
    for model in os.listdir(CONFIG_PATH):
        varying_horizon(model, 'MSFT')
        
    