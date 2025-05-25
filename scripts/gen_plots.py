import matplotlib.pyplot as plt
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
    results_filepath: str = ''
    training_period_value: int = 0

def setup_paths(exp_name: str):
    """Sets up global CONFIG_PATH and RESULTS_PATH."""
    global EXPERIMENT_NAME, CONFIG_PATH, RESULTS_PATH
    EXPERIMENT_NAME = exp_name
    CONFIG_PATH = Path('configs') / EXPERIMENT_NAME
    RESULTS_PATH = Path('results') / EXPERIMENT_NAME

    print(f"Experiment Name: {EXPERIMENT_NAME}")
    print(f"Config Path: {CONFIG_PATH}")
    print(f"Results Path: {RESULTS_PATH}")

    if not CONFIG_PATH.exists():
        # This is a critical check, as no configs means nothing to plot.
        raise NotADirectoryError(f"CRITICAL: CONFIG_PATH does not exist: {CONFIG_PATH}")

# Pause for now
def plot_varying_horizon(model:str) -> None:
    """Plots every configs based ont the horizon and changes"""

    # get the unique horizon lens
    #TODO: Change this when will add more stocks
    MODEL_PATH = CONFIG_PATH / model / "MSFT"
    
    unique_horizon_dicts = {}

    for conf_filename in os.listdir(MODEL_PATH):
        with open(MODEL_PATH / conf_filename, 'r') as f:
            config = yaml.safe_load(f)
        c = ConfDataClass(horizon_len=config['horizon_length'],
                        results_filepath=config['results_file_path'],
                        training_period_value=config['training_period_value'])
        
        if c.horizon_len not in unique_horizon_dicts:
            unique_horizon_dicts[c.horizon_len] = []
        unique_horizon_dicts[c.horizon_len].append(c)

    # now we have our configs
    print(f"Found horizons: {sorted(list(unique_horizon_dicts.keys()))}")

    plotted_on_fig_count = 0
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # Set the main title for the entire figure
    fig.suptitle(f"MSFT {model.upper()} Model: Forecasts vs. Training Data for Various Horizons", fontsize=16)
    axs = axs.flatten()

    for horizon_len, configs_for_this_horizon in sorted(unique_horizon_dicts.items()):
        if not configs_for_this_horizon:
            print(f"No configurations loaded for horizon: {horizon_len}")
            continue

        print(f"  Plotting for Horizon: {horizon_len} days, Number of configs: {len(configs_for_this_horizon)}")

        configs_for_this_horizon.sort(key=lambda x: x.training_period_value)

        print(configs_for_this_horizon)
        # continue
        PLOT_GROUND_TRUTH = True # Corrected variable name

        current_ax = axs[plotted_on_fig_count]
        # Set title for the current subplot (once per horizon)
        current_ax.set_title(f"Forecast Horizon: {horizon_len} days", fontsize=10)
        legend_items_for_ax = [] # To keep track of legend items for the current axis

        for i, conf_data in enumerate(configs_for_this_horizon):
            csv_file_path = Path(conf_data.results_filepath) 

            if not csv_file_path.exists():
                print(f"    Warning: CSV file not found at {csv_file_path}")
                # Optionally, add a text indication on the plot if a file is missing for a specific line
                continue
            
            results_df = pd.read_csv(csv_file_path)
            
            if 'Date' in results_df.columns:
                x_values = pd.to_datetime(results_df['Date'])
                x_label = 'Date'
            elif 'ds' in results_df.columns:
                x_values = pd.to_datetime(results_df['ds'])
                x_label = 'Date (ds)'
            else:
                x_values = results_df.index
                x_label = 'Index'

            # The subplot title is now set outside this inner loop.
            # plot_title = f"Forecast Horizon: {conf_data.horizon_len} days" 

            # Determine the forecast column name dynamically based on the model if possible,
            # or ensure it's consistent from pipeline.py (e.g., f"{model}_Forecast")
            forecast_col_name = f"{model}_Forecast" # Assuming 'arima_Forecast', 'naive_Forecast', etc.
            if model == 'fm': # TimesFM might have a different default name from pipeline.py
                forecast_col_name = "TimesFM Forecast" # Match the name set in pipeline.py
            
            if 'arima_Forecast' in results_df.columns:
                current_ax.plot(x_values, results_df['arima_Forecast'], label=f"Train Days: {conf_data.training_period_value}")
                if "Preds" not in legend_items_for_ax: legend_items_for_ax.append("Preds")
            
            if PLOT_GROUND_TRUTH and 'Actual' in results_df.columns:
                current_ax.plot(x_values, results_df['Actual'], label=f"Ground Truth Values")
                if "Actual Values" not in legend_items_for_ax: legend_items_for_ax.append("Actual Values")
                PLOT_GROUND_TRUTH = False # Plot ground truth only once per subplot
            
            current_ax.set_xlabel(x_label, fontsize=9)
            current_ax.set_ylabel("Value", fontsize=9)
            current_ax.tick_params(axis='x', rotation=30, labelsize=8)
            current_ax.tick_params(axis='y', labelsize=8)
            current_ax.grid(True, linestyle=':', alpha=0.6)
        if legend_items_for_ax or configs_for_this_horizon: # Add legend if there are items or if any config was processed
            current_ax.legend(fontsize=8)
        plotted_on_fig_count+=1
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle and x-axis labels
    plt.show()



def plot_varying_train_size(model:str) -> None:
    """Plots every configs based on the train size used"""
    # get the unique horizon lens
    #TODO: Change this when will add more stocks
    MODEL_PATH = CONFIG_PATH / model / "MSFT"
    
    unique_train_len = {}
    
    for conf_filename in os.listdir(MODEL_PATH):
        with open(MODEL_PATH / conf_filename, 'r') as f:
            config = yaml.safe_load(f)

        c = ConfDataClass(horizon_len=config['horizon_length'],
                        results_filepath=config['results_file_path'],
                        training_period_value=config['training_period_value'])
        
        if c.horizon_len not in unique_train_len:
            unique_train_len[c.training_period_value] = []
        unique_train_len[c.training_period_value].append(c)

    # now we have our configs
    print(f"Found train lens: {sorted(list(unique_train_len.keys()))}")

    for horizon_len, configs_for_this_horizon in sorted(unique_train_len.items()):
        if not configs_for_this_horizon:
            print(f"No configurations loaded for horizon: {horizon_len}")
            continue

        print(f"  Plotting for Horizon: {horizon_len} days, Number of configs: {len(configs_for_this_horizon)}")

        configs_for_this_horizon.sort(key=lambda x: x.horizon_len)


if __name__ == '__main__':
    print(f"--- Starting Plot Generation for Experiment: {EXPERIMENT_NAME} ---")

    setup_paths(EXPERIMENT_NAME)

    for model in os.listdir(CONFIG_PATH):
        # print(model)
        plot_varying_horizon(model)
        # plot_varying_train_size(model)
        # TODO: RM for every model to run
        exit()