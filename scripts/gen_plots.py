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


def plot_varying_horizon(model:str) -> None:
    """Plots forecasts for different training data sizes, grouped by horizon, on a 2x2 grid."""

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

    if not unique_horizon_dicts:
        print(f"No configurations found for model {model} under {MODEL_PATH}")
        return

    print(f"Found horizons: {sorted(list(unique_horizon_dicts.keys()))}")

    # Prepare style map for unique training periods
    all_configs = [item for sublist in unique_horizon_dicts.values() for item in sublist]
    unique_training_periods = sorted(list(set(c.training_period_value for c in all_configs)))

    style_map = {}
    if unique_training_periods:
        # Use tab10 colors, cycle if more than 10 TPs
        cmap = plt.colormaps.get_cmap('tab10')
        colors_palette = cmap.colors
        # linestyles_palette = ['--', 'solid', '-.', ':']
        linestyles_palette = ["solid"]
        
        style_map = {
            tp: (colors_palette[i % len(colors_palette)], 
                 linestyles_palette[i % len(linestyles_palette)])
            for i, tp in enumerate(unique_training_periods)
        }

    legend_handles_dict = {} # To store unique legend items (label: handle)

    plotted_on_fig_count = 0
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # Set the main title for the entire figure
    fig.suptitle(f"MSFT {model.upper()} Model: Forecasts vs. Actuals for Various Horizons & Training Durations", fontsize=16)
    axs = axs.flatten()
    max_subplots = len(axs)

    for horizon_len, configs_for_this_horizon in sorted(unique_horizon_dicts.items()):
        if plotted_on_fig_count >= max_subplots:
            print(f"Warning: More unique horizons ({len(unique_horizon_dicts)}) than available subplots ({max_subplots}). Plotting only the first {max_subplots}.")
            break

        if not configs_for_this_horizon:
            print(f"No configurations loaded for horizon: {horizon_len}")
            continue

        configs_for_this_horizon.sort(key=lambda x: x.training_period_value)

        PLOT_GROUND_TRUTH_ON_THIS_AX = True 

        current_ax = axs[plotted_on_fig_count]
        # Set title for the current subplot (once per horizon)
        current_ax.set_title(f"Forecast Horizon: {horizon_len} days", fontsize=10)

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

            # Plot Ground Truth (Actuals) - once per subplot
            if PLOT_GROUND_TRUTH_ON_THIS_AX and 'Actual' in results_df.columns:
                gt_label = "Actual Values"
                line, = current_ax.plot(x_values, results_df['Actual'], color='black', linestyle='-', linewidth=1.5, label=gt_label)
                if gt_label not in legend_handles_dict:
                    legend_handles_dict[gt_label] = line
                PLOT_GROUND_TRUTH_ON_THIS_AX = False 

            # Plot Forecast
            # The column name in the CSV is {method}_Forecast as per pipeline.py
            forecast_col_name = f"{model}_Forecast" 

            if forecast_col_name in results_df.columns:
                tp_value = conf_data.training_period_value
                color, style = style_map.get(tp_value, ('blue', '-')) # Default style if tp_value somehow not in map
                fc_label = f"Train Days: {tp_value}"
                
                line, = current_ax.plot(x_values, results_df[forecast_col_name], label=fc_label, color=color, linestyle=style)
                if fc_label not in legend_handles_dict:
                    legend_handles_dict[fc_label] = line
            else:
                print(f"    Warning: Forecast column '{forecast_col_name}' not found in {csv_file_path} for model '{model}'")

            current_ax.set_xlabel(x_label, fontsize=9)
            current_ax.set_ylabel("Value", fontsize=9)
            current_ax.tick_params(axis='x', rotation=30, labelsize=8)
            current_ax.tick_params(axis='y', labelsize=8)
            current_ax.grid(True, linestyle=':', alpha=0.6)

        plotted_on_fig_count+=1

    # Add a single legend for the entire figure
    if legend_handles_dict:
        fig.legend(legend_handles_dict.values(), legend_handles_dict.keys(), loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=min(5, len(legend_handles_dict)), fontsize=9)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust rect to make space for suptitle and legend
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
    # print(f"Found train lens: {sorted(list(unique_train_len.keys()))}")

    for horizon_len, configs_for_this_horizon in sorted(unique_train_len.items()):
        if not configs_for_this_horizon:
            print(f"No configurations loaded for horizon: {horizon_len}")
            continue

        # print(f"  Plotting for Horizon: {horizon_len} days, Number of configs: {len(configs_for_this_horizon)}")

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