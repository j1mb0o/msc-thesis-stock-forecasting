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
    global EXPERIMENT_NAME, CONFIG_PATH, RESULTS_PATH, FIGURES_PATH
    EXPERIMENT_NAME = exp_name
    CONFIG_PATH = Path('configs') / EXPERIMENT_NAME
    RESULTS_PATH = Path('results') / EXPERIMENT_NAME
    FIGURES_PATH = Path('figures') / EXPERIMENT_NAME


    print(f"Experiment Name: {EXPERIMENT_NAME}")
    print(f"Config Path: {CONFIG_PATH}")
    print(f"Results Path: {RESULTS_PATH}")

    if not CONFIG_PATH.exists():
        # This is a critical check, as no configs means nothing to plot.
        raise NotADirectoryError(f"CRITICAL: CONFIG_PATH does not exist: {CONFIG_PATH}")


def rewrite_varying_horizon(model:str, stock:str) -> None:
    MODEL_PATH = CONFIG_PATH / model / stock
    
    unique_horizon_dicts = {}

    for conf_filename in os.listdir(MODEL_PATH):
        with open(MODEL_PATH / conf_filename, 'r') as f:
            config = yaml.safe_load(f)
        c = ConfDataClass(
            horizon_len=config['horizon_length'],
            results_filepath=config['results_file_path'],
            training_period_value=config['training_period_value'])
        
        if c.horizon_len not in unique_horizon_dicts:
            unique_horizon_dicts[c.horizon_len] = []
        
        # here we create a dictionary with the horizon as key
        # and the config as value
        unique_horizon_dicts[c.horizon_len].append(c)

    
    if not unique_horizon_dicts:
        print(f"No configurations found for model {model} under {MODEL_PATH}")
        return

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    # line: name
    global_legent = {}

    for idx, (horizon_len, configs_for_this_horizon) in enumerate(sorted(unique_horizon_dicts.items())):
        if not configs_for_this_horizon:
            print(f"No configurations loaded for horizon: {horizon_len}")
            continue

        else:
            configs_for_this_horizon.sort(key=lambda x: x.training_period_value)
    
        PLOT_GROUND_TRUTH_ON_THIS_AX = True 
        for i, conf_data in enumerate(configs_for_this_horizon):
            # print(f"PLotting for {conf_data.training_period_value} and {conf_data.horizon_len}")
            csv_file_path = Path(conf_data.results_filepath) 
            resutlts_df = pd.read_csv(csv_file_path)

            x_values = pd.to_datetime(resutlts_df["Date"])
            x_label = "Date"

            if PLOT_GROUND_TRUTH_ON_THIS_AX :
                gt_label = "Actual Values"
                line, = axs[idx].plot(x_values, resutlts_df['Actual'], label=gt_label, color='black')

                global_legent[gt_label] = line
                PLOT_GROUND_TRUTH_ON_THIS_AX = False 

            forecast_col_name = f"{model}_Forecast" 
            label = f"Train Days: {conf_data.training_period_value}"

            line, = axs[idx].plot(x_values, resutlts_df[forecast_col_name], 
                     label=label)
            global_legent[label] = line

            
            axs[idx].set_xlabel(x_label)
            axs[idx].set_ylabel("Price")
            axs[idx].tick_params(axis='x', rotation=30)
            axs[idx].tick_params(axis='y')
            axs[idx].grid(True, linestyle=':', alpha=0.6)

    fig.legend(global_legent.values(), global_legent.keys(),
            #    loc='center right', bbox_to_anchor=(1, 0.5), 
               loc='outside lower center',
               ncol=5, fontsize=9)
            #    ncol=1, fontsize=9)
    for i, (horizon_len, _) in enumerate(sorted(unique_horizon_dicts.items())):
        if i < len(axs):
            axs[i].set_title(f"Forecast Horizon: {horizon_len} {'days' if horizon_len > 1 else 'day'}", fontsize=10)
    # plt.tight_layout(rect=[0, 0.05, 0.85, 0.95])
    
    fig.suptitle(f"Forecasts for {stock} using {model.upper()} Model: Varying Prediction Horizons", fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    os.makedirs(FIGURES_PATH / model / stock, exist_ok=True)
    # plt.savefig(FIGURES_PATH / model / f"{model}_varying_horizon.eps", format='eps')
    plt.savefig(FIGURES_PATH / model / stock / f"{model}_varying_horizon_tight.png", format='png', dpi=300, bbox_inches='tight')
    # plt.show()    # Add a single legend for the entire figure





# def plot_varying_train_size(model:str) -> None:
#     """Plots every configs based on the train size used"""
#     # get the unique horizon lens
#     #TODO: Change this when will add more stocks
#     MODEL_PATH = CONFIG_PATH / model / "MSFT"
    
#     unique_train_len = {}
    
#     for conf_filename in os.listdir(MODEL_PATH):
#         with open(MODEL_PATH / conf_filename, 'r') as f:
#             config = yaml.safe_load(f)

#         c = ConfDataClass(horizon_len=config['horizon_length'],
#                         results_filepath=config['results_file_path'],
#                         training_period_value=config['training_period_value'])
        
#         if c.horizon_len not in unique_train_len:
#             unique_train_len[c.training_period_value] = []
#         unique_train_len[c.training_period_value].append(c)

#     # now we have our configs
#     # print(f"Found train lens: {sorted(list(unique_train_len.keys()))}")

#     for horizon_len, configs_for_this_horizon in sorted(unique_train_len.items()):
#         if not configs_for_this_horizon:
#             print(f"No configurations loaded for horizon: {horizon_len}")
#             continue

#         # print(f"  Plotting for Horizon: {horizon_len} days, Number of configs: {len(configs_for_this_horizon)}")

#         configs_for_this_horizon.sort(key=lambda x: x.horizon_len)



if __name__ == '__main__':
    print(f"--- Starting Plot Generation for Experiment: {EXPERIMENT_NAME} ---")

    setup_paths(EXPERIMENT_NAME)

    for model in os.listdir(CONFIG_PATH):
        rewrite_varying_horizon(model, 'MSFT')
        # plot_varying_train_size(model)
        # TODO: RM for every model to run
        # exit()