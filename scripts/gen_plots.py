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


def varying_horizon(model:str, stock:str) -> None:
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
    plt.savefig(FIGURES_PATH / model / stock / f"{model}_{stock}_varying_horizon_tight.png", format='png', dpi=300, bbox_inches='tight')
    # plt.show()    # Add a single legend for the entire figure

# def varying_train_size(model: str, stock: str) -> None:
#     """Plots forecasts for a given model and stock, creating a subplot for each unique training data size."""
#     MODEL_PATH = CONFIG_PATH / model / stock
    
#     unique_train_size_dicts = {}

#     for conf_filename in os.listdir(MODEL_PATH):
#         with open(MODEL_PATH / conf_filename, 'r') as f:
#             config = yaml.safe_load(f)
        
#         c = ConfDataClass(
#             horizon_len=config['horizon_length'],
#             results_filepath=config['results_file_path'],
#             training_period_value=config['training_period_value']
#         )
        
#         if c.training_period_value not in unique_train_size_dicts:
#             unique_train_size_dicts[c.training_period_value] = []
        
#         unique_train_size_dicts[c.training_period_value].append(c)

    
#     if not unique_train_size_dicts:
#         print(f"No configurations found for model {model}, stock {stock} under {MODEL_PATH}")
#         return

#     fig, axs = plt.subplots(5, 2, figsize=(12, 15)) 
#     axs = axs.flatten() # Flatten to easily index as axs[i]

#     global_legend = {}

#     for idx, (train_size_val, configs_for_this_train_size) in enumerate(sorted(unique_train_size_dicts.items())):
#         if not configs_for_this_train_size:
#             print(f"No configurations loaded for training size: {train_size_val}")
#             continue
#         # this will be removed
#         # if idx >= len(axs): # Protective break to prevent IndexError if more unique train sizes than subplots
#         #     print(f"Warning: More unique training sizes ({len(unique_train_size_dicts)}) than available subplots ({len(axs)}). Plotting first {len(axs)}.")
#         #     break

#         # Sort by horizon length for consistent line plotting within a subplot
#         configs_for_this_train_size.sort(key=lambda x: x.horizon_len)
    
#         PLOT_GROUND_TRUTH_ON_THIS_AX = True 
#         current_ax = axs[idx] 

#         for i, conf_data in enumerate(configs_for_this_train_size):
#             csv_file_path = Path(conf_data.results_filepath) 
#             results_df = pd.read_csv(csv_file_path)

#             # Assumes 'Date' column exists and is convertible to datetime
#             x_values = pd.to_datetime(results_df["Date"])
#             x_label = "Date"

#             if PLOT_GROUND_TRUTH_ON_THIS_AX :
#                 gt_label = "Actual Values"
#                 line, = current_ax.plot(x_values, results_df['Actual'], label=gt_label, color='black')
#                 if gt_label not in global_legend: # Add to global legend only once per unique label
#                     global_legend[gt_label] = line
#                 PLOT_GROUND_TRUTH_ON_THIS_AX = False 

#             forecast_col_name = f"{model}_Forecast" 
#             # Assumes forecast column exists
#             label = f"Horizon: {conf_data.horizon_len} {'days' if conf_data.horizon_len > 1 else 'day'}"

#             line, = current_ax.plot(x_values, results_df[forecast_col_name], label=label)
#             if label not in global_legend: # Add to global legend only once per unique label
#                  global_legend[label] = line
            
#             current_ax.set_xlabel(x_label)
#             current_ax.set_ylabel("Price")
#             current_ax.tick_params(axis='x', rotation=30)
#             current_ax.tick_params(axis='y')
#             current_ax.grid(True, linestyle=':', alpha=0.6)

#     # Setting titles in a separate loop, similar to varying_horizon
#     # The check 'if i < len(axs)' is present in your varying_horizon code.
#     for i, (train_size_val, _) in enumerate(sorted(unique_train_size_dicts.items())):
#         if i < len(axs): # This check is from your varying_horizon structure
#             axs[i].set_title(f"Training Data: {train_size_val} days", fontsize=10)
    
#     # Remove unused subplots if fewer unique train_sizes than subplot cells (e.g. 2x2=4)
#     # This makes the plot cleaner if, for example, only 1 or 2 train_sizes are plotted.
#     plotted_train_size_count = min(len(unique_train_size_dicts), len(axs))
#     for i in range(plotted_train_size_count, len(axs)):
#         fig.delaxes(axs[i])

#     if global_legend: # Only add legend if there are items in it
#         fig.legend(global_legend.values(), global_legend.keys(),
#                    loc='outside lower center',
#                    ncol=min(5, len(global_legend)), fontsize=9) # Dynamically adjust ncol
    
#     fig.suptitle(f"Forecasts for {stock.upper()} using {model.upper()} Model: Varying Training Data Size", fontsize=14)
#     # Adjust rect based on whether legend is present, to prevent overlap
#     bottom_rect = 0.08 if global_legend else 0.03
#     fig.tight_layout(rect=[0, bottom_rect, 1, 0.95])


#     # Assumes FIGURES_PATH / model / stock exists or can be created
#     output_dir = FIGURES_PATH / model / stock
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Filename similar to varying_horizon's output naming convention
#     # Using .png as in your varying_horizon example's savefig.
#     plt.savefig(output_dir / f"{model}_{stock}_varying_train_size.png", format='png', dpi=300, bbox_inches='tight')
#     # plt.show() # Kept commented as in original varying_horizon
#     # plt.close(fig) # Good practice to close figures when done, especially in scripts

# TODO MAE vs Training data: MAE y axis and Train data as X axis and horizon as the subplot
def plot_mape_vs_training_days_by_horizon_matplotlib(stock_ticker: str = "MSFT"):
    """
    Plots MAPE vs. Training Data Size, faceted by Horizon, using Matplotlib.
    This function replicates the sns.relplot from the notebook.
    """

    mape_col = 'MAPE'
    days_col = 'training_period_value'
    horizon_col = 'forecast_horizon'
    model_type_col = 'Method'

    all_results_data = []
    model_methods = {
        "naive": "Naive",
        "arima": "ARIMA",
        "fm": "TimesFM" # Assuming 'fm' is the directory name for TimesFM
    }

    for method_dir, method_name in model_methods.items():
        MODEL_CONFIG_PATH = CONFIG_PATH / method_dir / stock_ticker
        if not MODEL_CONFIG_PATH.exists():
            print(f"Warning: Config path for {method_name} ({MODEL_CONFIG_PATH}) does not exist. Skipping.")
            continue

        for conf_filename in os.listdir(MODEL_CONFIG_PATH):
            if not conf_filename.endswith(".yaml"): # Process only YAML files
                continue
            with open(MODEL_CONFIG_PATH / conf_filename, 'r') as f:
                config = yaml.safe_load(f)
            
            all_results_data.append({
                model_type_col: method_name,
                days_col: config['training_period_value'],
                horizon_col: config['horizon_length'],
                mape_col: config['evaluation_metrics']['mape']
            })

    if not all_results_data:
        print(f"No data loaded for experiment {EXPERIMENT_NAME}, stock {stock_ticker}. Cannot generate plot.")
        return

    df_mape_results = pd.DataFrame(all_results_data)

    if df_mape_results.empty:
        print(f"DataFrame is empty after loading configs for {stock_ticker}. Cannot generate plot.")
        return

    # Define consistent plotting styles
    palette = {'Naive': 'grey', 'ARIMA': 'orange', 'TimesFM': 'blue'}
    markers_map = {'Naive': 'o', 'ARIMA': 's', 'TimesFM': '^'}

    unique_horizons = sorted(df_mape_results[horizon_col].unique())
    n_horizons = len(unique_horizons)
    if n_horizons == 0:
        print(f"No unique horizons found for {stock_ticker}. Cannot generate plot.")
        return
        
    ncols = 2
    nrows = int(np.ceil(n_horizons / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False, squeeze=False)
    axs_flat = axs.flatten()

    plot_handles = []
    plot_labels = []

    for idx, horizon_val in enumerate(unique_horizons):
        ax = axs_flat[idx]
        data_for_horizon = df_mape_results[df_mape_results[horizon_col] == horizon_val]

        for method in model_methods.values(): # Iterate in defined order for consistent legend
            if method not in palette: continue # Skip if method not in palette
            
            method_data = data_for_horizon[data_for_horizon[model_type_col] == method].sort_values(by=days_col)
            if not method_data.empty:
                line, = ax.plot(method_data[days_col], method_data[mape_col], 
                                label=method, 
                                color=palette.get(method, 'black'), # Use .get for safety
                                marker=markers_map.get(method, None), # Use .get for safety
                                linestyle='-')
                if method not in plot_labels: # Collect handles/labels for figure legend
                    plot_handles.append(line)
                    plot_labels.append(method)

        ax.set_title(f"Horizon: {horizon_val} {'days' if horizon_val > 1 else 'day'}")
        ax.set_xlabel('Days of Training/Context Data')
        ax.set_ylabel(f'{mape_col} (Lower is Better)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)

    # Remove any unused subplots
    for i in range(n_horizons, nrows * ncols):
        fig.delaxes(axs_flat[i])

    if plot_handles:
        fig.legend(plot_handles, plot_labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=len(model_methods))

    fig.suptitle(f'MAPE vs. Training/Context Data Size by Horizon for {stock_ticker} ({EXPERIMENT_NAME})', fontsize=14, y=1.03 if nrows ==1 else 0.98) # Adjust y for suptitle
    fig.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust rect to make space for legend

    output_dir = FIGURES_PATH / "mape_analysis" / stock_ticker
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = output_dir / f"mape_vs_training_days_by_horizon_{stock_ticker}_{EXPERIMENT_NAME}.png"
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_filename}")
    plt.close(fig) # Close the figure to free memory


if __name__ == '__main__':
    print(f"--- Starting Plot Generation for Experiment: {EXPERIMENT_NAME} ---")

    setup_paths(EXPERIMENT_NAME)

    # for model in os.listdir(CONFIG_PATH):
    #     varying_horizon(model, 'MSFT')
        
    # TODO Add previous plots from notebook
    plot_mape_vs_training_days_by_horizon_matplotlib(stock_ticker="MSFT")