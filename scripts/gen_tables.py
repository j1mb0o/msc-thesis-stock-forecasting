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
    model_methods = {
        "naive": "Naive",
        "arima": "ARIMA",
        "fm": "TimesFM",
        "sundial": "Sundial",
        "chronos_base": "Chronos"
    }
    model_name = model_methods.get(model, model.upper())

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

        df_data = []
        previous_metrics = {}
        metric_columns = ['MSE', 'MAE', 'RMSE', 'MAPE (%)']
        
        for conf_data in configs_for_this_horizon:
            current_metrics = {
                'MSE': conf_data.mse,
                'MAE': conf_data.mae,
                'RMSE': conf_data.rmse,
                'MAPE (%)': conf_data.mape
            }
            
            row_for_df = {
                'Training Period (Days)': conf_data.training_period_value,
                **current_metrics
            }

            if previous_metrics:
                for metric in metric_columns:
                    current_value = current_metrics[metric]
                    prev_value = previous_metrics.get(metric, 0)
                    if prev_value != 0:
                        percent_change = ((current_value - prev_value) / prev_value) * 100
                        row_for_df[f'{metric} % Change'] = f'{percent_change:.2f}%'
                    else:
                        row_for_df[f'{metric} % Change'] = 'N/A'
            
            df_data.append(row_for_df)
            previous_metrics = current_metrics
        
        if not df_data:
            print(f"No data to process for horizon {horizon_len}")
            continue

        df = pd.DataFrame(df_data)

        # Reorder columns for better readability
        ordered_cols = ['Training Period (Days)']
        change_cols = []
        for metric in metric_columns:
            ordered_cols.append(metric)
            change_col_name = f'{metric} % Change'
            if change_col_name in df.columns:
                ordered_cols.append(change_col_name)
                change_cols.append(change_col_name)
        df = df[ordered_cols]

        df = df.set_index('Training Period (Days)')
        
        # Calculate average percentile change for each metric
        avg_row_data = {}
        for col in change_cols:
            numeric_changes = pd.to_numeric(df[col].str.replace('%', '', regex=False), errors='coerce')
            if numeric_changes.notna().any():
                avg_change = numeric_changes.mean()
                avg_row_data[col] = f'{avg_change:.2f}%'
        
        if avg_row_data:
            avg_df = pd.DataFrame([avg_row_data], index=pd.Index(['Average % Change'], name=df.index.name))
            df = pd.concat([df, avg_df])

        # --- Prepare and Save Outputs ---
        output_dir = TABLES_PATH / model / stock
        os.makedirs(output_dir, exist_ok=True)
        
        # Optional: Save the raw data to a CSV file (with separate columns)
        table_filename = output_dir / f"{model}_{stock}_horizon_{horizon_len}_metrics.csv"
        # df.to_csv(table_filename) 

        # Transform DataFrame for LaTeX output:
        # Combines metric and % change into one column and escapes '%' for LaTeX.
        df_latex = df.copy()
        for metric in metric_columns:
            change_col = f'{metric} % Change'
            if change_col in df_latex.columns:
                
                new_col_values = []
                for index, row in df_latex.iterrows():
                    metric_val = row[metric]
                    change_val_str = row[change_col]

                    # Prepare the escaped change string
                    escaped_change_str = None
                    if pd.notna(change_val_str) and change_val_str != 'N/A':
                        escaped_change_str = change_val_str.replace('%', r'\%')

                    # Handle the 'Average % Change' row specifically
                    if index == 'Average % Change':
                        new_col_values.append(f"({escaped_change_str})" if escaped_change_str else '-')
                    else:
                        # Handle regular data rows
                        val_str = f"{metric_val:.4f}" if pd.notna(metric_val) else "-"
                        if escaped_change_str:
                            new_col_values.append(f"{val_str} ({escaped_change_str})")
                        else:
                            new_col_values.append(val_str)
                
                # Update the metric column and drop the now-redundant change column
                df_latex[metric] = new_col_values
                df_latex.drop(columns=[change_col], inplace=True)

        # Escape '%' in column names and index for LaTeX, as to_latex(escape=False) is used.
        df_latex.columns = [str(col).replace('%', r'\%') for col in df_latex.columns]
        df_latex.index = [str(idx).replace('%', r'\%') if isinstance(idx, str) else idx for idx in df_latex.index]
        if df_latex.index.name:
            df_latex.index.name = df_latex.index.name.replace('%', r'\%')

        # Save the transformed DataFrame to a LaTeX file
        latex_filename = output_dir / f"{model}_{stock}_horizon_{horizon_len}_metrics.tex"
        df_latex.to_latex(latex_filename, caption=f"Evaluation Metrics for {model_name} on {stock} with Horizon {horizon_len} {'days' if horizon_len > 1 else 'day'}", label=f"tab:{model}_{stock}_h{horizon_len}_metrics", na_rep='-', escape=False)
        
        print(f"Table for Horizon {horizon_len} saved to {latex_filename}")




if __name__ == '__main__':
    print(f"--- Starting Plot Generation for Experiment: {EXPERIMENT_NAME} ---")
    setup_paths(EXPERIMENT_NAME)
    # varying_horizon('fm', 'MSFT')
    for model in os.listdir(CONFIG_PATH):
        varying_horizon(model, 'MSFT')
        
    