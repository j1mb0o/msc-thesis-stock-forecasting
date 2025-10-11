import argparse
from pathlib import Path
import yaml
import pandas as pd

def load_config_data(config_root):
    """Loads all YAML configs from a directory into a pandas DataFrame."""
    all_results_data = []
    for config_file in config_root.glob("**/*.yaml"):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        metrics = config.pop("evaluation_metrics", {})
        config.update(metrics)
        config['experiment_name'] = config_file.parent.name
        all_results_data.append(config)
    
    if not all_results_data:
        return pd.DataFrame()
        
    return pd.DataFrame(all_results_data)

def generate_latex_table(df_group, tables_root, ticker, timefreq, method, exp_name, horizon):
    """Generates and saves a single LaTeX table for a group of experiment results."""
    
    df_group = df_group.sort_values(by='training_period_value')

    df_data = []
    previous_metrics = {}
    metric_columns = ['MSE', 'MAE', 'RMSE', 'MAPE (%)', 'SMAPE (%)']

    for _, row in df_group.iterrows():
        current_metrics = {
            'MSE': row['mse'],
            'MAE': row['mae'],
            'RMSE': row['rmse'],
            'MAPE (%)': row['mape'],
            'SMAPE (%)': row['smape']
        }
        
        row_for_df = {
            'Training Period (Days)': row['training_period_value'],
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
        return

    df = pd.DataFrame(df_data).set_index('Training Period (Days)')

    # --- Prepare and Save Outputs ---
    output_dir = tables_root / method / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Transform DataFrame for LaTeX output
    df_latex = df.copy()
    for metric in metric_columns:
        change_col = f'{metric} % Change'
        if change_col in df_latex.columns:
            new_col_values = []
            for index, row in df_latex.iterrows():
                metric_val = row[metric]
                change_val_str = row[change_col]
                escaped_change_str = None
                if pd.notna(change_val_str) and change_val_str != 'N/A':
                    escaped_change_str = change_val_str.replace('%', r'\%')
                
                val_str = f"{metric_val:.4f}" if pd.notna(metric_val) else "-"
                if escaped_change_str:
                    new_col_values.append(f"{val_str} ({escaped_change_str})")
                else:
                    new_col_values.append(val_str)
            
            df_latex[metric] = new_col_values
            df_latex.drop(columns=[change_col], inplace=True)

    df_latex.columns = [str(col).replace('%', r'\%') for col in df_latex.columns]
    df_latex.index.name = df_latex.index.name.replace('%', r'\%')

    latex_filename = output_dir / f"{method}_{ticker}_h{horizon}_metrics.tex"
    model_name = method.upper()
    df_latex.to_latex(latex_filename, 
                      caption=f"Evaluation Metrics for {model_name} on {ticker} with Horizon {horizon} {'days' if horizon > 1 else 'day'}", 
                      label=f"tab:{method}_{ticker}_h{horizon}_metrics", 
                      na_rep='-', 
                      escape=False)
    
    print(f"Table saved to {latex_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX metric tables for a given ticker and time frequency.")
    parser.add_argument("--ticker", required=True, help="Stock ticker (e.g., MSFT).")
    parser.add_argument("--timefreq", required=True, help="Time frequency (e.g., 1d).")
    args = parser.parse_args()

    config_root = Path("configs") / args.ticker / args.timefreq
    tables_root = Path("tables") / args.ticker / args.timefreq

    if not config_root.is_dir():
        raise NotADirectoryError(f"CRITICAL: Config directory does not exist: {config_root}")

    df_full = load_config_data(config_root)

    if df_full.empty:
        print(f"No configuration data found in {config_root}. Exiting.")
        return

    print(f"--- Starting Table Generation for Ticker: {args.ticker}, Timefreq: {args.timefreq} ---")

    # Group data to generate one table per model, experiment, and horizon
    grouped = df_full.groupby(['forecasting_method', 'experiment_name', 'horizon_length'])

    for (method, exp_name, horizon), group in grouped:
        generate_latex_table(group, tables_root, args.ticker, args.timefreq, method, exp_name, horizon)

    print("--- Table Generation Complete ---")

if __name__ == "__main__":
    main()