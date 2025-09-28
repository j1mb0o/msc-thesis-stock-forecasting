import argparse
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_config_data(config_root):
    """Loads all YAML configs from a directory into a pandas DataFrame."""
    all_results_data = []
    for config_file in config_root.glob("**/*.yaml"):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        metrics = config.pop("evaluation_metrics", {})
        config.update(metrics)

        # Add identifiers from the path
        config["experiment_name"] = config_file.parent.name
        config["method_from_path"] = config_file.parent.parent.name

        all_results_data.append(config)

    if not all_results_data:
        return pd.DataFrame()

    return pd.DataFrame(all_results_data)


def plot_metrics_vs_training_days(df, ticker, timefreq, figures_root):
    """
    Plots metrics (e.g., MAPE, MAE) vs. Training Data Size, faceted by Horizon.
    """
    if df.empty:
        return

    for metric_to_plot in ["MAPE", "MAE", "RMSE"]:
        unique_horizons = sorted(df["horizon_length"].unique())
        n_horizons = len(unique_horizons)
        if n_horizons == 0:
            continue

        ncols = 2
        nrows = int(np.ceil(n_horizons / ncols))
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False, squeeze=False
        )
        axs_flat = axs.flatten()

        plot_handles, plot_labels = [], []

        model_methods = df["forecasting_method"].unique()
        palette = plt.get_cmap("viridis", len(model_methods))
        color_map = {method: palette(i) for i, method in enumerate(model_methods)}
        markers_map = {
            method: marker
            for marker, method in zip(["o", "s", "^", "d", "p", "X"], model_methods)
        }

        for idx, horizon_val in enumerate(unique_horizons):
            ax = axs_flat[idx]
            data_for_horizon = df[df["horizon_length"] == horizon_val]

            for method in model_methods:
                method_data = data_for_horizon[
                    data_for_horizon["forecasting_method"] == method
                ].sort_values(by="training_period_value")
                if not method_data.empty:
                    (line,) = ax.plot(
                        method_data["training_period_value"],
                        method_data[metric_to_plot.lower()],
                        label=method,
                        color=color_map.get(method),
                        marker=markers_map.get(method),
                        linestyle="-",
                    )
                    if method not in plot_labels:
                        plot_handles.append(line)
                        plot_labels.append(method)

            unit = (
                data_for_horizon["training_period_unit"].iloc[0]
                if not data_for_horizon.empty
                else "units"
            )
            ax.set_title(
                f"Horizon: {horizon_val} {'days' if horizon_val > 1 else 'day'}"
            )
            ax.set_xlabel(f"Training/Context Data ({unit.capitalize()})")
            y_label = f"{metric_to_plot} (Lower is Better)"
            if metric_to_plot == "MAPE":
                y_label = f"{metric_to_plot} (%) (Lower is Better)"
            ax.set_ylabel(y_label)
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)
            ax.tick_params(axis="x", rotation=45)

        for i in range(n_horizons, nrows * ncols):
            fig.delaxes(axs_flat[i])

        if plot_handles:
            fig.legend(
                plot_handles,
                plot_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.05),
                ncol=len(model_methods),
            )

        fig.suptitle(
            f"{metric_to_plot} vs. Training Data Size for {ticker} ({timefreq})",
            fontsize=14,
            y=1.03,
        )
        fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.95])

        output_dir = figures_root / "metrics_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_filename = output_dir / f"{metric_to_plot}_vs_training_days.png"
        plt.savefig(plot_filename, format="png", dpi=300, bbox_inches="tight")
        print(f"Plot saved to {plot_filename}")
        plt.close(fig)


def plot_predictions_vs_actuals(df, ticker, timefreq, figures_root):
    """
    For each method, plots predictions vs. actuals.
    If multiple horizons are present, they are shown in subplots.
    """
    if df.empty:
        return

    grouped_by_exp_method = df.groupby(['experiment_name', 'forecasting_method'])

    for (experiment_name, method), group in grouped_by_exp_method:
        
        unique_horizons = sorted(group['horizon_length'].unique())
        n_horizons = len(unique_horizons)
        if n_horizons == 0:
            continue

        plot_handles, plot_labels = [], []
        training_periods = sorted(group['training_period_value'].unique())
        palette = plt.get_cmap('tab20', len(training_periods))
        color_map = {val: palette(i) for i, val in enumerate(training_periods)}
        markers = ['o', 's', '^', 'd', 'p', '*', 'X', '+', 'v', '<', '>']
        marker_map = {val: markers[i % len(markers)] for i, val in enumerate(training_periods)}

        if n_horizons == 1:
            fig, axs = plt.subplots(1, 1, figsize=(12, 6), squeeze=False)
        else:
            ncols = 2
            nrows = int(np.ceil(n_horizons / ncols))
            fig, axs = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4 * nrows), sharex=True, squeeze=False)
        
        axs_flat = axs.flatten()

        for idx, horizon_val in enumerate(unique_horizons):
            ax = axs_flat[idx]
            data_for_horizon = group[group['horizon_length'] == horizon_val]

            if data_for_horizon.empty:
                continue

            first_row = data_for_horizon.iloc[0]
            results_path = Path(first_row["results_file_path"])
            if not results_path.exists(): continue
            
            preds_df = pd.read_csv(results_path)
            preds_df['Date'] = pd.to_datetime(preds_df['Date'])
            actual_col = "Actual" if "Actual" in preds_df.columns else "y_true"
            
            if actual_col not in preds_df.columns: continue
            
            ax.plot(preds_df["Date"], preds_df[actual_col], label="Actual", color='black', linewidth=2)

            for _, row in data_for_horizon.sort_values(by='training_period_value').iterrows():
                results_path = Path(row["results_file_path"])
                if not results_path.exists(): continue

                preds_df = pd.read_csv(results_path)
                preds_df['Date'] = pd.to_datetime(preds_df['Date'])
                
                forecast_col = f"{row['forecasting_method']}_Forecast"
                if forecast_col not in preds_df.columns and 'y_pred' in preds_df.columns:
                    forecast_col = 'y_pred'

                if forecast_col not in preds_df.columns: continue
                
                train_val = row['training_period_value']
                train_unit = row['training_period_unit']
                label = f"Train: {train_val}{train_unit[0]}"
                
                line, = ax.plot(preds_df["Date"], preds_df[forecast_col], label=label, color=color_map[train_val], linestyle="--", alpha=0.8, marker=marker_map[train_val], markersize=3, markevery=20)
                
                if label not in plot_labels:
                    plot_handles.append(line)
                    plot_labels.append(label)

            if n_horizons > 1:
                ax.set_title(f"Horizon: {horizon_val} {'days' if horizon_val > 1 else 'day'}")
            
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.tick_params(axis="x", rotation=45)

        for i in range(n_horizons, axs.size):
            fig.delaxes(axs_flat[i])

        if n_horizons == 1:
            fig.suptitle(f"'{method}' Predictions vs. Actuals for {ticker} ({timefreq})\nExperiment: {experiment_name}, Horizon: {unique_horizons[0]} {'days' if unique_horizons[0] > 1 else 'day'}", fontsize=16)
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        else:
            fig.suptitle(f"'{method}' Predictions vs. Actuals for {ticker} ({timefreq})\nExperiment: {experiment_name}", fontsize=16, y=1.03)
            fig.tight_layout(rect=[0, 0.05, 1, 1])

        if plot_handles:
            fig.legend(plot_handles, plot_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05 if n_horizons > 1 else -0.15), ncol=min(6, len(plot_handles)), fancybox=True, shadow=True)

        output_dir = figures_root / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{method}_predictions.png" if n_horizons == 1 else f"{method}_varying_horizon_predictions.png"
        plot_filename = output_dir / filename
        plt.savefig(plot_filename, format="png", dpi=300, bbox_inches='tight')
        print(f"Aggregated prediction plot saved to {plot_filename}")
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Generate plots for a given ticker and time frequency."
    )
    parser.add_argument("--ticker", required=True, help="Stock ticker (e.g., MSFT).")
    parser.add_argument("--timefreq", required=True, help="Time frequency (e.g., 1d).")
    parser.add_argument(
        "--plot-type",
        type=str,
        default="all",
        choices=["metrics", "predictions", "all"],
        help="Type of plots to generate.",
    )
    args = parser.parse_args()

    config_root = Path("configs") / args.ticker / args.timefreq
    figures_root = Path("figures") / args.ticker / args.timefreq

    if not config_root.is_dir():
        raise NotADirectoryError(
            f"CRITICAL: Config directory does not exist: {config_root}"
        )

    df_full = load_config_data(config_root)

    if df_full.empty:
        print(f"No configuration data found in {config_root}. Exiting.")
        return

    print(
        f"--- Starting Plot Generation for Ticker: {args.ticker}, Timefreq: {args.timefreq} ---"
    )

    if args.plot_type in ["metrics", "all"]:
        print("--- Generating Metric Plots ---")
        plot_metrics_vs_training_days(df_full, args.ticker, args.timefreq, figures_root)

    if args.plot_type in ["predictions", "all"]:
        print("--- Generating Prediction vs. Actuals Plots ---")
        plot_predictions_vs_actuals(df_full, args.ticker, args.timefreq, figures_root)

    print("--- Plot Generation Complete ---")


if __name__ == "__main__":
    main()
