import argparse
import os
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
        palette = plt.cm.get_cmap("viridis", len(model_methods))
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots for a given ticker and time frequency."
    )
    parser.add_argument("--ticker", required=True, help="Stock ticker (e.g., MSFT).")
    parser.add_argument("--timefreq", required=True, help="Time frequency (e.g., 1d).")
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

    # Generate metric plots
    plot_metrics_vs_training_days(df_full, args.ticker, args.timefreq, figures_root)

    # Here you could add calls to other refactored plotting functions
    # e.g., plot_varying_horizon, plot_metrics_across_horizon

    print("--- Plot Generation Complete ---")


if __name__ == "__main__":
    main()

