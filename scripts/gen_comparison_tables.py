#!/usr/bin/env python3
"""
Generate model comparison tables from experiment results.

This script reads config YAML files from experiments and creates comparison
tables showing all models' performance at each training size.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_experiment_metrics(
    ticker: str,
    timefreq: str,
    experiment_type: str,
    horizon: int = 1
) -> pd.DataFrame:
    """
    Load metrics from all models for a specific experiment.

    Args:
        ticker: Stock ticker (e.g., 'MSFT')
        timefreq: Time frequency ('1h' or '1d')
        experiment_type: Experiment type ('train-less-year-log', 'train-less-year-linear', 'train-restricted-years')
        horizon: Forecast horizon (default: 1)

    Returns:
        DataFrame with columns: model, training_size, mse, mae, rmse, mape
    """
    base_path = Path("configs") / ticker / timefreq
    models = ['arima', 'chronos_base', 'sundial', 'fm', 'naive']

    all_data = []

    for model in models:
        model_path = base_path / model / experiment_type

        if not model_path.exists():
            logger.warning(f"Path does not exist: {model_path}")
            continue

        config_files = list(model_path.glob(f"*_horizon_{horizon}.yaml"))

        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)

                metrics = config.get('evaluation_metrics', {})
                training_size = config.get('training_period_value')
                training_unit = config.get('training_period_unit', 'days')

                if training_unit == 'years':
                    training_size = training_size * 365

                all_data.append({
                    'model': model,
                    'training_size': training_size,
                    'mse': metrics.get('mse'),
                    'mae': metrics.get('mae'),
                    'rmse': metrics.get('rmse'),
                    'mape': metrics.get('mape', metrics.get('mape_percent'))
                })

            except Exception as e:
                logger.error(f"Error loading {config_file}: {e}")
                continue

    return pd.DataFrame(all_data).sort_values('training_size')


def generate_comparison_table(
    df: pd.DataFrame,
    metric: str,
    caption: str,
    label: str
) -> str:
    """
    Generate LaTeX comparison table for a specific metric.

    Args:
        df: DataFrame with model metrics
        metric: Metric to display ('mse', 'mae', 'rmse', 'mape')
        caption: Table caption
        label: LaTeX label

    Returns:
        LaTeX table string
    """
    pivot = df.pivot(index='training_size', columns='model', values=metric)

    model_order = ['arima', 'chronos_base', 'sundial', 'fm', 'naive']
    available_models = [m for m in model_order if m in pivot.columns]
    pivot = pivot[available_models]

    column_names = {
        'arima': 'ARIMA',
        'chronos_base': 'Chronos',
        'sundial': 'Sundial',
        'fm': 'TimesFM',
        'naive': 'Naive'
    }
    pivot.columns = [column_names.get(col, col) for col in pivot.columns]

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append("\\small")

    col_spec = "l" + "r" * len(pivot.columns) + "l"
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")

    header = "Training Days & " + " & ".join(pivot.columns) + " & Best Model \\\\"
    latex.append(header)
    latex.append("\\midrule")

    for train_size, row in pivot.iterrows():
        best_model = row.idxmin()

        formatted_values = []
        for model_name, val in row.items():
            if pd.isna(val):
                formatted_values.append("--")
            elif model_name == best_model:
                formatted_values.append(f"\\textbf{{{val:.4f}}}")
            else:
                formatted_values.append(f"{val:.4f}")

        row_str = f"{int(train_size)} & " + " & ".join(formatted_values) + f" & {best_model} \\\\"
        latex.append(row_str)

    latex.append("\\midrule")

    mean_vals = pivot.mean()
    best_mean_model = mean_vals.idxmin()
    formatted_means = []
    for model_name, val in mean_vals.items():
        if pd.isna(val):
            formatted_means.append("--")
        elif model_name == best_mean_model:
            formatted_means.append(f"\\textbf{{{val:.4f}}}")
        else:
            formatted_means.append(f"{val:.4f}")

    mean_row = "\\textit{Mean} & " + " & ".join(formatted_means) + f" & {best_mean_model} \\\\"
    latex.append(mean_row)

    best_counts = pivot.idxmin(axis=1).value_counts()
    count_strs = []
    for model_name in pivot.columns:
        count = best_counts.get(model_name, 0)
        count_strs.append(str(count))

    count_row = "\\textit{\\# Best} & " + " & ".join(count_strs) + " & -- \\\\"
    latex.append(count_row)

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(
        description="Generate model comparison tables from experiment results"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="MSFT",
        help="Stock ticker (default: MSFT)"
    )
    parser.add_argument(
        "--timefreq",
        type=str,
        default="1h",
        choices=["1h", "1d"],
        help="Time frequency (default: 1h)"
    )
    parser.add_argument(
        "--exp-type",
        type=str,
        default="train-less-year-log",
        choices=["train-less-year-log", "train-less-year-linear", "train-restricted-years"],
        help="Experiment type (default: train-less-year-log)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tables/comparisons",
        help="Output directory for LaTeX tables (default: tables/comparisons)"
    )

    args = parser.parse_args()

    logger.info(f"Loading experiment data: {args.ticker}/{args.timefreq}/{args.exp_type} (h={args.horizon})")

    df = load_experiment_metrics(
        ticker=args.ticker,
        timefreq=args.timefreq,
        experiment_type=args.exp_type,
        horizon=args.horizon
    )

    if df.empty:
        logger.error("No data loaded! Check your paths and experiment configuration.")
        return

    logger.info(f"Loaded {len(df)} experiment results")
    logger.info(f"Models: {df['model'].unique()}")
    logger.info(f"Training sizes: {sorted(df['training_size'].unique())}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ('mse', 'Mean Squared Error (MSE)', 'MSE'),
        ('mae', 'Mean Absolute Error (MAE)', 'MAE'),
        ('rmse', 'Root Mean Squared Error (RMSE)', 'RMSE'),
        ('mape', 'Mean Absolute Percentage Error (MAPE)', 'MAPE')
    ]

    min_train = int(df['training_size'].min())
    max_train = int(df['training_size'].max())

    horizon_unit = "day" if args.timefreq == "1d" else "hour"
    horizon_plural = "s" if args.horizon > 1 else ""
    horizon_str = f"{args.horizon} {horizon_unit}{horizon_plural}"

    timefreq_display = "Daily" if args.timefreq == "1d" else "Hourly"

    for metric_key, metric_name, metric_abbrev in metrics:
        filename = f"{args.ticker}_{args.timefreq}_{args.exp_type}_h{args.horizon}_{metric_abbrev}_comparison.tex"
        output_path = output_dir / filename

        caption = f"Model Comparison: {metric_name} for {args.ticker} ({timefreq_display} data, Training: {min_train}-{max_train} days, Forecast Horizon: {horizon_str})"
        label = f"tab:comparison_{args.ticker}_{args.timefreq}_{args.exp_type}_h{args.horizon}_{metric_abbrev}"

        try:
            latex_table = generate_comparison_table(
                df=df,
                metric=metric_key,
                caption=caption,
                label=label
            )

            with open(output_path, 'w') as f:
                f.write(latex_table)

            logger.info(f"Generated table: {output_path}")

        except Exception as e:
            logger.error(f"Error generating {metric_abbrev} table: {e}")

    logger.info(f"All tables saved to: {output_dir}")

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    for metric_key, metric_name, _ in metrics:
        print(f"\n{metric_name}:")
        pivot = df.pivot(index='training_size', columns='model', values=metric_key)
        mean_perf = pivot.mean().sort_values()
        print(mean_perf.to_string())
        print(f"  Best overall: {mean_perf.idxmin()}")


if __name__ == "__main__":
    main()
