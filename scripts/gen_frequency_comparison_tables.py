#!/usr/bin/env python3
"""
Generate frequency comparison tables (1d vs 1h) from experiment results.

This script compares model performance between daily (1d) and hourly (1h) frequencies
by aggregating hourly forecasts to daily level and computing metrics on the same dates.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple
import yaml
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_aggregate_predictions(
    ticker: str,
    timefreq: str,
    model: str,
    experiment_type: str,
    training_size: int,
    horizon: int = 1
) -> Tuple[pd.Series, pd.Series]:
    """
    Load predictions from results CSV files and aggregate if needed.

    For 1h data: Aggregates to daily level by taking the last hourly value of each trading day.
    For 1d data: Returns as-is.

    Args:
        ticker: Stock ticker (e.g., 'MSFT')
        timefreq: Time frequency ('1h' or '1d')
        model: Model name
        experiment_type: Experiment type
        training_size: Training size in days
        horizon: Forecast horizon

    Returns:
        Tuple of (actual_series, forecast_series) indexed by date
    """
    results_dir = Path("results") / ticker / timefreq / model / experiment_type

    if not results_dir.exists():
        logger.warning(f"Results directory does not exist: {results_dir}")
        return None, None

    pattern = f"*_train_{training_size}d_test_*_horizon_{horizon}.csv"
    csv_files = list(results_dir.glob(pattern))

    if not csv_files:
        logger.warning(f"No results found matching: {results_dir / pattern}")
        return None, None

    if len(csv_files) > 1:
        logger.warning(f"Multiple results found, using first: {csv_files[0]}")

    csv_file = csv_files[0]

    try:
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])

        forecast_col = [col for col in df.columns if 'Forecast' in col][0]

        if timefreq == '1h':
            # Aggregate hourly bars to daily by taking the last bar of each trading day.
            df['DateOnly'] = df['Date'].dt.date
            daily_df = df.groupby('DateOnly').last().reset_index()
            daily_df['DateOnly'] = pd.to_datetime(daily_df['DateOnly'])

            actual = daily_df.set_index('DateOnly')['Actual']
            forecast = daily_df.set_index('DateOnly')[forecast_col]
        else:
            df['Date'] = df['Date'].dt.date
            df['Date'] = pd.to_datetime(df['Date'])
            actual = df.set_index('Date')['Actual']
            forecast = df.set_index('Date')[forecast_col]

        return actual, forecast

    except Exception as e:
        logger.error(f"Error loading {csv_file}: {e}")
        return None, None


def calculate_metrics(actual: pd.Series, forecast: pd.Series) -> Dict[str, float]:
    """
    Calculate evaluation metrics on aligned actual and forecast values.

    Args:
        actual: Actual values
        forecast: Forecast values

    Returns:
        Dictionary of metrics
    """
    aligned = pd.DataFrame({'actual': actual, 'forecast': forecast}).dropna()

    if len(aligned) == 0:
        return {}

    actual_vals = aligned['actual'].values
    forecast_vals = aligned['forecast'].values

    mae = np.mean(np.abs(actual_vals - forecast_vals))
    mse = np.mean((actual_vals - forecast_vals) ** 2)
    rmse = np.sqrt(mse)

    non_zero_mask = actual_vals != 0
    if non_zero_mask.any():
        mape = np.mean(np.abs((actual_vals[non_zero_mask] - forecast_vals[non_zero_mask]) / actual_vals[non_zero_mask])) * 100
    else:
        mape = np.nan

    denominator = (np.abs(actual_vals) + np.abs(forecast_vals)) / 2
    non_zero_denom = denominator != 0
    if non_zero_denom.any():
        smape = np.mean(np.abs(actual_vals[non_zero_denom] - forecast_vals[non_zero_denom]) / denominator[non_zero_denom]) * 100
    else:
        smape = np.nan

    if len(actual_vals) > 1:
        actual_direction = np.diff(actual_vals) > 0
        forecast_direction = np.diff(forecast_vals) > 0
        mda = np.mean(actual_direction == forecast_direction) * 100
    else:
        mda = np.nan

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'smape': smape,
        'mda': mda,
        'n_points': len(actual_vals)
    }


def load_experiment_metrics_for_frequency(
    ticker: str,
    timefreq: str,
    model: str,
    experiment_type: str,
    horizon: int = 1
) -> pd.DataFrame:
    """
    Load metrics for a specific model, frequency, and experiment type.

    This function now loads the raw predictions and computes metrics after aggregation
    to ensure fair comparison between 1d and 1h frequencies.

    Args:
        ticker: Stock ticker (e.g., 'MSFT')
        timefreq: Time frequency ('1h' or '1d')
        model: Model name
        experiment_type: Experiment type
        horizon: Forecast horizon (default: 1)

    Returns:
        DataFrame with columns: training_size, mse, mae, rmse, mape, smape, mda
    """
    base_path = Path("configs") / ticker / timefreq / model / experiment_type

    if not base_path.exists():
        logger.warning(f"Path does not exist: {base_path}")
        return pd.DataFrame()

    all_data = []

    config_files = list(base_path.glob(f"*_horizon_{horizon}.yaml"))

    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            training_size = config.get('training_period_value')
            training_unit = config.get('training_period_unit', 'days')

            if training_unit == 'years':
                training_size = training_size * 365

            actual, forecast = load_and_aggregate_predictions(
                ticker=ticker,
                timefreq=timefreq,
                model=model,
                experiment_type=experiment_type,
                training_size=training_size,
                horizon=horizon
            )

            if actual is not None and forecast is not None:
                metrics = calculate_metrics(actual, forecast)

                if metrics:
                    all_data.append({
                        'training_size': training_size,
                        'mse': metrics.get('mse'),
                        'mae': metrics.get('mae'),
                        'rmse': metrics.get('rmse'),
                        'mape': metrics.get('mape'),
                        'smape': metrics.get('smape'),
                        'mda': metrics.get('mda'),
                        'n_points': metrics.get('n_points')
                    })

        except Exception as e:
            logger.error(f"Error processing {config_file}: {e}")
            continue

    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.sort_values('training_size')
    return df


def generate_frequency_comparison_table(
    df_1d: pd.DataFrame,
    df_1h: pd.DataFrame,
    metric: str,
    model: str,
    caption: str,
    label: str
) -> str:
    """
    Generate LaTeX comparison table showing 1d vs 1h performance.

    Args:
        df_1d: DataFrame with 1d metrics
        df_1h: DataFrame with 1h metrics
        metric: Metric to display
        model: Model name
        caption: Table caption
        label: LaTeX label

    Returns:
        LaTeX table string
    """
    merged = pd.merge(
        df_1d[['training_size', metric]],
        df_1h[['training_size', metric]],
        on='training_size',
        how='inner',
        suffixes=('_1d', '_1h')
    )

    if merged.empty:
        logger.warning(f"No common experiments found for {model}")
        return ""

    # Negative pct_change means 1h aggregation improved over native 1d.
    merged['pct_change'] = ((merged[f'{metric}_1h'] - merged[f'{metric}_1d']) / merged[f'{metric}_1d']) * 100

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append("\\small")

    col_spec = "lrrrr"
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")

    metric_upper = metric.upper()
    header = f"Training Days & {metric_upper} (1d) & {metric_upper} (1h) & \\% Change & Better \\\\"
    latex.append(header)
    latex.append("\\midrule")

    for _, row in merged.iterrows():
        train_size = int(row['training_size'])
        val_1d = row[f'{metric}_1d']
        val_1h = row[f'{metric}_1h']
        pct_change = row['pct_change']

        if val_1h < val_1d:
            better = "1h"
            val_1h_str = f"\\textbf{{{val_1h:.4f}}}"
            val_1d_str = f"{val_1d:.4f}"
        elif val_1d < val_1h:
            better = "1d"
            val_1d_str = f"\\textbf{{{val_1d:.4f}}}"
            val_1h_str = f"{val_1h:.4f}"
        else:
            better = "Tie"
            val_1d_str = f"{val_1d:.4f}"
            val_1h_str = f"{val_1h:.4f}"

        if pct_change < 0:
            pct_str = f"\\textcolor{{green}}{{{pct_change:.2f}\\%}}"
        elif pct_change > 0:
            pct_str = f"\\textcolor{{red}}{{+{pct_change:.2f}\\%}}"
        else:
            pct_str = "0.00\\%"

        row_str = f"{train_size} & {val_1d_str} & {val_1h_str} & {pct_str} & {better} \\\\"
        latex.append(row_str)

    latex.append("\\midrule")

    mean_1d = merged[f'{metric}_1d'].mean()
    mean_1h = merged[f'{metric}_1h'].mean()
    mean_pct_change = ((mean_1h - mean_1d) / mean_1d) * 100

    if mean_1h < mean_1d:
        mean_1d_str = f"{mean_1d:.4f}"
        mean_1h_str = f"\\textbf{{{mean_1h:.4f}}}"
        mean_better = "1h"
    elif mean_1d < mean_1h:
        mean_1d_str = f"\\textbf{{{mean_1d:.4f}}}"
        mean_1h_str = f"{mean_1h:.4f}"
        mean_better = "1d"
    else:
        mean_1d_str = f"{mean_1d:.4f}"
        mean_1h_str = f"{mean_1h:.4f}"
        mean_better = "Tie"

    if mean_pct_change < 0:
        mean_pct_str = f"\\textcolor{{green}}{{{mean_pct_change:.2f}\\%}}"
    elif mean_pct_change > 0:
        mean_pct_str = f"\\textcolor{{red}}{{+{mean_pct_change:.2f}\\%}}"
    else:
        mean_pct_str = "0.00\\%"

    mean_row = f"\\textit{{Mean}} & {mean_1d_str} & {mean_1h_str} & {mean_pct_str} & {mean_better} \\\\"
    latex.append(mean_row)

    count_1d = (merged[f'{metric}_1d'] < merged[f'{metric}_1h']).sum()
    count_1h = (merged[f'{metric}_1h'] < merged[f'{metric}_1d']).sum()

    count_row = f"\\textit{{\\# Better}} & {count_1d} & {count_1h} & -- & -- \\\\"
    latex.append(count_row)

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(
        description="Generate frequency comparison tables (1d vs 1h with proper aggregation)"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="MSFT",
        help="Stock ticker (default: MSFT)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["arima", "chronos_base", "sundial", "fm", "naive"],
        help="Model to compare"
    )
    parser.add_argument(
        "--exp-type",
        type=str,
        default="train-less-year-log",
        choices=["train-less-year-log", "train-less-year-linear", "train-restricted-years",
                 "train-less-year-log-pct", "train-less-year-linear-pct", "train-restricted-years-pct"],
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
        default="tables/frequency_comparisons",
        help="Output directory for LaTeX tables (default: tables/frequency_comparisons)"
    )

    args = parser.parse_args()

    logger.info(f"Loading experiment data for {args.model} ({args.exp_type}, h={args.horizon})")
    logger.info("NOTE: 1h predictions will be aggregated to daily level for fair comparison")

    df_1d = load_experiment_metrics_for_frequency(
        ticker=args.ticker,
        timefreq="1d",
        model=args.model,
        experiment_type=args.exp_type,
        horizon=args.horizon
    )

    df_1h = load_experiment_metrics_for_frequency(
        ticker=args.ticker,
        timefreq="1h",
        model=args.model,
        experiment_type=args.exp_type,
        horizon=args.horizon
    )

    if df_1d.empty:
        logger.error(f"No 1d data loaded for {args.model}!")
        return

    if df_1h.empty:
        logger.error(f"No 1h data loaded for {args.model}!")
        return

    logger.info(f"Loaded {len(df_1d)} 1d experiments and {len(df_1h)} 1h experiments")

    common_sizes = set(df_1d['training_size']) & set(df_1h['training_size'])
    logger.info(f"Common training sizes: {sorted(common_sizes)}")

    if not common_sizes:
        logger.error("No common experiments found between 1d and 1h!")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = {
        'arima': 'ARIMA',
        'chronos_base': 'Chronos',
        'sundial': 'Sundial',
        'fm': 'TimesFM',
        'naive': 'Naive'
    }
    model_display = model_names.get(args.model, args.model)

    metrics = [
        ('mse', 'Mean Squared Error (MSE)', 'MSE'),
        ('mae', 'Mean Absolute Error (MAE)', 'MAE'),
        ('rmse', 'Root Mean Squared Error (RMSE)', 'RMSE'),
        ('mape', 'Mean Absolute Percentage Error (MAPE)', 'MAPE')
    ]

    horizon_str = f"{args.horizon}"

    for metric_key, metric_name, metric_abbrev in metrics:
        filename = f"{args.ticker}_{args.model}_{args.exp_type}_h{args.horizon}_{metric_abbrev}_freq_comparison.tex"
        output_path = output_dir / filename

        exp_name_display = args.exp_type.replace('-', ' ').title()
        caption = f"Frequency Comparison ({model_display}): {metric_name} - Daily vs Hourly Data (Aggregated) (Training: {exp_name_display}, Horizon: {horizon_str})"
        label = f"tab:freq_comp_{args.ticker}_{args.model}_{args.exp_type}_h{args.horizon}_{metric_abbrev}"

        try:
            latex_table = generate_frequency_comparison_table(
                df_1d=df_1d,
                df_1h=df_1h,
                metric=metric_key,
                model=args.model,
                caption=caption,
                label=label
            )

            if latex_table:
                with open(output_path, 'w') as f:
                    f.write(latex_table)
                logger.info(f"Generated table: {output_path}")
            else:
                logger.warning(f"Skipped {metric_abbrev} table (no data)")

        except Exception as e:
            logger.error(f"Error generating {metric_abbrev} table: {e}")

    logger.info(f"All tables saved to: {output_dir}")

    print("\n" + "="*60)
    print(f"FREQUENCY COMPARISON SUMMARY - {model_display}")
    print("="*60)

    for metric_key, metric_name, _ in metrics:
        print(f"\n{metric_name}:")

        merged = pd.merge(
            df_1d[['training_size', metric_key]],
            df_1h[['training_size', metric_key]],
            on='training_size',
            how='inner',
            suffixes=('_1d', '_1h')
        )

        if not merged.empty:
            mean_1d = merged[f'{metric_key}_1d'].mean()
            mean_1h = merged[f'{metric_key}_1h'].mean()
            pct_change = ((mean_1h - mean_1d) / mean_1d) * 100

            print(f"  Average 1d: {mean_1d:.4f}")
            print(f"  Average 1h (aggregated): {mean_1h:.4f}")
            print(f"  % Change: {pct_change:+.2f}%")

            count_1d_better = (merged[f'{metric_key}_1d'] < merged[f'{metric_key}_1h']).sum()
            count_1h_better = (merged[f'{metric_key}_1h'] < merged[f'{metric_key}_1d']).sum()

            print(f"  1d better: {count_1d_better}/{len(merged)} times")
            print(f"  1h better: {count_1h_better}/{len(merged)} times")

            if mean_1h < mean_1d:
                print(f"  Overall winner: 1h (hourly data aggregated to daily)")
            elif mean_1d < mean_1h:
                print(f"  Overall winner: 1d (daily data)")
            else:
                print(f"  Overall: Tie")


if __name__ == "__main__":
    main()
