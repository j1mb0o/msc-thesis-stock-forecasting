import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_experiment_data(
    experiment_name: str,
    metric: str,
    base_path: Path = None,
) -> Dict[Tuple[int, str], Dict[str, List[float]]]:
    """
    Load experiment data from YAML configs.

    Args:
        experiment_name: Name of the experiment (e.g., 'train-restricted-years-pct')
        metric: Error metric to extract (e.g., 'rmse', 'mae', 'mape')
        base_path: Base path to configs directory (defaults to configs/MSFT/1d)

    Returns:
        Dictionary with structure: {(horizon, method): {training_period: [metric_values]}}
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent / "configs" / "MSFT" / "1h"
    else:
        base_path = Path(base_path).resolve()

    if not base_path.exists():
        raise ValueError(f"Base path does not exist: {base_path}")

    logger.info(f"Loading data from: {base_path}")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Metric: {metric}")

    # Valid metrics
    valid_metrics = ["mse", "mae", "rmse", "mape", "smape", "mean_directional_accuracy"]
    if metric.lower() not in valid_metrics:
        raise ValueError(f"Invalid metric '{metric}'. Must be one of: {valid_metrics}")

    data = {}

    # Iterate through method directories
    for method_dir in base_path.iterdir():
        if not method_dir.is_dir() or method_dir.name.startswith("."):
            continue

        method_name = method_dir.name
        logger.info(f"Processing method: {method_name}")

        # Look for the experiment subdirectory
        exp_dir = method_dir / experiment_name
        if not exp_dir.exists():
            logger.warning(
                f"Experiment '{experiment_name}' not found for method '{method_name}'"
            )
            continue

        # Load all YAML files in the experiment directory
        yaml_files = list(exp_dir.glob("*.yaml"))
        logger.info(f"Found {len(yaml_files)} YAML files for {method_name}")

        for yaml_file in yaml_files:
            try:
                with open(yaml_file, "r") as f:
                    config = yaml.safe_load(f)

                # Extract horizon length
                horizon = config.get("horizon_length", 1)

                # Create key as (horizon, method)
                key = (horizon, method_name)

                if key not in data:
                    data[key] = {}

                # Extract training period
                train_value = config.get("training_period_value")
                train_unit = config.get("training_period_unit", "")

                # Create a key for the training period
                if train_unit == "years":
                    train_key = f"{int(float(train_value))}y"
                elif train_unit == "days":
                    train_key = f"{int(float(train_value))}d"
                else:
                    train_key = str(train_value)

                # Extract the metric value
                metrics = config.get("evaluation_metrics", {})
                metric_value = metrics.get(metric.lower())

                if metric_value is not None:
                    if train_key not in data[key]:
                        data[key][train_key] = []
                    data[key][train_key].append(float(metric_value))
                    logger.debug(
                        f"h{horizon}-{method_name} - {train_key}: {metric} = {metric_value}"
                    )

            except Exception as e:
                logger.error(f"Error processing {yaml_file}: {e}")
                continue

    return data


def prepare_table_data(
    experiment_data: Dict[Tuple[int, str], Dict[str, List[float]]],
    training_periods: List[str] = None,
) -> Tuple[np.ndarray, List[Tuple[int, str]], List[str], List[str]]:
    """
    Prepare data for LaTeX table generation.

    Args:
        experiment_data: Data from load_experiment_data
        training_periods: Ordered list of training periods (columns)

    Returns:
        Tuple of (data_array, idx_structure, idx_names, columns)
    """
    # Sort by horizon first, then method name
    horizon_method_pairs = sorted(experiment_data.keys(), key=lambda x: (x[0], x[1]))

    # If training periods not specified, extract and sort them
    if training_periods is None:
        all_periods = set()
        for method_data in experiment_data.values():
            all_periods.update(method_data.keys())

        # Sort training periods intelligently
        def sort_key(period):
            if period.endswith("y"):
                return (1, float(period[:-1]))
            elif period.endswith("d"):
                return (0, float(period[:-1]))
            else:
                try:
                    return (2, float(period))
                except ValueError:
                    return (3, 0)

        training_periods = sorted(all_periods, key=sort_key)

    logger.info(f"Training periods (columns): {training_periods}")
    logger.info(f"Horizon-Method pairs (rows): {horizon_method_pairs}")

    # Build data matrix
    data_matrix = []
    idx_structure = []

    for horizon, method in horizon_method_pairs:
        row = []
        for period in training_periods:
            values = experiment_data.get((horizon, method), {}).get(period, [])
            if values:
                # Use mean if multiple values exist for same training period
                row.append(np.mean(values))
            else:
                # Use NaN for missing data
                row.append(np.nan)

        data_matrix.append(row)
        idx_structure.append((f"{horizon}d", method))

    return (
        np.array(data_matrix),
        idx_structure,
        ["Horizon", "Method"],
        training_periods,
    )


def generate_latex_table(
    data,
    idx_structure,
    idx_names,
    columns,
    caption="My Table",
    label="tab:mytable",
    filename="output_table.tex",
):
    """
    Generates a LaTeX table matching the specific 'Horizon | Method | training_periods' structure.
    Uses \adjustbox to fit content to \textwidth.
    """

    # 1. Construct MultiIndex for Rows (Horizon, Method)
    index = pd.MultiIndex.from_tuples(idx_structure, names=idx_names)

    # 2. Create DataFrame
    df = pd.DataFrame(data, index=index, columns=columns)

    # 3. Generate just the tabular code using Pandas
    # We set caption=None and positioning=None to get ONLY the tabular environment
    # This allows us to wrap it in \adjustbox manually below.
    latex_tabular = df.style.format(precision=4, na_rep="-").to_latex(
        position=None,  # Do not generate \begin{table} wrapper yet
        caption=None,  # Do not generate \caption yet
        label=None,  # Do not generate \label yet
        hrules=True,  # Adds \toprule, \midrule, \bottomrule
        multirow_align="c",  # Center vertical alignment for merged cells
        multicol_align="c",  # Center horizontal alignment
        column_format="c"
        * (len(idx_names) + len(columns)),  # Force 'c' for all columns
    )

    # 4. Manually construct the Table Wrapper with \adjustbox
    # This gives us exact control over the layout requested.
    full_latex = f"""\\begin{{table}}[h!]
  \\centering
  \\caption{{{caption}}}
  \\label{{{label}}}
  \\adjustbox{{max width=\\textwidth}}{{
{latex_tabular}
 }}
\\end{{table}}"""

    # 5. Save to file
    with open(filename, "w") as f:
        f.write(full_latex)

    logger.info(f"Successfully generated table: {filename}")
    print("-" * 60)
    print(full_latex)
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from experiment YAML configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate RMSE table for train-restricted-years-pct experiment
  python scripts/generate_tables_v2.py \\
    --experiment train-restricted-years-pct \\
    --metric rmse

  # Generate MAE table with custom output file
  python scripts/generate_tables_v2.py \\
    --experiment train-less-year-linear \\
    --metric mae \\
    --output tables/mae_linear.tex

  # Specify custom configs directory
  python scripts/generate_tables_v2.py \\
    --experiment train-restricted-years \\
    --metric mape \\
    --base-path /path/to/configs/MSFT/1d
        """,
    )

    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        required=True,
        help="Experiment name (e.g., 'train-restricted-years-pct', 'train-less-year-linear')",
    )

    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        required=True,
        choices=["mse", "mae", "rmse", "mape", "smape", "mean_directional_accuracy"],
        help="Error metric to extract from experiment results",
    )

    parser.add_argument(
        "--base-path",
        "-b",
        type=str,
        default=None,
        help="Base path to configs directory (default: configs/MSFT/1d relative to script)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output LaTeX file path (default: {experiment}_{metric}_table.tex)",
    )

    parser.add_argument(
        "--caption",
        "-c",
        type=str,
        default=None,
        help="LaTeX table caption (default: auto-generated from experiment and metric)",
    )

    parser.add_argument(
        "--label",
        "-l",
        type=str,
        default=None,
        help="LaTeX table label (default: tab:{experiment}_{metric})",
    )

    args = parser.parse_args()

    # Set defaults
    if args.output is None:
        # Create output directory structure: tables/1d/{experiment_name}/
        output_dir = Path(__file__).parent.parent / "tables" / "1h" / args.experiment
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"{args.experiment}_{args.metric}_table.tex")

    if args.caption is None:
        metric_display = args.metric.upper()
        exp_display = args.experiment.replace("-", " ").title()
        args.caption = f"{metric_display} - {exp_display}"

    if args.label is None:
        args.label = f"tab:{args.experiment}_{args.metric}"

    # Load data
    try:
        experiment_data = load_experiment_data(
            experiment_name=args.experiment,
            metric=args.metric,
            base_path=args.base_path,
        )

        if not experiment_data:
            logger.error("No data loaded. Check experiment name and base path.")
            return 1

        # Prepare table data
        data, idx_structure, idx_names, columns = prepare_table_data(experiment_data)

        # Generate LaTeX table
        generate_latex_table(
            data=data,
            idx_structure=idx_structure,
            idx_names=idx_names,
            columns=columns,
            caption=args.caption,
            label=args.label,
            filename=args.output,
        )

        logger.info("Table generation completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error generating table: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
