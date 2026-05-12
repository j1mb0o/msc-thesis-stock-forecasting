#!/usr/bin/env python3
"""
Generate all frequency comparison tables (1d vs 1h) for all models and experiment types.

This script iterates through all available models and experiment types to generate
comprehensive frequency comparison tables.
"""

import argparse
import logging
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate all frequency comparison tables"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="MSFT",
        help="Stock ticker (default: MSFT)"
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
        help="Output directory for LaTeX tables"
    )

    args = parser.parse_args()

    models = ["arima", "chronos_base", "sundial", "fm", "naive"]

    base_exp_types = [
        "train-less-year-log",
        "train-less-year-linear",
        "train-restricted-years"
    ]

    exp_types = []
    for exp in base_exp_types:
        exp_types.append(exp)
        exp_types.append(f"{exp}-pct")

    total_combinations = len(models) * len(exp_types)
    current = 0

    logger.info(f"Generating frequency comparison tables for:")
    logger.info(f"  Models: {', '.join(models)}")
    logger.info(f"  Experiment types: {', '.join(exp_types)}")
    logger.info(f"  Horizon: {args.horizon}")
    logger.info(f"  Total combinations: {total_combinations}")
    logger.info("")

    successful = []
    failed = []
    skipped = []

    for model in models:
        for exp_type in exp_types:
            current += 1
            logger.info(f"\n[{current}/{total_combinations}] Processing: {model} - {exp_type}")

            config_path_1d = Path("configs") / args.ticker / "1d" / model / exp_type
            config_path_1h = Path("configs") / args.ticker / "1h" / model / exp_type

            if not config_path_1d.exists():
                logger.warning(f"  Skipping: 1d configs not found at {config_path_1d}")
                skipped.append(f"{model}/{exp_type} (no 1d data)")
                continue

            if not config_path_1h.exists():
                logger.warning(f"  Skipping: 1h configs not found at {config_path_1h}")
                skipped.append(f"{model}/{exp_type} (no 1h data)")
                continue

            cmd = [
                "python3", "scripts/gen_frequency_comparison_tables.py",
                "--ticker", args.ticker,
                "--model", model,
                "--exp-type", exp_type,
                "--horizon", str(args.horizon),
                "--output-dir", args.output_dir
            ]

            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info(f"  Success: Generated tables for {model}/{exp_type}")
                successful.append(f"{model}/{exp_type}")

                if "FREQUENCY COMPARISON SUMMARY" in result.stdout:
                    summary_lines = result.stdout.split("FREQUENCY COMPARISON SUMMARY")[1].split("\n")[:10]
                    for line in summary_lines[:5]:
                        if line.strip():
                            logger.info(f"    {line.strip()}")

            except subprocess.CalledProcessError as e:
                logger.error(f"  Failed: {model}/{exp_type}")
                logger.error(f"    Error: {e.stderr[:200]}")
                failed.append(f"{model}/{exp_type}")

    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80)
    print(f"\nTotal combinations: {total_combinations}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Skipped: {len(skipped)}")

    if successful:
        print(f"\nSuccessful generations:")
        for item in successful:
            print(f"  ✓ {item}")

    if failed:
        print(f"\nFailed generations:")
        for item in failed:
            print(f"  ✗ {item}")

    if skipped:
        print(f"\nSkipped (missing data):")
        for item in skipped:
            print(f"  - {item}")

    print(f"\nAll tables saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
