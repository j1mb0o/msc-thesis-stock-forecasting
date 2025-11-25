#!/bin/bash

# Shell script to generate all LaTeX tables from experiment results
# Iterates through experiment names and error metrics

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Define experiment names
EXPERIMENTS=(
    "train-restricted-years"
    "train-restricted-years-pct"
    "train-less-year-linear"
    "train-less-year-linear-pct"
    "train-less-year-log"
    "train-less-year-log-pct"
)

# Define error metrics
METRICS=(
    "rmse"
    "mae"
    "mse"
    "mape"
    "smape"
    "mean_directional_accuracy"
)

# Base directory for configs (can be overridden with -b flag)
BASE_PATH="configs/MSFT/1d"

# Parse command line arguments
SPECIFIC_EXPERIMENT=""
SPECIFIC_METRIC=""

while getopts "e:m:b:h" opt; do
    case $opt in
        e)
            SPECIFIC_EXPERIMENT="$OPTARG"
            ;;
        m)
            SPECIFIC_METRIC="$OPTARG"
            ;;
        b)
            BASE_PATH="$OPTARG"
            ;;
        h)
            echo "Usage: $0 [-e experiment] [-m metric] [-b base_path]"
            echo ""
            echo "Options:"
            echo "  -e    Generate tables for a specific experiment only"
            echo "  -m    Generate tables for a specific metric only"
            echo "  -b    Base path to configs directory (default: configs/MSFT/1d)"
            echo "  -h    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Generate all tables"
            echo "  $0 -e train-restricted-years          # Generate all metrics for one experiment"
            echo "  $0 -m rmse                            # Generate RMSE tables for all experiments"
            echo "  $0 -e train-restricted-years -m rmse  # Generate one specific table"
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

# Filter experiments and metrics if specified
if [ -n "$SPECIFIC_EXPERIMENT" ]; then
    EXPERIMENTS=("$SPECIFIC_EXPERIMENT")
fi

if [ -n "$SPECIFIC_METRIC" ]; then
    METRICS=("$SPECIFIC_METRIC")
fi

# Counter for tracking progress
TOTAL_TABLES=$((${#EXPERIMENTS[@]} * ${#METRICS[@]}))
CURRENT=0
SUCCESS=0
FAILED=0

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Table Generation${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Total tables to generate: $TOTAL_TABLES"
echo "Base path: $BASE_PATH"
echo ""

# Iterate through all combinations
for experiment in "${EXPERIMENTS[@]}"; do
    echo -e "${YELLOW}Processing experiment: $experiment${NC}"

    for metric in "${METRICS[@]}"; do
        CURRENT=$((CURRENT + 1))

        echo -e "  [$CURRENT/$TOTAL_TABLES] Generating ${metric} table..."

        # Run the Python script
        if python scripts/generate_tables_v2.py \
            --experiment "$experiment" \
            --metric "$metric" \
            --base-path "$BASE_PATH" 2>&1 | grep -q "Successfully generated"; then

            SUCCESS=$((SUCCESS + 1))
            echo -e "    ${GREEN}✓${NC} Success"
        else
            FAILED=$((FAILED + 1))
            echo -e "    ${RED}✗${NC} Failed"
        fi
    done

    echo ""
done

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Table Generation Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Total:   $TOTAL_TABLES"
echo -e "${GREEN}Success: $SUCCESS${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed:  $FAILED${NC}"
else
    echo -e "Failed:  $FAILED"
fi
echo ""

# Show output directory
OUTPUT_DIR="tables/1d"
if [ -d "$OUTPUT_DIR" ]; then
    echo "Generated tables saved to: $OUTPUT_DIR/"
    echo ""
    echo "Directory structure:"
    tree "$OUTPUT_DIR" 2>/dev/null || find "$OUTPUT_DIR" -type f -name "*.tex" | head -10
fi

exit 0
