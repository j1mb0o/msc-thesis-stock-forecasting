# Market Crash Analysis - Plot Explanation

## Plot Explanation:

### Top Plot (Price with Crash Periods):
- Black line shows MSFT hourly price from 2018 onwards
- Colored vertical bands automatically highlight ALL periods where drawdown exceeded -10%
- **Color coding by severity:**
  - **Orange** = Moderate crash (10-20% drawdown)
  - **Red** = Major crash (20-30% drawdown)
  - **Dark Red** = Severe crash (>30% drawdown)
- Labels show the max drawdown % for major crashes (>20%)

### Bottom Plot (Drawdown):
- Shows percentage decline from the running maximum (peak)
  - **0%** = Stock is at all-time high
  - **-10%** = Stock is 10% below its previous peak
  - **-20%** = Stock is 20% below its previous peak
- Red shaded area = any drawdown (below peak)
- Dashed lines show severity thresholds
- Same colored bands as top plot highlight crash periods

## Key Crashes Visible in 2018-2024:

From the data, here are the significant crashes highlighted:

1. **Dec 2018 Correction** (-19.03%, Dec 19 - Jan 9)
2. **COVID-19 Crash** (-28.61%, Mar 5 - Apr 13, 2020) - RED band
3. **Jan-Feb 2022 Selloff** (-18.89%, Jan 19 - Feb 1)
4. **Ukraine War Period** (-21.48%, Feb 3 - Mar 29, 2022) - RED band
5. **2022-2023 Bear Market** (-38.44%, Apr 5, 2022 - May 16, 2023) - DARK RED band

## Usage for RQ3 Analysis:

These crash periods are data-driven (based on drawdown thresholds) rather than predefined dates. This allows for objective identification of market disruptions for testing:

- **H6:** Foundation models' performance degradation during crisis periods compared to traditional approaches
- **H7:** Consistency of model rankings across normal and crisis periods

## Script Usage:

```bash
# Generate the crash analysis plot
source base_env/bin/activate
python scripts/identify_market_crashes.py --ticker MSFT --threshold -10.0 --start-year 2018 --top-n 15

# Adjust parameters:
# --threshold: Change drawdown threshold (e.g., -15.0 for more severe crashes only)
# --start-year: Change starting year for visualization
# --top-n: Number of crashes to show in console output
```

## Generated Files:

- **Plot:** `figures/market_crashes_hourly.png`
- **Script:** `scripts/identify_market_crashes.py`
