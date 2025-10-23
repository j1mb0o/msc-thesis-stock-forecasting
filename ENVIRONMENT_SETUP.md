# Environment Setup Guide

This project requires separate virtual environments for different models due to conflicting transformers versions:
- **Chronos**: requires `transformers>=4.48.0`
- **Sundial**: requires `transformers==4.40.1`
- **ARIMA/Naive/TimesFM**: can use base environment (no transformers needed)

## Environment Setup

### 1. Base Environment (for ARIMA, Naive)

```bash
# Create environment
uv venv -p python3.11 base_env

# Activate
source base_env/bin/activate

# Install base dependencies
uv pip install -e ".[dev]"

# Verify installation
python -c "import pmdarima; import pandas; print('Base environment ready')"

# Deactivate when done
deactivate
```

### 2. Chronos Environment

```bash
# Create environment
uv venv -p python3.11 chronos_env

# Activate
source chronos_env/bin/activate

# Install with Chronos dependencies
uv pip install -e ".[dev,chronos]"

# Verify transformers version
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
# Should show: 4.48.0 or higher

# Deactivate when done
deactivate
```

### 3. Sundial Environment

```bash
# Create environment
uv venv -p python3.11 sundial_env

# Activate
source sundial_env/bin/activate

# Install base dependencies first
uv pip install -e ".[dev]"

# Install Sundial's specific transformers version
uv pip install transformers==4.40.1

# Verify transformers version
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
# Should show: 4.40.1

# Deactivate when done
deactivate
```

### 4. TimesFM Environment (Linux only)

```bash
# Create environment
uv venv -p python3.11 timesfm_env

# Activate
source timesfm_env/bin/activate

# Install with TimesFM dependencies
uv pip install -e ".[dev,times]"

# Deactivate when done
deactivate
```

## Running Experiments

### Single Experiments

```bash
# For Chronos
source chronos_env/bin/activate
python scripts/pipeline.py --method chronos_base --ticker MSFT --timefreq 1d --horizon_len 1
deactivate

# For Sundial
source sundial_env/bin/activate
python scripts/pipeline.py --method sundial --ticker MSFT --timefreq 1d --horizon_len 1
deactivate

# For ARIMA (use base_env)
source base_env/bin/activate
python scripts/pipeline.py --method arima --ticker MSFT --timefreq 1d --horizon_len 1
deactivate

# For Naive (use base_env)
source base_env/bin/activate
python scripts/pipeline.py --method naive --ticker MSFT --timefreq 1d --horizon_len 1
deactivate
```

### Batch Experiments

**Important:** Activate the correct environment BEFORE running batch experiments:

```bash
# For Chronos batch experiments
source chronos_env/bin/activate
./run_experiments.sh chronos_base
deactivate

# For Sundial batch experiments
source sundial_env/bin/activate
./run_experiments.sh sundial
deactivate

# For ARIMA batch experiments
source base_env/bin/activate
./run_experiments.sh arima
deactivate

# For Naive batch experiments
source base_env/bin/activate
./run_experiments.sh naive
deactivate
```

## Troubleshooting

### Check which environment is active
```bash
which python
# Should show path containing chronos_env, sundial_env, or base_env
```

### Verify transformers version
```bash
python -c "import transformers; print(transformers.__version__)"
```

### ImportError: No module named 'transformers'
Make sure you're in the correct environment and have run the installation commands.

### Wrong transformers version
Deactivate and activate the correct environment:
```bash
deactivate
source chronos_env/bin/activate  # or sundial_env/bin/activate
```

## Environment Reference

| Model | Environment | Transformers Version | Command |
|-------|-------------|---------------------|---------|
| Chronos | `chronos_env` | >=4.48.0 | `source chronos_env/bin/activate` |
| Sundial | `sundial_env` | ==4.40.1 | `source sundial_env/bin/activate` |
| ARIMA | `base_env` | N/A | `source base_env/bin/activate` |
| Naive | `base_env` | N/A | `source base_env/bin/activate` |
| TimesFM | `timesfm_env` | N/A | `source timesfm_env/bin/activate` |
