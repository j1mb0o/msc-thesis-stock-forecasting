import yaml
import os
import itertools
from datetime import datetime
import copy

# --- Define Parameters and Variations ---

# Define datasets you want to run experiments on
# Include all necessary info needed by the pipeline config's 'data' section
DATASETS = {
    'MSFT_1d': {
        'raw_path': 'data/MSFT/MSFT_1d.csv',
        'split': {'method': 'date', 'value': '2024-01-01'}, # Example split config
        'target_column': 'Close' # Assuming 'Close' price is the target
    },
    # Add more datasets like:
    # 'AAPL_1d': {
    #     'raw_path': 'data/AAPL/AAPL_1d.csv',
    #     'split': {'method': 'fraction', 'value': 0.8},
    #     'target_column': 'Adj Close'
    # },
}

# Define models and their potential parameter variations
MODELS = {
    'naive': [ # Naive usually has no params, so just one entry
        {'params': {}}
    ],
    'arima': [ # List different parameter sets for ARIMA
        {'params': {'order': (1,1,1), 'seasonal_order': (0,0,0,0)}},
        {'params': {'order': (5,1,0), 'seasonal_order': (0,0,0,0)}},
        {'params': {'order': (2,1,2), 'seasonal_order': (0,0,0,0)}},
        # Add more ARIMA orders or seasonal orders
    ],
    # Add other models like:
    # 'prophet': [
    #     {'params': {'growth': 'linear'}},
    #     {'params': {'growth': 'logistic', 'cap': 1000}} # Requires cap/floor in data
    # ],
}

# Define forecast strategies and their parameters
FORECAST_STRATEGIES = [
    {'strategy': 'direct'},
    {'strategy': 'rolling', 'n_steps': 1},
    {'strategy': 'rolling', 'n_steps': 5},
    # Add other rolling step sizes if needed
]

# --- Base Configuration Template ---
# Define settings common to all experiments
BASE_CONFIG = {
    'evaluation': {
        'metrics': ['mae', 'rmse', 'mape', 'smape'] # Standard metrics
    },
    'visualize': True, # Generate plot by default
    'save_model': False, # Don't save model by default
    # Add any other common settings your pipeline uses
}

# --- Output Directory for Configs ---
CONFIGS_DIR = 'configs'

# --- Main Generation Function ---
def generate_configs():
    """Generates YAML config files based on defined variations."""
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    config_count = 0
    timestamp = datetime.now().strftime("%Y%m%d") # Timestamp for grouping runs

    print(f"Generating configuration files in '{CONFIGS_DIR}'...")

    # Iterate through all combinations using itertools.product
    # Creates tuples like: (dataset_name, model_name, model_cfg, forecast_cfg)
    combinations = itertools.product(
        DATASETS.keys(),
        MODELS.keys(),
        FORECAST_STRATEGIES
    )

    for dataset_name, model_name, forecast_cfg in combinations:
        # Get the list of parameter variations for the current model
        model_param_list = MODELS[model_name]
        # Get the dataset configuration details
        dataset_cfg = DATASETS[dataset_name]

        # Iterate through each parameter set defined for the model
        for model_params_cfg in model_param_list:

            # --- Create Specific Config Dictionary ---
            config = copy.deepcopy(BASE_CONFIG) # Start with a deep copy of the base

            # 1. Add Data Info
            config['data'] = dataset_cfg

            # 2. Add Model Info
            config['model'] = {'name': model_name}
            config['model'].update(model_params_cfg) # Add specific params

            # 3. Add Forecast Strategy Info
            config['forecast'] = forecast_cfg

            # 4. Generate Unique Experiment Name and Output Directory
            # Create a descriptive suffix based on variations
            exp_suffix_parts = [model_name]
            # Add model parameters to suffix if they exist and are not empty
            if model_params_cfg.get('params'):
                 # Simple representation of params - adjust if needed for complex dicts
                 params_str = '_'.join(f"{k}-{v}" for k, v in model_params_cfg['params'].items())
                 # Sanitize param string for file names if necessary
                 params_str = params_str.replace('(', '').replace(')', '').replace(',', '_').replace(' ', '')
                 exp_suffix_parts.append(f"p-{params_str}")

            exp_suffix_parts.append(forecast_cfg['strategy'])
            if forecast_cfg['strategy'] == 'rolling':
                exp_suffix_parts.append(f"s{forecast_cfg['n_steps']}")

            exp_suffix = '_'.join(exp_suffix_parts)
            experiment_name = f"exp_{timestamp}_{dataset_name}_{exp_suffix}"

            config['experiment_name'] = experiment_name
            # Store results in a structured way
            config['output_dir'] = os.path.join('results', dataset_name, model_name, experiment_name)

            # --- Write YAML File ---
            file_name = f"{experiment_name}.yaml"
            file_path = os.path.join(CONFIGS_DIR, file_name)

            try:
                # Create nested directories for results if needed before writing config
                # os.makedirs(config['output_dir'], exist_ok=True) # Pipeline should create this

                with open(file_path, 'w') as f:
                    # Use sort_keys=False to maintain order (optional, but often nicer)
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                config_count += 1
                # print(f"Generated: {file_path}") # Uncomment for verbose output
            except Exception as e:
                print(f"ERROR: Could not write config file {file_path}: {e}")

    print(f"\nSuccessfully generated {config_count} configuration files.")

# --- Script Execution ---
if __name__ == "__main__":
    # You might need to install PyYAML: pip install pyyaml
    generate_configs()
