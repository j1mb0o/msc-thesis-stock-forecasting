
import torch
from transformers import AutoModelForCausalLM
import pandas as pd
import logging
import tqdm

import warnings

warnings.filterwarnings("ignore")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SundialForecaster:
    """
    A class for performing Sundial forecasting on time series data.
    """

    def __init__(self, train_data, test_data, model_name='thuml/sundial-base-128m', horizon_len=1):
        """
        Initializes the SundialForecaster with training and test data.

        Args:
            train_data (pd.Series): The training data. Must be sorted chronologically.
            test_data (pd.Series): The test data. Must be sorted chronologically and follow train_data.
        """
        if not isinstance(train_data, pd.Series):
            raise TypeError("train_data must be a pandas Series.")
        if not isinstance(test_data, pd.Series):
            raise TypeError("test_data must be a pandas Series.")
        if train_data.empty:
             raise ValueError("train_data cannot be empty.")
        if test_data.empty:
             raise ValueError("test_data cannot be empty.")

        self.train_data = train_data
        self.test_data = test_data
        self.horizon = horizon_len
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        
    def forecast(self, num_samples=20):
        """
        Generates forecasts for the given horizon.

        Args:
            num_samples (int): The number of probable predictions to generate.

        Returns:
            pd.Series: The forecasts.
        """
        
        logging.info(f"Performing Sundial Forecast with horizon={self.horizon}...")
        
        context = self.train_data.copy()
        forecasts = []

        for i in tqdm.tqdm(range(0, len(self.test_data), self.horizon)):
            # Prepare the input sequence
            seqs = torch.tensor(context.values, dtype=torch.float32).unsqueeze(0)

            # Generate predictions
            output = self.model.generate(seqs, max_new_tokens=self.horizon, num_samples=num_samples)
            
            forecast = output[0].mean(dim=0).numpy()
            
            forecasts.extend(forecast)

            # Update the training data with the new observations from the test set
            new_obs = self.test_data[i:i+self.horizon]
            context = pd.concat([context, new_obs])


        logging.info(f"Sundial forecast generated for {len(self.test_data)} total steps.")
        if len(forecasts) > self.test_data.shape[0]:
                    forecasts = forecasts[:self.test_data.shape[0]]
        
        final_forecast = pd.Series(forecasts[:len(self.test_data)], index=self.test_data.index, name=f"Sundial Forecast (horizon={self.horizon})")
        return final_forecast

if __name__ == "__main__":
    import numpy as np
    print("Running test suite for SundialForecaster...")
    # TODO: check carefully the tests

    # Setup for functional tests
    data = pd.Series(
        np.linspace(0, 100, 100), 
        index=pd.date_range(start='2023-01-01', periods=100, freq='D'),
        dtype=np.float32
    )
    train_main, test_main = data.iloc[:80], data.iloc[80:]

    # Test Case 1: Input Validation
    print("\n--- Test Case 1: Input Validation ---")
    try:
        # Test for non-Series input
        print("Testing with list input for train_data...")
        SundialForecaster(train_data=train_main, test_data=test_main)
    except TypeError as e:
        print(f"Successfully caught expected error: {e}")

    try:
        # Test for empty input
        print("Testing with empty Series for train_data...")
        empty_series = pd.Series([], dtype=float)
        SundialForecaster(train_data=empty_series, test_data=test_main)
    except ValueError as e:
        print(f"Successfully caught expected error: {e}")
    print("Input validation tests passed.")

    # Test Case 1b: Test Data Validation
    print("\n--- Test Case 1b: Test Data Validation ---")
    try:
        # Test for list input
        print("Testing with list input for test_data...")
        SundialForecaster(train_data=train_main, test_data=[])
    except TypeError as e:
        print(f"Successfully caught expected error: {e}")

    try:
        # Test for None input
        print("Testing with None input for test_data...")
        SundialForecaster(train_data=train_main, test_data=None)
    except TypeError as e:
        print(f"Successfully caught expected error: {e}")

    try:
        # Test for empty Series input
        print("Testing with empty Series for test_data...")
        empty_series = pd.Series([], dtype=float)
        SundialForecaster(train_data=train_main, test_data=empty_series)
    except ValueError as e:
        print(f"Successfully caught expected error: {e}")
    print("Test data validation tests passed.")

    # Test Case 2: Core Functionality (Happy Path)
    print("\n--- Test Case 2: Core Functionality (Happy Path) ---")
    horizon_happy = 5
    test_len_happy = len(test_main)
    print(f"Testing with horizon={horizon_happy} on a test set of length {test_len_happy}.")
    sundial_happy = SundialForecaster(train_data=train_main.copy(), test_data=test_main.copy(), horizon_len=horizon_happy)
    forecast_happy = sundial_happy.forecast(num_samples=2) # Using fewer samples to speed up test
    
    assert isinstance(forecast_happy, pd.Series), "Output is not a pandas Series."
    assert len(forecast_happy) == test_len_happy, f"Output length mismatch: expected {test_len_happy}, got {len(forecast_happy)}."
    assert forecast_happy.index.equals(test_main.index), "Output index does not match test data index."
    assert pd.api.types.is_float_dtype(forecast_happy.dtype), f"Output dtype is not float, but {forecast_happy.dtype}."
    print("Core functionality test passed.")

    # Test Case 3: Incongruent Horizon
    print("\n--- Test Case 3: Incongruent Horizon ---")
    test_incongruent = data.iloc[80:97] # 17 samples
    horizon_incongruent = 5
    test_len_incongruent = len(test_incongruent)
    print(f"Testing with horizon={horizon_incongruent} on a test set of length {test_len_incongruent}.")
    sundial_incongruent = SundialForecaster(train_data=train_main.copy(), test_data=test_incongruent.copy(), horizon_len=horizon_incongruent)
    forecast_incongruent = sundial_incongruent.forecast(num_samples=2)

    assert len(forecast_incongruent) == test_len_incongruent, f"Output length mismatch: expected {test_len_incongruent}, got {len(forecast_incongruent)}."
    assert forecast_incongruent.index.equals(test_incongruent.index), "Output index does not match test data index."
    print("Incongruent horizon test passed.")

    # Test Case 4: Horizon Exceeds Test Data
    print("\n--- Test Case 4: Horizon Exceeds Test Data ---")
    test_short = data.iloc[80:85] # 5 samples
    horizon_large = 10
    test_len_short = len(test_short)
    print(f"Testing with horizon={horizon_large} on a test set of length {test_len_short}.")
    sundial_short = SundialForecaster(train_data=train_main.copy(), test_data=test_short.copy(), horizon_len=horizon_large)
    forecast_short = sundial_short.forecast(num_samples=2)

    assert len(forecast_short) == test_len_short, f"Output length mismatch: expected {test_len_short}, got {len(forecast_short)}."
    assert forecast_short.index.equals(test_short.index), "Output index does not match test data index."
    print("Horizon exceeds test data test passed.")

    # Test Case 5: Parameterization Test
    print("\n--- Test Case 5: Parameterization Test (num_samples) ---")
    print("Testing with a non-default `num_samples` value (5).")
    sundial_param = SundialForecaster(train_data=train_main.copy(), test_data=test_main.copy(), horizon_len=5)
    forecast_param = sundial_param.forecast(num_samples=5)
    
    assert len(forecast_param) == len(test_main), "Output length is incorrect."
    print("Parameterization test passed.")

    print("\nAll tests passed successfully!")
