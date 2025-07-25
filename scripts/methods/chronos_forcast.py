import logging
import warnings

import numpy as np
import pandas as pd
import torch
import tqdm
from chronos import BaseChronosPipeline

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ChronosForecaster:
    """
    A class for performing Chronos forecasting on time series data using PyTorch.
    """

    def __init__(
        self, train_data, test_data, model_name="amazon/chronos-bolt-base", horizon_len=1
    ):
        """
        Initializes the ChronosForecaster with training and test data.

        Args:
            train_data (pd.Series): The training data. Must be sorted chronologically.
            test_data (pd.Series): The test data. Must be sorted chronologically and follow train_data.
            model_name (str): The name of the Chronos model to use.
            horizon_len (int): The forecast horizon.
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
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_name,
            device_map="mps",
            torch_dtype=torch.bfloat16,
        )

    def forecast(self):
        """
        Generates forecasts for the given horizon.

        Returns:
            pd.Series: The forecasts.
        """
        logging.info(f"Performing Chronos Forecast with horizon={self.horizon}...")

        context = self.train_data.copy()
        forecasts = []

        for i in tqdm.tqdm(range(0, len(self.test_data), self.horizon)):
            # Prepare the input context
            context_tensor = torch.tensor(context.values, dtype=torch.bfloat16)

            # Generate predictions
            # forecast shape: [num_series, num_quantiles, prediction_length]
            forecast_tensor = self.pipeline.predict(
                context_tensor,
                self.horizon,
            )

            # For simplicity, we'll take the median of the quantiles as the forecast
            forecast = np.median(forecast_tensor[0].cpu().numpy(), axis=0)
            forecasts.extend(forecast)

            # Update the context with the new observations from the test set
            new_obs = self.test_data[i : i + self.horizon]
            context = pd.concat([context, new_obs])

        logging.info(
            f"Chronos forecast generated for {len(self.test_data)} total steps."
        )
        if len(forecasts) > self.test_data.shape[0]:
            forecasts = forecasts[: self.test_data.shape[0]]

        final_forecast = pd.Series(
            forecasts,
            index=self.test_data.index,
            name=f"Chronos Forecast (horizon={self.horizon})",
        )
        return final_forecast


if __name__ == "__main__":
    import numpy as np

    print("Running test suite for ChronosForecaster...")

    # Setup for functional tests
    data = pd.Series(
        np.linspace(0, 100, 100),
        index=pd.date_range(start="2023-01-01", periods=100, freq="D"),
        dtype=np.float32,
    )
    train_main, test_main = data.iloc[:80], data.iloc[80:]

    # Test Case 1: Input Validation
    print("\n--- Test Case 1: Input Validation ---")
    try:
        # Test for non-Series input
        print("Testing with list input for train_data...")
        ChronosForecaster(train_data=list(train_main), test_data=test_main)
    except TypeError as e:
        print(f"Successfully caught expected error: {e}")

    try:
        # Test for empty input
        print("Testing with empty Series for train_data...")
        empty_series = pd.Series([], dtype=float)
        ChronosForecaster(train_data=empty_series, test_data=test_main)
    except ValueError as e:
        print(f"Successfully caught expected error: {e}")
    print("Input validation tests passed.")

    # Test Case 2: Core Functionality (Happy Path)
    print("\n--- Test Case 2: Core Functionality (Happy Path) ---")
    horizon_happy = 5
    test_len_happy = len(test_main)
    print(
        f"Testing with horizon={horizon_happy} on a test set of length {test_len_happy}."
    )
    chronos_happy = ChronosForecaster(
        train_data=train_main.copy(),
        test_data=test_main.copy(),
        horizon_len=horizon_happy,
    )
    forecast_happy = chronos_happy.forecast()

    assert isinstance(forecast_happy, pd.Series), "Output is not a pandas Series."
    assert len(forecast_happy) == test_len_happy, (
        f"Output length mismatch: expected {test_len_happy}, got {len(forecast_happy)}."
    )
    assert forecast_happy.index.equals(test_main.index), (
        "Output index does not match test data index."
    )
    assert pd.api.types.is_float_dtype(forecast_happy.dtype), (
        f"Output dtype is not float, but {forecast_happy.dtype}."
    )
    print("Core functionality test passed.")

    # Test Case 3: Incongruent Horizon
    print("\n--- Test Case 3: Incongruent Horizon ---")
    test_incongruent = data.iloc[80:97]  # 17 samples
    horizon_incongruent = 5
    test_len_incongruent = len(test_incongruent)
    print(
        f"Testing with horizon={horizon_incongruent} on a test set of length {test_len_incongruent}."
    )
    chronos_incongruent = ChronosForecaster(
        train_data=train_main.copy(),
        test_data=test_incongruent.copy(),
        horizon_len=horizon_incongruent,
    )
    forecast_incongruent = chronos_incongruent.forecast()

    assert len(forecast_incongruent) == test_len_incongruent, (
        f"Output length mismatch: expected {test_len_incongruent}, got {len(forecast_incongruent)}."
    )
    assert forecast_incongruent.index.equals(test_incongruent.index), (
        "Output index does not match test data index."
    )
    print("Incongruent horizon test passed.")

    # Test Case 4: Horizon Exceeds Test Data
    print("\n--- Test Case 4: Horizon Exceeds Test Data ---")
    test_short = data.iloc[80:85]  # 5 samples
    horizon_large = 10
    test_len_short = len(test_short)
    print(
        f"Testing with horizon={horizon_large} on a test set of length {test_len_short}."
    )
    chronos_short = ChronosForecaster(
        train_data=train_main.copy(),
        test_data=test_short.copy(),
        horizon_len=horizon_large,
    )
    forecast_short = chronos_short.forecast()

    assert len(forecast_short) == test_len_short, (
        f"Output length mismatch: expected {test_len_short}, got {len(forecast_short)}."
    )
    assert forecast_short.index.equals(test_short.index), (
        "Output index does not match test data index."
    )
    print("Horizon exceeds test data test passed.")

    print("\nAll tests passed successfully!")