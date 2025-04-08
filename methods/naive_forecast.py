import logging
import pandas as pd
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NaiveForecaster:
    """
    A class for performing rolling naive forecasting on time series data.

    In each step, it forecasts `horizon` ahead using the last known *actual*
    observation from the combined training and testing data available up to that point.
    """

    def __init__(self, train_data, test_data):
        """
        Initializes the NaiveForecaster with training and test data.

        Args:
            train_data (pd.Series): The training data. Must be sorted chronologically.
            test_data (pd.Series): The test data (provides the forecast horizon and
                                   actual values for rolling forecast updates).
                                   Must be sorted chronologically and follow train_data.
        """
        # Input type validation
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
        self.test_index = test_data.index
        self.full_data_for_lookup = pd.concat([train_data, test_data]) # Combine for easier lookup

        # logging.info("NaiveForecaster initialized.")
        # logging.info(f"Training data length: {len(train_data)}, Test data length: {len(test_data)}")

    def forecast(self, horizon=1):
        """
        Performs a rolling naive forecast.

        Iterates through the test period. For each block of `horizon`,
        it uses the *last actual observed value* prior to that block
        as the forecast for all steps within that block.

        Args:
            horizon (int): The number of steps to forecast in each rolling window.

        Returns:
            pd.Series: The naive forecast, indexed like the original test data.
        """
        # logging.info(f"Performing rolling Naive Forecast with horizon={horizon}...")
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer.")

        forecast_values = []
        total_test_len = len(self.test_data)
        train_len = len(self.train_data)

        # Iterate through the test set indices in chunks of horizon
        for i in range(0, total_test_len, horizon):
            # Determine the index in the *combined* data of the last known actual value
            # If i=0, use the last value of train data (index train_len - 1)
            # If i>0, use the actual value from test data just before this chunk starts.
            # The index in test_data is i-1. The index in full_data is train_len + (i-1).
            last_known_actual_idx = train_len - 1 if i == 0 else train_len + i - 1

            # Get the actual value from the combined series
            last_known_value = self.full_data_for_lookup.iloc[last_known_actual_idx]
            logging.debug(f"Forecast chunk starting at test index {i}. Using value from full_data index {last_known_actual_idx}: {last_known_value}")

            # Determine how many steps to forecast in *this* specific chunk
            steps_in_this_chunk = min(horizon, total_test_len - i)

            # Create the forecast for this chunk by repeating the last known value
            chunk_forecast = [last_known_value] * steps_in_this_chunk
            forecast_values.extend(chunk_forecast)
            logging.debug(f"  > Forecasting {steps_in_this_chunk} steps with value {last_known_value}")

        # Create the final pandas Series
        final_forecast = pd.Series(forecast_values, index=self.test_index, name=f"Rolling Naive Forecast (horizon={horizon})")

        # logging.info(f"Rolling naive forecast generated for {total_test_len} total steps.")
        return final_forecast

