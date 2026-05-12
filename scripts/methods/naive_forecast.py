import logging
import pandas as pd

from methods import _validate_train_test

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
        _validate_train_test(train_data, test_data)

        self.train_data = train_data
        self.test_data = test_data
        self.test_index = test_data.index
        self.full_data_for_lookup = pd.concat([train_data, test_data])

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
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer.")

        forecast_values = []
        total_test_len = len(self.test_data)
        train_len = len(self.train_data)

        for i in range(0, total_test_len, horizon):
            # i==0 -> last train value, i>0 -> last test value before this chunk
            last_known_actual_idx = train_len - 1 if i == 0 else train_len + i - 1
            last_known_value = self.full_data_for_lookup.iloc[last_known_actual_idx]

            steps_in_this_chunk = min(horizon, total_test_len - i)
            forecast_values.extend([last_known_value] * steps_in_this_chunk)

        return pd.Series(forecast_values, index=self.test_index, name=f"Rolling Naive Forecast (horizon={horizon})")

