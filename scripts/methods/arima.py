import pmdarima as pm
from pmdarima.arima import StepwiseContext
import pandas as pd
import logging
import tqdm

import warnings

from methods import _validate_train_test

warnings.filterwarnings("ignore")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArimaForecaster:
    """
    A class for performing ARIMA forecasting on time series data.
    """

    def __init__(self, train_data, test_data):
        """
        Initializes the ArimaForecaster with training and test data.

        Args:
            train_data (pd.Series): The training data. Must be sorted chronologically.
            test_data (pd.Series): The test data. Must be sorted chronologically and follow train_data.
        """
        _validate_train_test(train_data, test_data)

        self.train_data = train_data
        self.test_data = test_data
        self.model = None
        self.order = None

    def fit(self):
        """Trains the ARIMA model on the training data using AutoArima from pmdarima."""
        with StepwiseContext(max_steps=200):
            self.model = pm.auto_arima(
                self.train_data,
                start_p=1,
                start_q=1,
                max_d= 10,
                max_p=10,
                max_q=10,
                suppress_warnings=True,
                stepwise=True,
                out_of_sample_size=10,
                error_action='ignore'
            )
        logging.info("ARIMA model training complete.")
        self.order = self.model.order
        logging.info(f"Best order: {self.order}")

    def forecast(self, horizon=1):
        """Generates rolling forecasts for the given horizon, updating the model with actuals."""
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer.")

        logging.info(f"Performing ARIMA Forecast with horizon={horizon}...")
        forecasts = []
        for i in tqdm.tqdm(range(0,self.test_data.shape[0], horizon)):
            new_obs = self.test_data[i:i+horizon]
            forecast = self.model.predict(n_periods=horizon) #type: ignore
            forecasts.extend(forecast)
            self.model.update(new_obs) # type: ignore

        logging.info(f"ARIMA forecast generated for {len(self.test_data)} total steps.")

        return pd.Series(forecasts[:len(self.test_data)], index=self.test_data.index, name=f"ARIMA Forecast (horizon={horizon})")


if __name__ == "__main__":
    data = pd.Series(range(101), index=pd.date_range(start='2023-01-01', periods=101, freq='D'))
    train, test = data.iloc[:80], data.iloc[80:]
    arima = ArimaForecaster(train_data=train, test_data=test)
    arima.fit()
    arima.forecast(horizon=1)
