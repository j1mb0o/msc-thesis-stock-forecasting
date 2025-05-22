import pmdarima as pm
from pmdarima.arima import StepwiseContext
import pandas as pd
import logging
import tqdm

import warnings

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
            order (tuple): The (p, d, q) order of the ARIMA model.
            seasonal_order (tuple): The (P, D, Q, s) seasonal order of the ARIMA model.
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
        self.model = None
        self.order = None
        
    # the results are the same, so we can avoid fitting multiple times
    def fit(self):
        """
        Trains the ARIMA model on the training data. Using AutoArima from pmdarima.
        """
        # logging.info(f"Training ARIMA model with order={self.order} and seasonal_order={self.seasonal_order}...")
        # adf = pm.arima.ndiffs(self.train_data, test='adf')
        # kpps = pm.arima.ndiffs(self.train_data, test='kpss')
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
        
        # self.model.fit(self.train_data)
        logging.info("ARIMA model training complete.")
        self.order = self.model.order
        logging.info(f"Best order: {self.order}")
        
    def forecast(self, horizon=1):
        """
        Generates forecasts for the given horizon.

        Args:
            horizon (int): The number of steps to forecast.

        Returns:
            pd.Series: The forecasts.
        """
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        
        logging.info(f"Performing ARIMA Forecast with horizon={horizon}...")
        forecasts = []
        for i in tqdm.tqdm(range(0,self.test_data.shape[0], horizon)):
        # for i in tqdm.tqdm(range(0, 252, horizon)):
            new_obs = self.test_data[i:i+horizon]
            
            forecast = self.model.predict(n_periods=horizon)
            forecasts.extend(forecast)

            self.model.update(new_obs)

        logging.info(f"ARIMA forecast generated for {len(self.test_data)} total steps.")

        final_forecast = pd.Series(forecasts[:len(self.test_data)], index=self.test_data.index, name=f"ARIMA Forecast (horizon={horizon})")
        return final_forecast
        
# def experiment_n_times(n=10, train=None, test=None):
#     if train is None or test is None:
#         raise ValueError("train and test must be provided.")
    
#     for i in range(n):
#         arima = ArimaForecaster(train_data=train, test_data=test)
#         arima.fit()
        


if __name__ == "__main__":
    from utils.model_data_prep import prepare_data_for_modeling
    train, test = prepare_data_for_modeling()
    
    train, test = prepare_data_for_modeling()
    arima = ArimaForecaster(train_data=train, test_data=test)
    # arima.fit()
    # arima.forecast(horizon=1)
    # arima.experiment_n_times(10)
    # experiment_n_times(10, train, test)
    
