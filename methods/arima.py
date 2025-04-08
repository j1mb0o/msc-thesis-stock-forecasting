import pmdarima as pm
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
        

    

    def fit(self):
        """
        Trains the ARIMA model on the training data. Using AutoArima from pmdarima.
        """
        # logging.info(f"Training ARIMA model with order={self.order} and seasonal_order={self.seasonal_order}...")
        adf = pm.arima.ndiffs(self.train_data, test='adf')
        kpps = pm.arima.ndiffs(self.train_data, test='kpss')
        
        self.model = pm.auto_arima(
            self.train_data,
            start_p=1, 
            start_q=1,
            max_d= max(adf, kpps),
            suppress_warnings=True,
            stepwise=True,
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
            new_obs = self.test_data[i:i+horizon]
            
            forecast = self.model.predict(n_periods=horizon)
            forecasts.extend(forecast)

            self.model.update(new_obs)

        logging.info(f"ARIMA forecast generated for {len(self.test_data)} total steps.")

        final_forecast = pd.Series(forecasts, index=self.test_data.index, name=f"ARIMA Forecast (horizon={horizon})")
        return final_forecast
        

if __name__ == "__main__":
    data = pd.Series(range(100), index=pd.date_range(start='2023-01-01', periods=100, freq='D'))
    train, test = data.iloc[:80], data.iloc[80:] # Train on first 15, test on last 5 (indices 15-19)
    arima = ArimaForecaster(train_data=train, test_data=test)
    arima.fit()
    arima.forecast(horizon=1)
    