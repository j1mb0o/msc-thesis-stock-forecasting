import pandas as pd
import logging
import tqdm

import timesfm

# import datetime
# import os

import warnings

warnings.filterwarnings("ignore")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TimesFMForecaster:
    """
    A class for performing TimesFM forecasting on time series data.
    """

    def __init__(self, train_data, test_data, horizon_len=1, freq=0):
        """
        Initializes the TimesFMForecaster with training and test data.

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
        if not isinstance(horizon_len, int) or horizon_len <= 0:
            raise ValueError("horizon must be a positive integer.")
        

        self.horizon = horizon_len
        self.train_data = train_data
        self.test_data = test_data
        self.freq = freq
        self.model = timesfm.TimesFm(
                        hparams=timesfm.TimesFmHparams(
                            backend="gpu",
                            per_core_batch_size=32,
                            horizon_len=self.horizon,
                            num_layers=50,
                            use_positional_embedding=False,
                            context_len=2048,
                        ),
                        checkpoint=timesfm.TimesFmCheckpoint(
                            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
                    )
    def forecast(self):
        """
        Generates forecasts for the given horizon.

        Args:
            horizon (int): The number of steps to forecast.

        Returns:
            pd.Series: The forecasts.
        """
        
        context = self.train_data.copy()
        logging.info(f"Performing TimesFM Forecast with horizon={self.horizon}...")
        forecasts = []
        for i in tqdm.tqdm(range(0,self.test_data.shape[0], self.horizon)):
        # for i in tqdm.tqdm(range(0, 252, self.horizon)):
            context = pd.concat([context, self.test_data[i:i+self.horizon]])
            pred, _ = self.model.forecast([context.values], [self.freq])
            # print(pred[0], type(pred[0]))

            forecasts.extend(pred[0])
        # print(len(forecasts))
        if len(forecasts) > self.test_data.shape[0]:
            forecasts = forecasts[:self.test_data.shape[0]]
        final_forecast = pd.Series(forecasts, index=self.test_data.index, name=f"TimesFM Forecast (horizon={self.horizon})")
        return final_forecast
        

if __name__ == "__main__":
    data = pd.Series(range(101), index=pd.date_range(start='2023-01-01', periods=101, freq='D'))
    train, test = data.iloc[:80], data.iloc[80:] # Train on first 15, test on last 5 (indices 15-19)
    times = TimesFMForecaster(train_data=train, test_data=test, horizon_len=5)

    result = times.forecast()
    print(result)
    # perform test to increate the context Series