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

    def __init__(self, train_data, test_data, horizon_len=1):
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

        self.train_data = train_data
        self.test_data = test_data
        self.model = None
        
        self.tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=32,
                horizon_len=horizon_len,
                num_layers=50,
                use_positional_embedding=False,
                context_len=2048,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
        )
    def forecast(self, horizon:int =1):
        """
        Generates forecasts for the given horizon.

        Args:
            horizon (int): The number of steps to forecast.

        Returns:
            pd.Series: The forecasts.
        """
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        
        logging.info(f"Performing TimesFM Forecast with horizon={horizon}...")
        forecasts = []
        for i in tqdm.tqdm(range(0,self.test_data.shape[0], horizon)):
            new_obs = self.test_data[i:i+horizon]
            
            #TODO: Implement TimesFM model prediction
            forecast = [0]*horizon
            forecasts.extend(forecast)

            #TODO: Implement TimesFM model update

        