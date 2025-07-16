
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
            
            # For simplicity, we'll take the mean of the samples as the forecast
            forecast = output[0].mean(dim=0).numpy()
            
            forecasts.extend(forecast)

            # Update the training data with the new observations from the test set
            new_obs = self.test_data[i:i+self.horizon]
            context = pd.concat([context, new_obs])


        logging.info(f"Sundial forecast generated for {len(self.test_data)} total steps.")

        final_forecast = pd.Series(forecasts[:len(self.test_data)], index=self.test_data.index, name=f"Sundial Forecast (horizon={self.horizon})")
        return final_forecast

if __name__ == "__main__":
    data = pd.Series(range(101), index=pd.date_range(start='2023-01-01', periods=101, freq='D'))
    train, test = data.iloc[:80], data.iloc[80:] # Train on first 15, test on last 5 (indices 15-19)
    
    sundial = SundialForecaster(train_data=train, test_data=test)
    sundial.forecast()
