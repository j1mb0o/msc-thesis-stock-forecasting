import pandas as pd
import logging
import tqdm

import timesfm

import warnings

from methods import _validate_train_test

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
        _validate_train_test(train_data, test_data)
        if not isinstance(horizon_len, int) or horizon_len <= 0:
            raise ValueError("horizon must be a positive integer.")

        self.horizon = horizon_len
        self.train_data = train_data
        self.test_data = test_data
        self.freq = 0 if horizon_len == 1 else 1 if (horizon_len > 1 and horizon_len <= 5) else 2
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
        """Generates rolling forecasts using the configured horizon."""
        context = self.train_data.copy()
        logging.info(f"Performing TimesFM Forecast with horizon={self.horizon}...")
        forecasts = []
        for i in tqdm.tqdm(range(0,self.test_data.shape[0], self.horizon)):
            pred, _ = self.model.forecast([context.values], [self.freq])
            context = pd.concat([context, self.test_data[i:i+self.horizon]])
            forecasts.extend(pred[0])

        if len(forecasts) > self.test_data.shape[0]:
            forecasts = forecasts[:self.test_data.shape[0]]
        return pd.Series(forecasts[:self.test_data.shape[0]], index=self.test_data.index, name=f"TimesFM Forecast (horizon={self.horizon})")


if __name__ == "__main__":
    data = pd.Series(range(101), index=pd.date_range(start='2023-01-01', periods=101, freq='D'))
    train, test = data.iloc[:80], data.iloc[80:]
    times = TimesFMForecaster(train_data=train, test_data=test, horizon_len=5)

    result = times.forecast()
    print(result)