import logging
import warnings
import multiprocessing
import numpy as np
import pandas as pd
import tqdm
from chronos_mlx import ChronosPipeline

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _run_forecast_process(queue, train_data, test_data, horizon_len, num_samples, model_name):
    """
    A picklable function to run in a separate process to avoid resource leaks.
    """
    try:
        forecaster = ChronosForecaster(
            train_data=train_data,
            test_data=test_data,
            model_name=model_name,
            horizon_len=horizon_len,
        )
        forecast = forecaster._internal_forecast(num_samples=num_samples)
        queue.put(forecast)
    except Exception as e:
        logging.error(f"Error in forecast subprocess: {e}")
        queue.put(None)


class ChronosForecaster:
    """
    A class for performing Chronos forecasting on time series data using MLX.
    """

    def __init__(
        self, train_data, test_data, model_name="amazon/chronos-t5-base", horizon_len=1
    ):
        """
        Initializes the ChronosForecaster with training and test data.
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
        self.model_name = model_name
        self.pipeline = None  # Initialize pipeline later

    def _initialize_pipeline(self):
        """Initializes the Chronos pipeline."""
        if self.pipeline is None:
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                dtype="bfloat16",
            )

    def _internal_forecast(self, num_samples=20):
        """
        Internal forecast generation logic.
        """
        self._initialize_pipeline()
        logging.info(f"Performing Chronos Forecast with horizon={self.horizon}...")

        context = self.train_data.copy()
        forecasts = []

        for i in tqdm.tqdm(range(0, len(self.test_data), self.horizon)):
            context_values = context.values
            forecast_tensor = self.pipeline.predict(
                context_values,
                self.horizon,
                num_samples=num_samples,
            )
            forecast = np.quantile(forecast_tensor[0], 0.5, axis=0)
            forecasts.extend(forecast)
            new_obs = self.test_data[i : i + self.horizon]
            context = pd.concat([context, new_obs])

        logging.info(
            f"Chronos forecast generated for {len(self.test_data)} total steps."
        )
        if len(forecasts) > self.test_data.shape[0]:
            forecasts = forecasts[: self.test_data.shape[0]]

        return pd.Series(
            forecasts,
            index=self.test_data.index,
            name=f"Chronos Forecast (horizon={self.horizon})",
        )

    def forecast(self, num_samples=20):
        """
        Generates forecasts in a separate process to prevent resource leaks.
        """
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        process = ctx.Process(
            target=_run_forecast_process,
            args=(
                queue,
                self.train_data,
                self.test_data,
                self.horizon,
                num_samples,
                self.model_name,
            ),
        )
        process.start()
        result = queue.get()
        process.join()

        if result is None:
            raise RuntimeError("Chronos forecasting subprocess failed.")
        return result


if __name__ == "__main__":
    # Required for macOS to prevent fork-related issues
    multiprocessing.set_start_method("spawn", force=True)
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
        ChronosForecaster(train_data=list(train_main), test_data=test_main)
    except TypeError as e:
        print(f"Successfully caught expected error: {e}")

    try:
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
    # Note: We now instantiate the class and then call forecast
    chronos_happy = ChronosForecaster(
        train_data=train_main.copy(),
        test_data=test_main.copy(),
        horizon_len=horizon_happy,
    )
    forecast_happy = chronos_happy.forecast(num_samples=2)

    assert isinstance(forecast_happy, pd.Series), "Output is not a pandas Series."
    assert len(forecast_happy) == test_len_happy, "Output length mismatch."
    assert forecast_happy.index.equals(test_main.index), "Output index mismatch."
    print("Core functionality test passed.")

    # Test Case 3: Incongruent Horizon
    print("\n--- Test Case 3: Incongruent Horizon ---")
    test_incongruent = data.iloc[80:97]
    horizon_incongruent = 5
    chronos_incongruent = ChronosForecaster(
        train_data=train_main.copy(),
        test_data=test_incongruent.copy(),
        horizon_len=horizon_incongruent,
    )
    forecast_incongruent = chronos_incongruent.forecast(num_samples=2)

    assert len(forecast_incongruent) == len(test_incongruent), "Output length mismatch."
    assert forecast_incongruent.index.equals(test_incongruent.index), "Output index mismatch."
    print("Incongruent horizon test passed.")

    print("\nAll tests passed successfully!")

