import pandas as pd


def _validate_train_test(train_data, test_data):
    """Common input validation for forecaster constructors.

    All forecasters in this package expect both inputs to be non-empty pandas
    Series sorted chronologically; the latter is the caller's responsibility.
    """
    if not isinstance(train_data, pd.Series):
        raise TypeError("train_data must be a pandas Series.")
    if not isinstance(test_data, pd.Series):
        raise TypeError("test_data must be a pandas Series.")
    if train_data.empty:
        raise ValueError("train_data cannot be empty.")
    if test_data.empty:
        raise ValueError("test_data cannot be empty.")
