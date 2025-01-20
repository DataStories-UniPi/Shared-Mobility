from itertools import product
from typing import List, Tuple, TypeAlias

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

Column: TypeAlias = str | List[str]
Window: TypeAlias = int | List[int]


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Feature extraction transformer for time-series data.

    Parameters
    ----------
    time_column : str
        Name of the column containing timestamps.
    target_column : str
        Name of the column containing timestamps.
    columns : List[str]
        List of columns to compute features for.
    lags : List[int]
        List of lag periods for lagged features.
    windows : List[int]
        List of rolling window sizes for statistical and exponential smoothing features.
    """

    def __init__(
        self,
        time_column: str,
        columns: Column,
        lags: Window,
        windows: Window,
    ):
        self.time_column = time_column
        self.columns = [columns] if isinstance(columns, str) else columns
        self.lags = list(range(1, lags + 1)) if isinstance(lags, int) else lags
        self.windows = [windows] if isinstance(windows, int) else windows

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from the input DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with time-series data.

        Returns
        -------
        pd.DataFrame
            DataFrame with original and extracted features.
        """

        df = X.copy()
        time_col = df[self.time_column]

        # Time-related features
        time_features = {
            "hour": time_col.dt.hour,
            "month": time_col.dt.month,
            "minute": time_col.dt.minute,
            "weekday": time_col.dt.dayofweek,
        }

        # Lagged features
        lagged_features = {
            f"{column}_lag_{lag}": df[column].shift(lag)
            for column, lag in product(self.columns, self.lags)
        }

        # Rolling features
        stats = ["mean"]
        rolling_features = {}

        for column, window in product(self.columns, self.windows):
            roll = df[column].rolling(window=window)
            for stat in stats:
                rolling_features[f"{column}_rolling_{stat}_{window}"] = getattr(roll, stat)()

        # Exponential smoothing features
        exp_smoothing_features = {
            f"{column}_ewm_{window}": df[column].ewm(span=window, adjust=True).mean()
            for column, window in product(self.columns, self.windows)
        }

        return pd.concat(
            [
                df,
                pd.DataFrame(time_features),
                pd.DataFrame(lagged_features),
                pd.DataFrame(rolling_features),
                pd.DataFrame(exp_smoothing_features),
            ],
            axis=1,
        )

    def fit_transform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def transform_data(
    X: pd.DataFrame,
    y: pd.Series,
    time_column: str = "timestamp",
    lags: List[int] = [1, 5, 10, 15],
    windows: List[int] = [5, 10, 15],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Transforms feature and target datasets by applying feature extraction and aligning indices.

    Parameters
    ----------
    X : pd.DataFrame
        Input features with a 'timestamp' column.
    y : pd.Series
        Target variable, aligned with the index of X.
    extractor : FeatureExtractor
        A scikit-learn compatible transformer to apply feature extraction.

    Returns
    -------
    X_transformed : pd.DataFrame
        Transformed features with 'timestamp' set as the index.
    y_transformed : pd.Series
        Target variable, aligned with the transformed feature set.

    Raises
    ------
    ValueError
        If 'timestamp' column is missing in X or if y cannot be aligned with X.
    """

    if time_column not in X.columns:
        raise ValueError("Input DataFrame X must contain a time column")

    columns = X.columns[1:].tolist()
    extractor = FeatureExtractor(time_column, columns, lags, windows)
    X_copy = extractor.fit_transform(X)

    # Align y with the transformed X
    if not y.index.isin(X.index).all():
        raise ValueError("Target y cannot be aligned with the features X.")

    X_copy = X_copy.set_index(time_column)
    y.index = X_copy.index

    if not len(X_copy) == len(y):
        raise ValueError("Lengths of 'X' and 'y' should be equal")

    return X_copy, y
