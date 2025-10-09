from itertools import product
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context, check_is_fitted

from config.constants import DEFAULT_LAGS, DEFAULT_WINDOWS


class TemporalExtractor(TransformerMixin, BaseEstimator):

    _parameter_constraints = {
        "lags": [int, "array-like"],
        "windows": [int, "array-like"],
        "stats": [str, "array-like"],
    }

    def __init__(
        self,
        lags: Iterable[int] = DEFAULT_LAGS,
        windows: Iterable[int] = DEFAULT_WINDOWS,
        stats: Iterable[str] = {"mean"},
    ) -> None:
        self.lags = lags
        self.windows = windows
        self.stats = stats

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        Fit the transformer (stateless operation).

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Input features.
        y : array-like, default=None
            Target values (ignored).

        Returns
        -------
        self : object
            Returns self.
        """
        X = self._validate_data(X, accept_sparse=False, cast_to_ndarray=False)

        return self

    def transform(self, X):
        """
        Transform the input data by adding temporal features.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Input features.

        Returns
        -------
        pd.DataFrame
            Original data with additional temporal features.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=False, reset=False, cast_to_ndarray=False)
        temporal_features = {}

        for col in X.columns:
            temporal_features[col] = X[col]

            # Lagged features
            temporal_features.update(
                {f"{col}_lag_{lag}": X[col].shift(lag).bfill() for lag in self.lags}
            )

            # Rolling features
            temporal_features.update(
                {
                    f"{col}_rolling_{stat}_{window}": getattr(
                        X[col].rolling(window, min_periods=1), stat
                    )()
                    for window, stat in product(self.windows, self.stats)
                }
            )

            # Exponential smoothing
            temporal_features.update(
                {
                    f"{col}_ewm_{window}": X[col]
                    .ewm(span=window, adjust=False, min_periods=1)
                    .mean()
                    for window in self.windows
                }
            )

        result = pd.DataFrame(temporal_features, index=X.index)
        self.feature_names_out_ = result.columns

        return result

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

        Returns
        -------
        np.ndarray
            Output feature names.
        """
        return self.feature_names_out_
        return self.feature_names_out_
        return self.feature_names_out_
        return self.feature_names_out_
        return self.feature_names_out_
        return self.feature_names_out_
        return self.feature_names_out_
