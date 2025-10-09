from itertools import product
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_is_fitted


class DerivativeTransformer(TransformerMixin, BaseEstimator):
    _parameter_constraints = {
        "orders": ["array-like", "integer"],
        "replacement_value": ["array-like"],
    }

    def __init__(self, orders: List[int], replacement_value: float = 0.0):
        """
        Initialize the DerivativeTransformer.

        Parameters:
        - orders (List[int]): List of derivative orders to compute.
        - replacement_value (List[float]): Values to replace NaN and infinities with.
        """
        self.orders = orders
        self.replacement_value = replacement_value

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pd.DataFrame, y=None) -> "DerivativeTransformer":
        """Fit the transformer on a DataFrame."""
        X = self._validate_data(X, accept_sparse=False, reset=False, cast_to_ndarray=False)

        # Store feature names
        self.feature_names_in_ = X.columns

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame by computing derivatives."""
        check_is_fitted(self)

        # Check if feature names are consistent
        if not all(self.feature_names_in_ == X.columns):
            raise ValueError(
                f"Feature names provided at fit time {self.feature_names_in_} "
                f"do not match the feature names seen in transform: {X.columns}"
            )

        # Compute various derivatives
        derivatives = {}
        for col, order in product(X.columns, self.orders):
            X_rolling = X[col].rolling(window=order, min_periods=1)
            derivatives.update(
                {
                    f"{col}_diff_{order}": X[col].diff(order),
                    f"{col}_diff_rate_{order}": X[col].diff(order).diff(),
                    f"{col}_pct_change_{order}": X[col]
                    .pct_change(periods=order)
                    .clip(lower=-1, upper=1),
                    f"{col}_deviation_rolling_{order}": X[col] - X_rolling.mean(),
                    f"{col}_cv_rolling_{order}": X_rolling.std() / X_rolling.mean(),
                }
            )

        derivatives_df = pd.DataFrame(derivatives, index=X.index)
        self.feature_names_in_out_ = derivatives_df.columns
        return derivatives_df

    def get_feature_names_out(self, input_features=None):
        """Return feature names after transformation."""
        return self.feature_names_in_out_
