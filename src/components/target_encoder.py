from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed Target Encoder with shrinkage towards global mean.

    Performs mean encoding of a categorical variable with Bayesian smoothing.
    Encoding = (count_cat * mean_cat + prior_weight * global_mean) / (count_cat + prior_weight)

    Parameters:
    -----------
    prior_weight : float, default=10.0
        Weight for the global mean (higher = more smoothing)
    min_samples : int, default=1
        Minimum number of samples to compute a reliable mean
    smoothing_strategy : {'additive', 'exponential'}, default='additive'
        How to apply smoothing:
        - 'additive': standard Bayesian average
        - 'exponential': weight = prior_weight / (1 + np.exp(-(count - min_samples)/10))
    """

    def __init__(
        self,
        prior_weight: float = 10.0,
        min_samples: int = 1,
        smoothing_strategy: Literal["additive", "exponential"] = "additive",
    ):
        self.prior_weight = prior_weight
        self.min_samples = min_samples
        self.smoothing_strategy = smoothing_strategy

    def fit(self, X, y=None):
        """
        Fit the encoder on X and y.

        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with categorical columns
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values (e.g., 'outbound', 'inbound')
        """
        if y is None:
            raise ValueError("Target y must be provided for target encoding.")

        X = X.copy()
        y = np.array(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim > 2:
            raise ValueError("y must be 1D or 2D")

        self.n_targets_ = y.shape[1]
        self.global_mean_ = np.nanmean(y, axis=0)
        self.encodings_ = {}

        for col in X.columns:

            # Store mapping for unseen categories
            self.encodings_[col] = {}

            for target_idx in range(self.n_targets_):
                target_name = f"target_{target_idx}"
                df = pd.DataFrame({"group": X[col], "target": y[:, target_idx]})

                # Compute group statistics
                stats = (
                    df.groupby("group")["target"]
                    .agg(["mean", "count"])
                    .rename(columns={"mean": "mean", "count": "count"})
                )

                # Apply smoothing
                if self.smoothing_strategy == "additive":
                    weight = self.prior_weight
                elif self.smoothing_strategy == "exponential":
                    # Weight increases with count
                    weight = self.prior_weight / (
                        1 + np.exp(-(stats["count"] - self.min_samples) / 10)
                    )
                else:
                    weight = self.prior_weight

                smoothed = (
                    stats["count"] * stats["mean"] + weight * self.global_mean_[target_idx]
                ) / (stats["count"] + weight)
                self.encodings_[col][target_name] = smoothed.to_dict()

        # Store feature names
        self.feature_names_out_ = [
            f"{col}__target_{i}_enc" for col in X.columns for i in range(self.n_targets_)
        ]

        return self

    def transform(self, X):
        """
        Transform categorical columns to smoothed target encodings.

        Parameters:
        -----------
        X : pandas.DataFrame

        Returns:
        --------
        X_out : pd.DataFrame or np.array (if remainder='drop' in ColumnTransformer)
        """
        check_is_fitted(self, "encodings_")
        X = X.copy()

        encoded_columns = []
        for col in X.columns:
            for target_idx in range(self.n_targets_):
                target_name = f"target_{target_idx}"
                enc_map = self.encodings_[col][target_name]
                global_val = self.global_mean_[target_idx]

                # Map known categories, fill unknown with global mean
                encoded = X[col].map(enc_map).fillna(global_val)
                encoded_columns.append(encoded.values)

        # Return as 2D array
        return np.column_stack(encoded_columns)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        check_is_fitted(self, "feature_names_out_")
        return np.array(self.feature_names_out_)
        return np.array(self.feature_names_out_)
