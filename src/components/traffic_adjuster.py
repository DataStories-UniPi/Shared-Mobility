from typing import List, Self

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context, check_is_fitted
from sklearn.utils._param_validation import HasMethods


class TrafficAdjuster(TransformerMixin, BaseEstimator):
    _parameter_constraints = {
        "quantiles": [HasMethods(["__iter__"]), None],
        "scale": [int],
        "use_diff": ["boolean"],
    }

    def __init__(
        self,
        quantiles: List[float] | None = None,
        scale: int = 2,
        use_diff: bool = True,
    ):
        self.quantiles = quantiles
        self.scale = scale
        self.use_diff = use_diff
        self.bins_ = {}

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None) -> Self:
        X = self._validate_data(X, accept_sparse=False, cast_to_ndarray=False)

        self.quantiles = self._validate_quantiles()

        for col in X.columns:
            series = X[col].diff().fillna(0) if self.use_diff else X[col]
            _, bins = pd.qcut(series, q=self.quantiles, retbins=True, duplicates="drop")

            if len(bins) <= 2:
                raise ValueError("Quantile bins not diverse enough. Try fewer bins.")

            # Extend edges to cover all future data
            bins = np.concatenate(([float("-inf")], bins, [float("inf")]))
            self.bins_[col] = bins

        return self

    def transform(self, X) -> pd.DataFrame:
        """Transform data by applying quantile-based adjustments.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        X_out : pd.DataFrame of shape (n_samples, 2 * n_features)
            Transformed DataFrame with two new columns per input feature:
            - `{col}_change`: Categorical bin label (ordered)
            - `{col}_adjusted`: Original value multiplied by bin-specific factor

        Raises
        ------
        NotFittedError
            If the transformer has not been fitted.
        """
        check_is_fitted(self, ["bins_"])
        X = self._validate_data(X, accept_sparse=False, reset=False, cast_to_ndarray=False)

        transformed = {}

        for col in X.columns:
            series = X[col].diff().fillna(0) if self.use_diff else X[col]

            n_bins = len(self.bins_[col]) - 1
            labels = self._create_labels(n_bins)
            mapper = self._create_adjustment_mapping(labels)

            binned = pd.cut(
                series,
                bins=self.bins_[col],
                include_lowest=True,
                labels=labels,
            )

            transformed.update(
                {
                    f"{col}_magnitude": binned,
                    f"{col}_adjusted": X[col] * binned.map(mapper).astype(float),
                }
            )

        self.feature_names_out_ = list(transformed.keys())

        return pd.DataFrame(transformed, index=X.index)

    def _create_adjustment_mapping(self, labels: List[str]) -> dict:
        """Create a mapping from bin labels to adjustment multipliers.

        Multipliers follow a geometric progression centered around 'Medium'.
        For example, with scale=2 and 5 bins: [0.25, 0.5, 1, 2, 4]

        Parameters
        ----------
        labels : List[str]
            List of bin labels (e.g., ['Low', 'Medium', 'High']).

        Returns
        -------
        mapping : dict[str, float]
            Mapping from label to multiplier.
        """
        n = len(labels)
        bound = n // 2  # Number of levels below and above neutral (center)
        exponents = range(-bound, n - bound)  # e.g., [-2, -1, 0, 1, 2] for n=5
        multipliers = [self.scale**x for x in exponents]
        return dict(zip(labels, multipliers, strict=True))

    def _create_labels(self, num_bins: int):
        """Generate semantic labels for quantile bins.

        Parameters
        ----------
        num_bins : int
            Number of bins to label.

        Returns
        -------
        labels : List[str]
            List of human-readable bin labels. Uses presets for common sizes,
            falls back to generic 'Q0', 'Q1', ...

        Notes
        -----
        Presets:
        - 3 bins → ['Low', 'Medium', 'High']
        - 5 bins → ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        """
        match num_bins:
            case 3:
                return ["Low", "Medium", "High"]
            case 5:
                return ["Very_Low", "Low", "Medium", "High", "Very_High"]
            case _:
                return [f"Q{i}" for i in range(num_bins)]

    def _validate_quantiles(self) -> List[float]:
        """
        Validate quantiles and return them if they are valid.

        Returns
        -------
        quantiles : np.ndarray[float]
            The validated quantiles.
        """
        quantiles = np.asarray(self.quantiles)

        if quantiles is None:
            raise ValueError(f"No quantiles provided. Got: {quantiles}")

        if quantiles[0] != 0 or quantiles[-1] != 1:
            logger.error(
                f"Quantiles must start at 0 and end at 1. "
                f"Found {quantiles[0]=}, {quantiles[-1]=}"
            )
            raise ValueError("Quantiles must start at 0 and end at 1.")

        if not np.all((quantiles >= 0) & (quantiles <= 1)):
            logger.error(f"Quantiles must be within [0, 1]. Found: {quantiles}")
            raise ValueError("Quantiles must be within [0, 1].")

        if not np.all(np.diff(quantiles) > 0):
            logger.error(
                f"Quantiles must be strictly monotonically increasing. Found: {quantiles}"
            )
            raise ValueError("Quantiles must be strictly monotonically increasing.")

        return list(quantiles)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
        return np.array(self.feature_names_out_)
        return np.array(self.feature_names_out_)
        return np.array(self.feature_names_out_)
        return np.array(self.feature_names_out_)
        return np.array(self.feature_names_out_)
