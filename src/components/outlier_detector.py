import numbers
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions


class OutlierDetector(TransformerMixin, BaseEstimator):
    _parameter_constraints = {
        "lower": [Interval(numbers.Integral, 0, 100, closed="both")],
        "upper": [Interval(numbers.Integral, 0, 100, closed="both")],
        "thresh": [Interval(numbers.Real, 0, 3, closed="right")],
        "method": [StrOptions(options={"drop", "cap"})],
    }

    def __init__(
        self,
        lower: int = 25,
        upper: int = 75,
        thresh: float = 1.5,
        method: Literal["drop", "cap"] = "drop",
    ):
        """
        Initialize the OutlierDetector with specified percentile bounds and threshold.

        Parameters
        ----------
        lower : int, optional
            Lower percentile for the interquartile range calculation.
        upper : int, optional
            Upper percentile for the interquartile range calculation.
        thresh : float, optional
            Threshold multiplier for the interquartile range to determine outliers.

        """
        self.lower_percentile = lower
        self.upper_percentile = upper
        self.thresh = thresh
        self.method = method

    def _detect_outliers(self, data: pd.Series | np.ndarray) -> pd.Series:
        """
        Detects outliers in the data using vectorized operations for efficiency.
        Points are classified as outliers if they are more than `thresh` times
        the IQR away from the median.

        Parameters
        ----------
        data : pd.Series or np.ndarray
            Input data to detect outliers from.

        Returns
        -------
        np.ndarray
            Boolean array where `True` indicates that the element is an outlier.
        """

        Q1, Q3 = np.percentile(data, [self.lower_percentile, self.upper_percentile])
        IQR = Q3 - Q1
        bounds = np.array([Q1 - self.thresh * IQR, Q3 + self.thresh * IQR])

        return (data < bounds[0]) | (data > bounds[1])

    def cap_outliers(self, data: pd.Series | np.ndarray) -> np.ndarray:
        """
        Caps the outliers in the data to the percentile bounds.

        This function replaces values outside the specified percentile bounds with the
        closest bound value, effectively capping the outliers to avoid extreme values.

        Parameters
        ----------
        data : pd.Series or np.ndarray
            Input data in which outliers need to be capped.

        Returns
        -------
        np.ndarray
            Array with outliers capped to the lower or upper percentile bounds.
        """

        bounds = np.percentile(data, [self.lower_percentile, self.upper_percentile])
        return np.where(
            data < bounds[0], bounds[0], np.where(data > bounds[1], bounds[1], data)
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Use vectorized operations to detect outliers and assign NaNs
        """
        Transforms the input data by replacing outliers with NaNs.

        Parameters
        ----------
        X : pd.Series or np.ndarray
            Input data in which outliers need to be replaced with NaNs.

        Returns
        -------
        np.ndarray
            Array with outliers replaced with NaNs.
        """
        X[:] = np.where(self._detect_outliers(X), np.nan, X)

        return X
        return X
        return X
        return X
        return X
