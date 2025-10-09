from collections.abc import Iterable
from datetime import timedelta
from typing import Optional

import holidays
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context


class TimeExtractor(TransformerMixin, BaseEstimator):
    """
    Transformer to extract time features and domain-specific patterns from datetime columns.

    Designed for use with ColumnTransformer. Focuses solely on feature extraction
    without encoding or temporal sequence features.

    Parameters
    ----------
    time_features : list of str
        Basic time features to extract. Valid values are 'year', 'month', 'day',
        'hour', 'minute', 'second', 'weekofyear', 'dayofyear', 'dayofweek'.

    time_periods : bool, default=False
        Whether to extract time period categories (night, morning_rush, etc.).

    transit_patterns : bool, default=False
        Whether to extract transit-specific patterns (rush hours, peak times).

    holiday_country : str, default=None
        Country code for holiday detection (e.g., 'LV' for Latvia).

    custom_periods : dict, default=None
        Custom time period definitions. dict with period names as keys and
        hour ranges as values. E.g., {'custom_rush': (7, 9)}

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>> df = pd.DataFrame({'timestamp': [datetime(2020, 1, 1, 8, 30, 0),
    ...                                 datetime(2020, 1, 2, 17, 45, 0)]})
    >>> extractor = TimeExtractor(
    ...     time_features=['hour', 'dayofweek'],
    ...     time_periods=True,
    ...     transit_patterns=True,
    ...     holiday_country='LV'
    ... )
    >>> result = extractor.fit_transform(df)
    """

    _parameter_constraints = {
        "time_features": ["array-like"],
        "time_periods": ["boolean"],
        "transit_patterns": ["boolean"],
        "holiday_country": [str, None],
        "custom_periods": [dict, None],
    }

    def __init__(
        self,
        time_features: Iterable[str],
        time_periods: bool = False,
        transit_patterns: bool = False,
        holiday_country: Optional[str] = None,
        custom_periods: Optional[dict[str, tuple]] = None,
    ) -> None:
        self.time_features = time_features
        self.time_periods = time_periods
        self.transit_patterns = transit_patterns
        self.holiday_country = holiday_country
        self.custom_periods = custom_periods

        # Initialize holiday calendar if specified
        self.holidays_calendar: Optional[holidays.HolidayBase] = None
        if self.holiday_country:
            self.holidays_calendar = holidays.country_holidays(self.holiday_country)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Input data with datetime column(s).
        y : array-like, default=None
            Target values (ignored).

        Returns
        -------
        self : object
            Returns self.
        """
        X = self._validate_data(X, accept_sparse=False, cast_to_ndarray=False)

        self.time_features = list(self.time_features)
        self.custom_periods = self.custom_periods or {}

        # # Validate that we can convert columns to datetime
        # for col in X.columns:
        #     # Skip columns that are already transformed
        #     if pd.api.types.is_datetime64_any_dtype(X[col]):
        #         continue
        #     try:
        #         pd.to_datetime(X[col].iloc[: min(5, len(X))])  # Test conversion on sample
        #     except (ValueError, TypeError) as e:
        #         raise ValueError(f"Column '{col}' cannot be converted to datetime: {e}")

        return self

    def transform(self, X) -> pd.DataFrame:
        """
        Transform the input data by extracting time features.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Input data with datetime column(s).

        Returns
        -------
        pd.DataFrame
            Extracted time features.
        """
        X = self._validate_data(X, accept_sparse=False, reset=False, cast_to_ndarray=False)

        result_dfs = []

        for col in X.columns:
            # Convert to datetime
            time_col = pd.to_datetime(X[col], unit="s")
            col_features = self._extract_features_for_column(time_col)
            result_dfs.append(col_features)

        # Combine all features
        result = pd.concat(result_dfs, axis=1) if result_dfs else pd.DataFrame(index=X.index)

        # Store output information
        self.feature_names_out_ = list(result.columns)
        self.n_features_out_ = result.shape[1]

        return result

    def _extract_features_for_column(self, time_series: pd.Series) -> pd.DataFrame:
        """Extract all requested features from a datetime series."""
        features = {}

        for feature in self.time_features:
            if hasattr(time_series.dt, feature):
                features[feature] = getattr(time_series.dt, feature)
            elif feature in ["week_of_year", "weekofyear"]:
                features[feature] = time_series.dt.isocalendar().week
            else:
                raise ValueError(f"Invalid time feature: '{feature}'")

        # Time period categories
        if self.time_periods:
            time_period_features = self._extract_time_periods(time_series)
            features.update(time_period_features)

        # Transit-specific patterns
        if self.transit_patterns:
            transit_features = self._extract_transit_patterns(time_series)
            features.update(transit_features)

        # Holiday features
        if self.holidays_calendar is not None:
            holiday_features = self._extract_holiday_features(time_series)
            features.update(holiday_features)

        # Custom time periods
        if self.custom_periods:
            custom_features = self._extract_custom_periods(time_series)
            features.update(custom_features)

        return pd.DataFrame(features, index=time_series.index)

    def _extract_time_periods(self, time_series: pd.Series) -> dict[str, pd.Series]:
        """Extract standard time period categories."""
        features = {}
        hour = time_series.dt.hour

        # Define standard time periods
        period_mapping = pd.cut(
            hour,
            bins=[0, 6, 9, 17, 20, 24],
            # labels=["night", "morning_rush", "midday", "evening_rush", "evening"],
            include_lowest=True,
            ordered=True,
        ).cat.codes
        features["time_period"] = period_mapping.astype(int)

        return features

    def _extract_transit_patterns(self, time_series: pd.Series) -> dict[str, pd.Series]:
        """Extract transit-specific temporal patterns."""
        hour = time_series.dt.hour
        dayofweek = time_series.dt.dayofweek

        features = {
            "is_morning_rush": ((hour >= 7) & (hour <= 9) & (dayofweek < 5)).astype(int),
            "is_evening_rush": ((hour >= 17) & (hour <= 20) & (dayofweek < 5)).astype(int),
            "is_weekend": (dayofweek >= 5).astype(int),
            "is_business_hours": ((hour >= 9) & (hour <= 18) & (dayofweek < 5)).astype(int),
            "is_school_hours": ((hour >= 8) & (hour <= 15) & (dayofweek < 5)).astype(int),
        }

        return features

    def _extract_holiday_features(self, time_series: pd.Series) -> dict[str, pd.Series]:
        """Extract holiday-related features."""

        dates = time_series.dt.date

        features = {
            "is_holiday": dates.apply(lambda x: x in self.holidays_calendar).astype(int),
            "is_day_before_holiday": dates.apply(
                lambda x: (x + timedelta(days=1)) in self.holidays_calendar
            ).astype(int),
            "is_day_after_holiday": dates.apply(
                lambda x: (x - timedelta(days=1)) in self.holidays_calendar
            ).astype(int),
        }

        # Holiday period (holiday or adjacent day)
        features["is_holiday_period"] = (
            features["is_holiday"]
            | features["is_day_before_holiday"]
            | features["is_day_after_holiday"]
        ).astype(int)

        return features

    def _extract_custom_periods(self, time_series: pd.Series) -> dict[str, pd.Series]:
        """Extract custom time period features."""
        features = {}
        hour = time_series.dt.hour

        for period_name, (start_hour, end_hour) in self.custom_periods.items():
            if start_hour <= end_hour:
                # Normal range (e.g., 9 to 17)
                condition = (hour >= start_hour) & (hour <= end_hour)
            else:
                # Overnight range (e.g., 22 to 6)
                condition = (hour >= start_hour) | (hour <= end_hour)

            features[f"is_{period_name}"] = condition.astype(int)

        return features

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features (not used, kept for API compatibility).

        Returns
        -------
        np.ndarray
            Output feature names.
        """
        if hasattr(self, "feature_names_out_"):
            return np.array(self.feature_names_out_)
        else:
            # Return empty array if not fitted
            return np.array([])
