from typing import Any, List

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context, check_is_fitted, clone
from sklearn.utils._param_validation import HasMethods
from tqdm import tqdm

from config.constants import TIME_COLUMN


class GroupTransformer(TransformerMixin, BaseEstimator):
    _parameter_constraints = {
        "base_transformer": [HasMethods(["fit", "transform"])],  # type: ignore
        "group_col": [str, HasMethods(["__iter__", "__len__"])],  # type: ignore
        "time_col": [str],
    }

    def __init__(
        self,
        base_transformer: TransformerMixin,
        group_col: str | List[str],
        time_col: str = TIME_COLUMN,
    ):
        self.base_transformer = base_transformer
        self.group_col = [group_col] if isinstance(group_col, str) else group_col
        self.time_col = time_col

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pd.DataFrame, y=None):
        X = self._validate_data(X, cast_to_ndarray=False)  # type: ignore

        if not all([group_col in X.columns for group_col in self.group_col]):
            raise ValueError("Group column(s) not found in DataFrame")

        def fit_group(group: Any, group_data: pd.DataFrame):
            group_data = self._validate_timestamps(group, group_data)
            transformer = clone(self.base_transformer)  # type: ignore
            fitted_transformer = transformer.fit(group_data, y)
            return group, fitted_transformer

        grouped = X.groupby(self.group_col, observed=True)

        results = [
            fit_group(group_key, group_data)
            for group_key, group_data in tqdm(grouped, "Fitting groups", grouped.ngroups)
        ]

        self.fitted_transformers_ = dict(results)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame):
        """Transform each group separately, then combine results."""
        check_is_fitted(self)

        X = self._validate_data(X, reset=False, cast_to_ndarray=False)  # type: ignore

        def transform_group(group_key: Any, group_data: pd.DataFrame) -> pd.DataFrame:

            group_data = self._validate_timestamps(group_key, group_data)
            timestamps = group_data[self.time_col]
            transformer = self.fitted_transformers_[group_key]
            transformed = transformer.transform(group_data)

            if not isinstance(transformed, pd.DataFrame):
                try:
                    feature_names = transformer.get_feature_names_out()
                except (AttributeError, NotImplementedError):
                    feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]
                transformed = pd.DataFrame(transformed, columns=feature_names)

            return transformed.assign(**{TIME_COLUMN: timestamps})

        grouped = X.groupby(self.group_col, observed=True, as_index=False)

        results = [
            transform_group(group_key, group_data)
            for group_key, group_data in tqdm(grouped, "Transforming groups", grouped.ngroups)
        ]

        if results:
            # Concatenate the results
            self.groups_ = [key for key, _ in grouped]

            logger.debug("Constructing `MultiIndex` for new data")
            index = pd.MultiIndex.from_tuples(
                [
                    tuple(group) + (timestamp,)
                    for group, df in zip(self.groups_, results, strict=True)
                    for timestamp in pd.to_datetime(df[self.time_col], unit="s")
                ],
                names=list(self.group_col) + [self.time_col],
            )

            logger.debug(f"Index construction complete ({index.names})")

            logger.debug("Merging groups")
            result = (
                pd.concat(results, axis=0)
                .drop(self.time_col, axis=1)
                .reset_index(drop=True)
                .set_index(index)
            )

            self.feature_names_out_ = result.columns
            rows, cols = result.shape
            logger.info(f"Merge completed: {rows:,} rows, {cols} features")
            return result

        return pd.DataFrame()

    def _validate_timestamps(
        self,
        group: str | List[str],
        group_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate the timestamps in the group data.

        If the timestamps are not unique or not in a monotonic increasing order,
        sort the group data by timestamp.

        Parameters
        ----------
        group : object
            The group name.
        group_data : pd.DataFrame
            The group data.

        Returns
        -------
        pd.DataFrame
            The validated and sorted group data, if necessary.
        """

        timestamps = group_data[self.time_col]

        if not pd.api.types.is_datetime64_any_dtype(timestamps):
            # logger.warning(
            #     f"Group '{' '.join(group)}' has non-datetime timestamps."
            #     f"Converting to datetime."
            # )
            group_data[self.time_col] = pd.to_datetime(timestamps, unit="s")

        if timestamps.nunique() != len(timestamps) or not timestamps.is_monotonic_increasing:
            # logger.warning(
            #     f"Group '{' '.join(group)}' has non-unique or non-monotonic timestamps. "
            #     f"Sorting by timestamp."
            # )
            group_data = group_data.sort_values(self.time_col)

        return group_data

    def get_feature_names_out(self, input_features=None):
        """Get feature names for the transformed data."""
        check_is_fitted(self)

        return self.feature_names_out_
        return self.feature_names_out_
        return self.feature_names_out_
        return self.feature_names_out_
        return self.feature_names_out_
        return self.feature_names_out_
        return self.feature_names_out_
        return self.feature_names_out_
        return self.feature_names_out_
        return self.feature_names_out_
