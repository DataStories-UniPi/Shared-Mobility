"""
Data validation for panel time series.
"""

from typing import List, Optional

import pandas as pd
from loguru import logger


class DataValidator:
    """
    Validates input data structure for panel forecasting evaluation.
    """

    @staticmethod
    def validate_index(
        df: pd.DataFrame,
        y: Optional[pd.Series] = None,
        index_cols: Optional[List[str]] = None,
    ) -> None:
        """
        Ensure DataFrame/Series has MultiIndex with names ['group_id', 'timestamp'].

        Args:
            df: Input feature DataFrame
            y: Optional target series

        Raises:
            ValueError: If index is invalid
        """
        index_cols = index_cols or ["group_id", "timestamp"]
        obj = y if y is not None else df

        if not isinstance(obj.index, pd.MultiIndex):
            raise ValueError(
                "Index must be a MultiIndex with levels ['group_id', 'timestamp']"
            )

        if list(obj.index.names) != index_cols:
            raise ValueError(f"Index level names must be {index_cols}, got {obj.index.names}")
        logger.debug("Index validation passed.")

    @staticmethod
    def validate_time_structure(df: pd.DataFrame, group_col: str = "group_id") -> None:
        """
        Check that timestamps are monotonic within each group.
        """
        for group_id, group in df.groupby(level=group_col):
            ts = group.index.get_level_values("timestamp")
            if not ts.is_monotonic_increasing:
                raise ValueError(
                    f"Timestamps for group '{group_id}' are not monotonically increasing."
                )
        logger.debug("Time structure validation passed.")
