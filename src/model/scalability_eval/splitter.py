"""
Incremental dataset splitter using expanding window.
"""

from typing import Iterator, Tuple

import numpy as np
import pandas as pd


class IncrementalSplitter:
    """
    Generates N chronologically increasing training subsets.
    Each subset includes more time steps (not more groups).
    """

    def __init__(self, n_increments: int = 5):
        """
        Args:
            n_increments: Number of incremental splits to create
        """
        self.n_increments = n_increments

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Iterator[Tuple[pd.DataFrame, pd.Series]]:
        """
        Yield increasing training sets sorted by timestamp.

        Args:
            X: Feature matrix with MultiIndex (group_id, timestamp)
            y: Target vector with same index

        Yields:
            X_train_subset, y_train_subset: Increasingly larger training sets
        """
        # Sort by timestamp globally
        df = pd.concat([X, y.to_frame("target")], axis=1).sort_index(level="timestamp")

        unique_times = df.index.get_level_values("timestamp").unique()
        split_points = np.linspace(0, len(unique_times), self.n_increments + 1, dtype=int)[1:]

        for end_idx in split_points:
            cutoff_time = unique_times[end_idx - 1]
            mask = df.index.get_level_values("timestamp") <= cutoff_time
            X_tr = X.loc[mask]
            y_tr = y.loc[mask]

            yield X_tr, y_tr
