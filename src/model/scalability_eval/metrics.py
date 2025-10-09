"""
Forecasting metrics using sktime.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sktime.performance_metrics.forecasting import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_absolute_scaled_error,
    mean_squared_error,
)


class ForecastingMetricCalculator:
    """
    Compute forecasting metrics per group and aggregate.
    """

    def __init__(self, fh: List[int], group_col: str):

        self.fh = fh
        self.group_col = group_col

    def _get_sp_from_fh(self) -> int:
        """Infer seasonal period from fh (simple heuristic)."""
        diffs = np.diff(self.fh)
        return int(diffs[0]) if len(diffs) > 0 else 1

    def calculate(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_train: pd.Series,
    ) -> Dict[str, float]:
        """
        Calculate metrics for aligned y_true and y_pred (both with MultiIndex).

        Args:
            y_true: Ground truth
            y_pred: Predictions
            y_train: Training series for MASE scaling

        Returns:
            Dictionary of aggregated metrics (mean across groups)
        """
        metrics = {}
        groups = y_true.index.get_level_values(self.group_col).unique()

        mae_scores, rmse_scores, mape_scores, mase_scores = [], [], [], []

        for gid in groups:
            y_true_g = y_true.xs(gid, level=self.group_col)
            y_pred_g = y_pred.xs(gid, level=self.group_col)
            y_train_g = (
                y_train.xs(gid, level=self.group_col) if gid in y_train.index else pd.Series()
            )

            if y_pred_g.empty or y_true_g.empty:
                continue

            # Align indices
            common_idx = y_true_g.index.intersection(y_pred_g.index)
            y_true_g = y_true_g[common_idx]
            y_pred_g = y_pred_g[common_idx]

            if len(y_true_g) == 0:
                continue

            # Convert to sktime format
            y_true_g = y_true_g.sort_index()
            y_pred_g = y_pred_g.reindex(y_true_g.index, method="nearest")

            # Compute metrics
            mae = mean_absolute_error(y_true_g, y_pred_g)
            rmse = mean_squared_error(y_true_g, y_pred_g, square_root=True)
            mape = mean_absolute_percentage_error(y_true_g, y_pred_g, symmetric=True)

            # MASE requires training data
            if len(y_train_g) > 1:
                mase = mean_absolute_scaled_error(
                    y_true_g,
                    y_pred_g,
                    sp=self._get_sp_from_fh(),
                    y_train=y_train_g,
                )
            else:
                mase = np.nan

            mae_scores.append(mae)
            rmse_scores.append(rmse)
            mape_scores.append(mape)
            mase_scores.append(mase)

        metrics.update(
            {
                "MAE": float(np.nanmean(mae_scores)) if mae_scores else np.nan,
                "RMSE": float(np.nanmean(rmse_scores)) if rmse_scores else np.nan,
                "MAPE": float(np.nanmean(mape_scores)) if mape_scores else np.nan,
                "MASE": float(np.nanmean(mase_scores)) if mase_scores else np.nan,
            }
        )

        return metrics
