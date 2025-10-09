"""
Main module for evaluating model scalability w.r.t. dataset size.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit

from config.constants import TIME_COLUMN
from model.evaluation.evaluator import Evaluator
from utils.models import Trainable

from .splitter import IncrementalSplitter
from .validator import DataValidator


class GroupedTimeSeriesEvaluator:
    """
    Evaluate model performance across increasing dataset sizes using TimeSeriesSplit.
    Supports panel data with (group_id, timestamp) index.
    """

    def __init__(
        self,
        model: Trainable,
        group_col: str,
        n_increments: int = 5,
        n_cv_folds: int = 3,
        time_col: str = TIME_COLUMN,
    ):
        """
        Args:
            model: Sklearn-compatible model (with fit/predict)
            n_increments: Number of incremental dataset sizes
            n_cv_folds: Number of folds in TimeSeriesSplit per increment
        """
        self.model = model
        self.group_col = group_col
        self.n_increments = n_increments
        self.n_cv_folds = n_cv_folds
        self.time_col = time_col

        self.metric_calculator_ = Evaluator()
        self.incremental_splitter_ = IncrementalSplitter(n_increments=n_increments)
        self.cv_ = TimeSeriesSplit(n_splits=n_cv_folds)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, pd.DataFrame]:
        """
        Run full evaluation: incremental training size + CV + metric aggregation.

        Args:
            X: Features with MultiIndex (group_id, timestamp)
            y: Target with same index

        Returns:
            DataFrame with columns: increment, fold, MAE, MASE, MAPE, RMSE
        """
        DataValidator.validate_index(X, y, index_cols=[self.group_col, self.time_col])
        DataValidator.validate_time_structure(X, group_col=self.group_col)

        results = []

        # Generate N incremental training sets (start counting from 1)
        for i, (X_tr_inc, y_tr_inc) in enumerate(self.incremental_splitter_.split(X, y), 1):
            logger.info(
                f"Evaluating increment [{i}/{self.n_increments}], "
                f"training size: {len(X_tr_inc)}"
            )

            # Perform TimeSeriesSplit on current incremental set
            X_tr_inc = X_tr_inc.sort_index(level=self.time_col)
            y_tr_inc = y_tr_inc.loc[X_tr_inc.index]

            # Extract unique timestamps for CV
            timestamps = X_tr_inc.index.get_level_values(self.time_col).unique()
            if len(timestamps) < 3:
                logger.warning(
                    f"Too few timestamps ({len(timestamps)}) for CV in increment {i+1}"
                )
                continue

            cv_results = self._cross_validate(X_tr_inc, y_tr_inc, timestamps)

            for res in cv_results:
                res["increment"] = i
                results.append(res)

        results = pd.DataFrame(results)

        return {
            "raw": results,
            "grouped": results.groupby("increment")
            .mean()
            .drop(columns=["fold"])
            .reset_index(),
        }

    def _cross_validate(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        timestamps: pd.Index,
        **fit_params: Any,
    ) -> List[Dict[str, Any]]:
        """
        Perform TimeSeriesSplit on current training set.

        Args:
            X_tr: Training features
            y_tr: Training target
            timestamps: Unique timestamps

        Returns:
            List of metric dictionaries per fold
        """
        cv_results = []

        for fold, (train_idx, test_idx) in enumerate(self.cv_.split(timestamps)):
            logger.debug(f"Fold [{fold + 1}/{self.cv_.get_n_splits()}]")
            if len(test_idx) == 0:
                continue

            cutoff_time = timestamps[train_idx[-1]]
            test_times = timestamps[test_idx]

            # Train set: all data <= cutoff
            train_mask = X_tr.index.get_level_values(self.time_col) <= cutoff_time
            X_tr_fold = X_tr[train_mask]
            y_tr_fold = y_tr[train_mask]

            # Test set: data in test_times (per group)
            test_mask = X_tr.index.get_level_values(self.time_col).isin(test_times)
            X_te_fold = X_tr[test_mask]  # Use full X since y may not exist yet

            y_pred = self.model.fit(X_tr_fold, y_tr_fold, **fit_params).predict(X_te_fold)

            # Ensure predictions are non-negative integers
            y_pred = np.round(np.maximum(y_pred, 0)).astype(int)

            y_pred_series = pd.Series(y_pred, index=X_te_fold.index, name=y_tr.name)
            y_true_series = y_tr.loc[y_pred_series.index]
            metrics = self.metric_calculator_.evaluate(
                y_true_series, y_pred_series, y_train=y_tr_fold
            )

            avg_metrics = {col: np.mean(metrics[col]) for col in metrics.columns}

            cv_results.append({"fold": fold + 1, **avg_metrics})

        return cv_results
