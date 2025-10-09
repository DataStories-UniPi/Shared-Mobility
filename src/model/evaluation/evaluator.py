from collections.abc import Callable
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .models import EvaluationConfig, ResultFormatter, TaskType


class Evaluator:
    """Concrete implementation of the evaluation component."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize the evaluator with a configuration.

        Args:
            config (optional): Configuration for the evaluator
        """
        self.config = config or EvaluationConfig()

    def evaluate(
        self,
        y_true: pd.DataFrame | pd.Series,
        y_pred: pd.DataFrame | np.ndarray,
        *,
        y_train: Optional[pd.DataFrame | pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Evaluate model predictions with comprehensive metrics.

        Args:
            y_true : True target values
            y_pred : Predicted values
            y_train : Training target values

        Returns:
            DataFrame with evaluation metrics
        """
        # Prepare data

        y_true_df, y_pred_df, y_train_df = self._prepare_data(y_true, y_pred, y_train=y_train)

        # Select metrics based on evaluation type or custom metrics
        match self.config.eval_type:
            case TaskType.REGRESSION:
                metrics = self.config.REGRESSION_METRICS
            case TaskType.CLASSIFICATION:
                metrics = self.config.CLASSIFICATION_METRICS
            case _:
                if self.config.custom_metrics:
                    metrics = self.config.custom_metrics
                else:
                    raise ValueError(
                        "Evaluation type not specified and no custom metrics provided"
                    )

        # Initialize results storage
        results = {}

        # Get unique stations and directions
        try:
            groups = y_true_df.index.get_level_values(self.config.groupby_level).unique()
        except KeyError:
            raise ValueError(f"Index level '{self.config.groupby_level}' not found in y_true")

        directions = y_true_df.columns

        # Calculate metrics for each station and direction
        for group in groups:
            try:
                yt_station = y_true_df.xs(group, level=self.config.groupby_level)
                yp_station = y_pred_df.xs(group, level=self.config.groupby_level)

                if y_train_df is not None:
                    ytr_station = y_train_df.xs(group, level=self.config.groupby_level)

            except KeyError:
                logger.warning(f"Station {group} not found in all datasets")
                continue

            for direction in directions:
                try:
                    y_true_col = yt_station[direction]
                    y_pred_col = yp_station[f"{direction}_pred"]

                    if y_train_df is not None:
                        y_train_col = ytr_station[direction]

                    # Skip if all values are NaN
                    if np.all(np.isnan(y_true_col)) or np.all(np.isnan(y_pred_col)):
                        continue

                    # Calculate metrics
                    station_metrics = self._calculate_metrics(
                        y_true_col.values,
                        y_pred_col.values,
                        metrics,
                        y_train=y_train_col.values,  # type: ignore
                    )

                    results[(group, direction)] = station_metrics

                except Exception as e:
                    logger.warning(f"Error processing {group=}, {direction=}: {e}")
                    continue

        # Create result DataFrame
        result_df = pd.DataFrame.from_dict(results, orient="index")
        result_df.index.names = [
            self.config.groupby_level,
            "output" if len(directions) > 1 else "",
        ]

        # Print results if verbose is enabled
        if self.config.verbose:
            formatter = ResultFormatter()
            formatter.print_results(result_df)

        # Save results if path provided
        if self.config.save_path:
            self.config.save_path.parent.mkdir(exist_ok=True, parents=True)
            result_df.to_csv(self.config.save_path)
            print(f"Results saved to {self.config.save_path}")

        return result_df

    def _prepare_data(
        self,
        y_true: pd.DataFrame | pd.Series,
        y_pred: pd.DataFrame | np.ndarray,
        *,
        y_train: Optional[pd.DataFrame | pd.Series],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Prepare and validate input data for evaluation.

        Args:
            y_true : True target values
            y_pred :Predicted values
            y_train : Training target values

        Returns:
            Prepared true values, predictions, and training values
        """
        # Convert Series to DataFrame
        y_true_df = y_true.to_frame() if isinstance(y_true, pd.Series) else y_true.copy()

        # Convert predictions to DataFrame with proper indexing
        y_pred_df = pd.DataFrame(y_pred, index=y_true_df.index)
        y_pred_df.columns = [f"{col}_pred" for col in y_true_df.columns]
        if y_train is not None:
            y_train_df = (
                y_train.to_frame() if isinstance(y_train, pd.Series) else y_train.copy()
            )

            return y_true_df, y_pred_df, y_train_df

        return y_true_df, y_pred_df, None

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Dict[str, Callable],
        *,
        y_train: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics for given predictions.

        Args:
        y_true : True values
        y_pred : Predicted values
        metrics : Dictionary of metric names and functions
        y_train : Training values for metrics that require them

        Returns:
            Dictionary of metric names and values
        """
        results = {}

        for metric_name, metric_func in metrics.items():
            try:
                if metric_name == "MASE" and y_train is not None:
                    value = metric_func(y_true, y_pred, y_train=y_train)
                else:
                    value = metric_func(y_true, y_pred)
                results[metric_name] = value
            except Exception as e:
                logger.warning(f"Could not calculate {metric_name}: {e}")
                results[metric_name] = np.nan

        return results
