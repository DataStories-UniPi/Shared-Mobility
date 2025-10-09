import warnings
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sktime.performance_metrics.forecasting import (
    mean_absolute_error,
    mean_absolute_percentage_error,
)

from utils.models import TaskType

from .core import BaseEvaluator
from .models import BenchmarkResult


class PerformanceBenchmarker(BaseEvaluator):
    """
    Performance benchmarker with quantile-based metrics and group-level analysis
    """

    def __init__(
        self,
        task_type: TaskType,
        target_col: str | List[str],
        regression_metrics: List[str] = ["mae", "rmse", "mape", "r2"],
        classification_metrics: List[str] = ["accuracy", "f1", "precision", "recall"],
        quantiles: List[float] = [0.25, 0.5, 0.75],
        group_col: Optional[str] = None,
    ):
        """
        Initialize performance benchmarker

        Parameters
        ----------
        task_type : str
            Type of ML task ('reg' or 'clf')
        target_col : str
            Name of target column
        regression_metrics : List[str], default=['mae', 'mse', 'mape', 'smape']
            Regression metrics to compute
        classification_metrics : List[str], default=['accuracy', 'f1', 'precision', 'recall']
            Classification metrics to compute
        quantiles : List[float], default=[0.25, 0.5, 0.75]
            Quantiles to compute for each metric
        group_col : Optional[str], default=None
            Column name for group-level analysis
        """
        super().__init__(task_type, target_col)
        self.regression_metrics = regression_metrics
        self.classification_metrics = classification_metrics
        self.quantiles = quantiles
        self.group_col = group_col
        self.metric_functions = self._get_metric_functions()

    def _get_metric_functions(self) -> Dict[str, Callable]:
        """
        Get metric functions based on task type

        Returns
        -------
        Dict[str, Callable]
            dictionary mapping metric name to function
        """
        if self.task_type == TaskType.REGRESSION:
            return {
                "mae": mean_absolute_error,
                "rmse": root_mean_squared_error,
                "mape": mean_absolute_percentage_error,
                "r2": r2_score,
            }

        return {
            "accuracy": accuracy_score,
            "f1": lambda y_true, y_pred: f1_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "precision": lambda y_true, y_pred: precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall": lambda y_true, y_pred: recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
        }

    def compute_sample_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> Dict[str, float]:
        """
        Compute metrics for a single sample or group

        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        X_test : Optional[pd.DataFrame], default=None
            Test features (needed for group-level analysis)

        Returns
        -------
        Dict[str, float]
            dictionary of computed metrics
        """
        metrics = {}
        active_metrics = (
            self.regression_metrics
            if self.task_type == TaskType.REGRESSION
            else self.classification_metrics
        )

        for metric_name in active_metrics:
            try:
                if metric_name in self.metric_functions:
                    if metric_name == "mape":
                        value = self.metric_functions[metric_name](
                            y_true, y_pred, symmetric=True
                        )
                    else:
                        value = self.metric_functions[metric_name](y_true, y_pred)

                    metrics[metric_name] = float(value)
                else:
                    warnings.warn(f"Metric '{metric_name}' not implemented")
            except Exception as e:
                warnings.warn(f"Error computing {metric_name}: {str(e)}")
                metrics[metric_name] = np.nan

        return metrics

    def compute_group_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> Dict[str, List[float]]:
        """
        Compute metrics at group level

        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        X_test : pd.DataFrame
            Test features containing group column

        Returns
        -------
        Dict[str, List[float]]
            dictionary mapping metric name to List of group-level values
        """
        if self.group_col is None or self.group_col not in y_true.index.names:
            raise ValueError(f"Group column '{self.group_col}' not found in X_test")

        group_metrics = {
            metric: []
            for metric in (
                self.regression_metrics
                if self.task_type == TaskType.REGRESSION
                else self.classification_metrics
            )
        }

        # Get unique groups
        groups = y_true.index.get_level_values(self.group_col).unique()

        # Calculate metrics for each station and direction
        for group in groups:
            try:
                group_y_true = y_true.xs(group, level=self.group_col)
                group_y_pred = y_pred.xs(group, level=self.group_col)

                metrics = self.compute_sample_metrics(group_y_true, group_y_pred)

                for metric_name, value in metrics.items():
                    if not np.isnan(value):
                        group_metrics[metric_name].append(value)

            except KeyError:
                logger.warning(f"Group {group} not found in all datasets")
                continue
        return group_metrics

    def evaluate(
        self,
        results: Dict[Tuple[str, int], List[BenchmarkResult]],
        X_test_data: Optional[Dict[Tuple[str, int], pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Evaluate performance across horizons with quantile statistics.

        Parameters
        ----------
        results : Dict[Tuple[str, int], List[BenchmarkResult]]
            Benchmark results grouped by (identifier, horizon).
        X_test_data : Optional[Dict[Tuple[str, int], pd.DataFrame]]
            Optional test features for computing group-level metrics.

        Returns
        -------
        pd.DataFrame
            Pivoted DataFrame with quantile statistics per metric.
        """
        performance_data = []

        # Group results by identifier and horizon
        for (identifier, horizon), run_results in results.items():
            run_results = [r for r in run_results if r.error is None]
            if not run_results:
                continue

            for result in run_results:
                if result.actuals.ndim == 1:
                    y_true = pd.Series(result.actuals)
                    y_pred = pd.Series(result.predictions)
                else:
                    y_true = pd.DataFrame(result.actuals)
                    y_pred = pd.DataFrame(result.predictions)

                if self.group_col and X_test_data and (identifier, horizon) in X_test_data:
                    X_test = X_test_data[(identifier, horizon)]
                    logger.debug(f"Computing group metrics for {identifier=} with {horizon=}")

                    y_true.index = X_test.index
                    y_pred.index = X_test.index

                    group_metrics = self.compute_group_metrics(y_true, y_pred)
                    for metric_name, values in group_metrics.items():
                        if values:
                            q_values = np.quantile(values, self.quantiles)
                            performance_data.extend(
                                [
                                    {
                                        "identifier": identifier,
                                        "horizon": horizon,
                                        "metric": metric_name,
                                        "quantile": q,
                                        "value": qv,
                                    }
                                    for q, qv in zip(self.quantiles, q_values)
                                ]
                            )
                else:
                    metrics = self.compute_sample_metrics(y_true, y_pred)
                    for metric_name, val in metrics.items():
                        if not np.isnan(val):
                            performance_data.append(
                                {
                                    "identifier": identifier,
                                    "horizon": horizon,
                                    "metric": metric_name,
                                    "quantile": 0.5,  # Median fallback for single sample
                                    "value": val,
                                }
                            )

        if not performance_data:
            return pd.DataFrame()

        df = pd.DataFrame(performance_data)

        return df.pivot_table(
            index=["identifier", "horizon", "metric"],
            columns="quantile",
            values="value",
            fill_value=np.nan,
        )

    def plot_performance(
        self,
        performance_df: pd.DataFrame,
        figsize: Tuple[int, int] = (15, 10),
    ):
        """
        Plot performance metrics with quantile distributions

        Parameters
        ----------
        performance_df : pd.DataFrame
            Results from evaluate method
        figsize : Tuple[int, int], default=(15, 10)
            Figure size
        """
        if performance_df.empty:
            logger.info("No performance data to plot")
            return None

        # Reset index for easier plotting
        df_plot = performance_df.reset_index()

        # Get unique metrics and horizons
        metrics = df_plot["metric"].unique()
        horizons = df_plot["horizon"].unique()

        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, squeeze=False)

        for i, metric in enumerate(metrics):
            ax = axes[i, 0]
            metric_data = df_plot[df_plot["metric"] == metric]

            if not metric_data.empty:
                # Create box plots for each horizon
                box_data = []
                labels = []

                for horizon in sorted(horizons):
                    horizon_data = metric_data[metric_data["horizon"] == horizon]
                    if not horizon_data.empty:
                        # Extract quantile values
                        values = []
                        for _, row in horizon_data.iterrows():
                            for col in performance_df.columns:
                                if not pd.isna(row[col]):
                                    values.append(row[col])

                        if values:
                            box_data.append(values)
                            labels.append(f"H{horizon}")

                if box_data:
                    ax.boxplot(box_data, tick_labels=labels)
                    ax.set_title(f"{metric.upper()} Distribution by Horizon")
                    ax.set_ylabel(f"{metric.upper()}")
                    ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_heatmap(
        self,
        performance_df: pd.DataFrame,
        quantile: float = 0.5,
        figsize: Tuple[int, int] = (10, 6),
    ):
        """
        Create heatmap of performance metrics across horizons

        Parameters
        ----------
        performance_df : pd.DataFrame
            Results from evaluate method
        quantile : float, default=0.5
            Which quantile to display in heatmap
        figsize : Tuple[int, int], default=(10, 6)
            Figure size
        """
        if performance_df.empty:
            logger.info("No performance data to plot")
            return None

        # Extract specific quantile
        if quantile in performance_df.columns:
            heatmap_data = performance_df[quantile].unstack(level="horizon")

            plt.figure(figsize=figsize)
            sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis")
            plt.title(f"Performance Metrics Heatmap (Q{quantile})")
            plt.ylabel("Metric")
            plt.xlabel("Horizon")
            return plt.gcf()
        else:
            logger.info(f"Quantile {quantile} not found in data")
            return None
