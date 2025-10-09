from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sktime.performance_metrics.forecasting import mean_absolute_error
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error as mape
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error as mase

from config.constants import GROUP_COLUMN
from utils.models import TaskType


class Split(StrEnum):
    """Enumeration for data splits"""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    NONE = "full"


class OutputFormat(StrEnum):
    """Enumeration for output formats"""

    RAW = "raw"
    LOG_SCALED = "log_scaled"
    NORMALIZED = "normalized"


class ImportanceType(StrEnum):
    """Enumeration for importance types"""

    WEIGHT = "weight"
    GAIN = "gain"
    COVER = "cover"


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""

    identifier: str
    horizon: int
    target_col: str | List[str]
    train_time: float = 0
    predict_time: float = 0
    total_time: float = 0
    train_size: int = 0
    test_size: int = 0
    predictions: Optional[np.ndarray] = None
    actuals: Optional[np.ndarray] = None
    error: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Configuration class for evaluation metrics."""

    # Regression metrics with their corresponding functions
    REGRESSION_METRICS: Dict[str, Callable] = field(
        default_factory=lambda: {
            "R2": r2_score,
            "MAE": mean_absolute_error,
            "RMSE": root_mean_squared_error,
            "sMAPE": lambda y_true, y_pred, **kwargs: mape(y_true, y_pred, symmetric=True),
            "MASE": mase,
        }
    )

    # Classification metrics with their corresponding functions
    CLASSIFICATION_METRICS: Dict[str, Callable] = field(
        default_factory=lambda: {
            "Accuracy": accuracy_score,
            "Precision": lambda y_true, y_pred: precision_score(
                y_true, y_pred, average="macro"
            ),
            "Recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="macro"),
            "F1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
        }
    )

    # Evaluation type (regression or classification)
    eval_type: TaskType = TaskType.REGRESSION

    # Custom metrics to use instead of defaults
    custom_metrics: Optional[Dict[str, Callable]] = None

    # Path to save results as CSV file
    save_path: Optional[Path] = None

    # Whether to print detailed results to console
    verbose: bool = False

    # Index level name for group-level analysis
    groupby_level: str = GROUP_COLUMN


@dataclass
class ResultFormatter:
    """Handles formatting and display of evaluation results."""

    percentage_metrics: List[str] = field(
        default_factory=lambda: ["R2", "Accuracy", "Precision", "Recall"]
    )

    def format_value(self, metric: str, value: float) -> str:
        """Format a single metric value for display."""
        if metric in self.percentage_metrics:
            return f"{value:>10.2%}"
        return f"{value:>10.4f}"

    def print_results(self, result_df: pd.DataFrame) -> None:
        """Print formatted results to console."""
        print(f"\n{f'[ Test Results ]':^35}\n")

        for idx, row in result_df.iterrows():
            print(f"{'=' * 35}")
            group_id, direction = idx if isinstance(idx, tuple) else (idx, "")
            header = f"[ Group {group_id} | {direction.capitalize()} ]"
            print(f"{header:^35}")

            for metric, value in row.items():
                if pd.isna(value):
                    formatted_value = f"{'N/A':>10}"
                else:
                    formatted_value = self.format_value(metric, value)
                print(f"{metric:<25}{formatted_value}")

        print(f"{'=' * 35}")
