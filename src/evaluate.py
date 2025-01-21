from typing import Dict, TypeAlias

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.preprocessing import LabelEncoder

MetricStats: TypeAlias = Dict[str, float]  # A single metric's statistics
BenchmarkMetrics: TypeAlias = Dict[str, MetricStats]  # All metrics in a benchmark

EPS = np.finfo(np.float64).eps


def _percentage_error(y_true, y_pred, symmetric=False):
    """Percentage error.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs)
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs)
             where fh is the forecasting horizon
        Forecasted values.

    symmetric : bool, default = False
        Whether to calculate symmetric percentage error.

    Returns
    -------
    percentage_error : float

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of \
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """
    if symmetric:

        percentage_error = (
            2 * np.abs(y_true - y_pred) / np.maximum(np.abs(y_true) + np.abs(y_pred), EPS)
        )
    else:
        percentage_error = (y_true - y_pred) / np.maximum(np.abs(y_true), EPS)
    return percentage_error


def mean_absolute_percentage_error(
    y_true,
    y_pred,
    symmetric: bool = True,
    multioutput: str | None = "uniform_average",
):

    output_errors = np.average(
        np.abs(_percentage_error(y_true, y_pred, symmetric=symmetric)),
        axis=0,
    )

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def evaluate(y_true, y_pred, method):
    metrics = {}

    if method == "reg":
        metrics["R2"] = r2_score(y_true, y_pred)
        metrics["RMSE"] = root_mean_squared_error(y_true, y_pred)
        metrics["MAE"] = mean_absolute_error(y_true, y_pred)
        metrics["sMAPE"] = mean_absolute_percentage_error(y_true, y_pred)
    else:
        labels = list(range(3))

        if len(np.unique(y_true)) < 3 or len(np.unique(y_pred)) < 3:
            le = LabelEncoder()
            y_true = le.fit_transform(y_true)
            y_pred = le.transform(y_pred)

        metrics["Accuracy"] = accuracy_score(y_true, y_pred)
        metrics["F1"] = f1_score(y_true, y_pred, average="macro", labels=labels)
        metrics["Recall"] = recall_score(y_true, y_pred, average="macro", labels=labels)
        metrics["Precision"] = precision_score(y_true, y_pred, average="macro", labels=labels)

    return metrics
