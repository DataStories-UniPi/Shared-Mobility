from typing import Any, Dict, List, Optional, Self, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.multiclass import unique_labels


class PlotConfig:
    """Configuration for plotting validation curves."""

    def __init__(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        show_training: bool = True,
        colors: Optional[Dict[str, str]] = None,
        line_styles: Optional[Dict[str, str]] = None,
    ):
        self.figsize = figsize
        self.show_training = show_training
        self.colors = colors or {
            "train": "#1f77b4",
            "validation": "#ff7f0e",
            "eval": "#ff7f0e",
        }
        self.line_styles = line_styles or {"train": "-", "validation": "--", "eval": "--"}


# Abstract base class to reduce code duplication
class _BaseMultiOutputXGB(BaseEstimator):
    """Base class for MultiOutput XGBoost estimators.

    Contains shared functionality between regressor and classifier.
    """

    def __init__(self, **xgb_params: Any) -> None:
        """Initialize with XGBoost parameters.

        Args:
            **xgb_params: Parameters to pass to individual XGBoost estimators
        """
        self.xgb_params = xgb_params

    def _ensure_2d_targets(self, y: np.ndarray) -> np.ndarray:
        """Ensure targets are 2D array."""
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    def _prepare_eval_sets(
        self,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]],
    ) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
        """Prepare evaluation sets for each output."""
        if eval_set is None:
            return None

        eval_sets_per_output = []
        for X_val, y_val in eval_set:
            y_val = self._ensure_2d_targets(y_val)
            y_val = np.asarray(y_val)

            eval_sets_per_output.append((X_val, y_val))
        return eval_sets_per_output

    def _fit_model_for_output(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]],
        **fit_params: Any,
    ) -> xgb.Booster:
        """Fit a single model for one output."""
        model = self._create_base_estimator()
        current_eval_set = None
        if eval_set is not None:
            current_eval_set = [(X_val, y_val) for X_val, y_val in eval_set]

        model.fit(X, y, eval_set=current_eval_set, **fit_params)
        return model

    def _fit_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        **fit_params: Any,
    ) -> Self:
        """Core fitting logic shared between regressor and classifier."""
        y = self._ensure_2d_targets(y)
        y = np.asarray(y)

        self.n_outputs_ = y.shape[1]
        self.models_ = []
        self.evals_result_ = []

        # Prepare eval_set for each output
        eval_sets_per_output = self._prepare_eval_sets(eval_set, self.n_outputs_)

        # Fit separate model for each output
        for i in range(self.n_outputs_):
            current_eval = eval_sets_per_output[i] if eval_sets_per_output else None
            model = self._fit_model_for_output(X, y[:, i], current_eval, **fit_params)
            self.models_.append(model)

            # Store evaluation results if available
            if hasattr(model, "evals_result_"):
                self.evals_result_.append(model.evals_result_)

        return self

    def _create_base_estimator(self) -> Union[xgb.XGBRegressor, xgb.XGBClassifier]:
        """Create base estimator - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_base_estimator")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using all fitted models."""
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return predictions

    def get_feature_importance(self, importance_type: str = "weight") -> np.ndarray:
        """Get feature importance for each output."""
        importances = []
        for model in self.models_:
            importances.append(model.feature_importances_)
        return np.array(importances)

    def plot_validation_curves(self, config: Optional[PlotConfig] = None) -> None:
        """Plot validation curves for each output.

        Args:
            config: Configuration for plotting. If None, default configuration is used.
        """
        if config is None:
            config = PlotConfig()

        n_outputs = len(self.models_)

        if config.figsize is None:
            config.figsize = (5 * min(n_outputs, 3), 5 * ((n_outputs - 1) // 3 + 1))

        # Determine subplot layout
        n_cols = min(n_outputs, 3)
        n_rows = (n_outputs - 1) // n_cols + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=config.figsize)

        # Handle single subplot case
        if n_outputs == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_outputs > 1 else [axes]
        else:
            axes = axes.flatten()

        for i, model in enumerate(self.models_):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                break

            if hasattr(model, "evals_result_") and model.evals_result_:
                # Track which metrics we've plotted to avoid duplicates
                plotted_metrics = set()

                for eval_name, metrics in model.evals_result_.items():
                    for metric_name, values in metrics.items():
                        # Determine if this is training or validation data
                        is_training = eval_name.lower() in ["train", "training"]
                        is_validation = eval_name.lower() in [
                            "validation",
                            "valid",
                            "eval",
                            "test",
                        ]

                        # Skip training loss if show_training is False
                        if is_training and not config.show_training:
                            continue

                        # Create descriptive label
                        if is_training:
                            label = f"Training {metric_name}"
                            color = config.colors.get("train", "#1f77b4")
                            linestyle = config.line_styles.get("train", "-")
                            alpha = 0.8
                        elif is_validation:
                            label = f"Validation {metric_name}"
                            color = config.colors.get("validation", "#ff7f0e")
                            linestyle = config.line_styles.get("validation", "--")
                            alpha = 1.0
                        else:
                            label = f"{eval_name} {metric_name}"
                            color = None  # Let matplotlib choose
                            linestyle = "-"
                            alpha = 0.8

                        # Avoid plotting duplicate metrics
                        metric_key = f"{eval_name}_{metric_name}"
                        if metric_key not in plotted_metrics:
                            ax.plot(
                                values,
                                label=label,
                                color=color,
                                linestyle=linestyle,
                                alpha=alpha,
                                linewidth=1.5,
                            )
                            plotted_metrics.add(metric_key)

                # Customize plot appearance
                ax.set_title(f"Output {i+1} - Training Progress", fontweight="bold")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Loss")
                ax.legend(loc="best", framealpha=0.9)
                ax.grid(True, alpha=0.3)

                # Add early stopping marker if available
                if hasattr(model, "best_iteration") and model.best_iteration is not None:
                    ax.axvline(
                        x=model.best_iteration,
                        color="red",
                        linestyle=":",
                        alpha=0.7,
                        label=f"Best iteration: {model.best_iteration}",
                    )
                    # Update legend to include early stopping line
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, loc="best", framealpha=0.9)

                # Improve y-axis formatting
                ax.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 2))

            else:
                # No evaluation results available
                ax.text(
                    0.5,
                    0.5,
                    "No evaluation\nresults available",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    alpha=0.6,
                )
                ax.set_title(f"Output {i+1} - No eval_set provided")

        # Hide unused subplots
        for i in range(n_outputs, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()


class MultiOutputXGBRegressor(_BaseMultiOutputXGB, RegressorMixin):
    """Multi-output XGBoost regressor with eval_set support.

    This estimator fits separate XGBoost regressors for each target variable,
    enabling early stopping and validation monitoring for multi-output regression.

    Attributes:
        models_: List of fitted XGBRegressor instances, one per output.
        n_outputs_: Number of output targets.
        evals_result_: Evaluation results for each model if eval_set was provided.
    """

    def _create_base_estimator(self) -> xgb.XGBRegressor:
        """Create XGBRegressor instance."""
        return xgb.XGBRegressor(**self.xgb_params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        **fit_params: Any,
    ) -> Self:
        """Fit separate XGBoost regressors for each output.

        Args:
            X: Training input samples.
            y: Target values.
            eval_set: List of validation sets for early stopping.
            **fit_params: Additional parameters for XGBoost fit method.

        Returns:
            self: Returns the instance itself.
        """
        return self._fit_models(X, y, eval_set, **fit_params)


class MultiOutputXGBClassifier(_BaseMultiOutputXGB, ClassifierMixin):
    """Multi-output XGBoost classifier with eval_set support.

    This estimator fits separate XGBoost classifiers for each target variable,
    enabling early stopping and validation monitoring for multi-output classification.

    Attributes:
        models_: List of fitted XGBClassifier instances, one per output.
        classes_: Classes for each output.
        n_outputs_: Number of output targets.
        evals_result_: Evaluation results for each model if eval_set was provided.
    """

    def _create_base_estimator(self) -> xgb.XGBClassifier:
        """Create XGBClassifier instance."""
        return xgb.XGBClassifier(**self.xgb_params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        **fit_params: Any,
    ) -> Self:
        """Fit separate XGBoost classifiers for each output.

        Args:
            X: Training input samples.
            y: Target class labels.
            eval_set: List of validation sets for early stopping.
            **fit_params: Additional parameters for XGBoost fit method.

        Returns:
            self: Returns the instance itself.
        """
        # Store classes for each output
        y = self._ensure_2d_targets(y)
        y = np.asarray(y)

        self.classes_ = []
        for i in range(y.shape[1]):
            self.classes_.append(unique_labels(y[:, i]))

        return self._fit_models(X, y, eval_set, **fit_params)

    def predict_proba(self, X: np.ndarray) -> List[np.ndarray]:
        """Predict class probabilities for each output.

        Args:
            X: Input samples.

        Returns:
            probabilities: List containing probability arrays for each output.
        """
        probabilities = []
        for model in self.models_:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)
                probabilities.append(prob)
            else:
                # Fallback for binary classification
                pred = model.predict(X)
                prob = np.column_stack([1 - pred, pred])
                probabilities.append(prob)
        return probabilities

    def predict_log_proba(self, X: np.ndarray) -> List[np.ndarray]:
        """Predict class log-probabilities for each output.

        Args:
            X: Input samples.

        Returns:
            log_probabilities: List containing log-probability arrays for each output.
        """
        probas = self.predict_proba(X)
        return [np.log(proba) for proba in probas]

    def decision_function(self, X: np.ndarray) -> List[np.ndarray]:
        """Decision function for each output.

        Args:
            X: Input samples.

        Returns:
            decisions: List containing decision function values for each output.
        """
        decisions = []
        for model in self.models_:
            if hasattr(model, "decision_function"):
                decision = model.decision_function(X)
                decisions.append(decision)
            else:
                # Fallback using predict_proba
                proba = model.predict_proba(X)
                if proba.shape[1] == 2:  # Binary case
                    decision = proba[:, 1] - proba[:, 0]
                else:  # Multiclass case
                    decision = proba
                decisions.append(decision)
        return decisions
