from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from utils.models import TaskType

from .core import BaseEvaluator
from .models import ImportanceType, OutputFormat


class FeatureImportanceAnalyzer(BaseEvaluator):
    """
    Analyzer for computing and visualizing feature importances across horizons
    """

    def __init__(
        self,
        task_type: TaskType,
        target_col: str,
        importance_types: List[ImportanceType] = [ImportanceType.WEIGHT],
        output_format: str = OutputFormat.RAW,
        top_k: Optional[int] = None,
    ):
        """Initialize feature importance analyzer"""
        super().__init__(task_type, target_col)
        self.importance_types = importance_types
        self.output_format = output_format
        self.top_k = top_k

    def compute_importance(self, model, feature_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Compute feature importance from trained model

        Parameters
        ----------
        model : object
            Trained model with feature_importances_ or get_booster() method
        feature_names : List[str]
            Names of features

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping importance type to importance values
        """
        importances = {}

        for imp_type in self.importance_types:
            try:
                if hasattr(model, "get_booster"):  # XGBoost model
                    importance_dict = model.get_booster().get_score(importance_type=imp_type)
                    # Align with feature names
                    imp_values = np.array(
                        [importance_dict.get(f, 0.0) for f in feature_names]
                    )
                elif hasattr(model, "feature_importances_"):  # Sklearn-style model

                    imp_values = model.feature_importances_
                else:
                    logger.warning(f"Model does not support importance type '{imp_type}'")
                    continue

                importances[imp_type] = self._format_importance(imp_values)

            except Exception as e:
                logger.error(f"Failed to compute {imp_type} importance: {str(e)}")
                continue

        return importances

    def _format_importance(self, importance_values: np.ndarray) -> np.ndarray:
        """
        Format importance values according to output_format

        Parameters
        ----------
        importance_values : np.ndarray
            Raw importance values

        Returns
        -------
        np.ndarray
            Formatted importance values
        """
        match self.output_format:
            case OutputFormat.RAW:
                return importance_values
            case OutputFormat.LOG_SCALED:
                return np.log1p(importance_values)
            case OutputFormat.NORMALIZED:
                total = np.sum(importance_values)
                return importance_values / total if total > 0 else importance_values
            case _:
                raise ValueError(f"Invalid output_format: {self.output_format}")

    def evaluate(
        self,
        models: Dict[Tuple[str, int], Any],
        feature_names: List[str],
    ) -> Dict[Tuple[str, int], pd.DataFrame]:
        """
        Evaluate feature importance across multiple horizons

        Parameters
        ----------
        models : Dict[int, Any]
            dictionary mapping horizon to trained model
        feature_names : List[str]
            List of feature names (must be consistent across horizons)

        Returns
        -------
        Dict[int, pd.DataFrame]
            dictionary mapping horizon to importance DataFrame
        """
        horizon_importances: Dict[Tuple[str, int], pd.DataFrame] = {}

        for (identifier, horizon), model in models.items():
            try:
                importances = self.compute_importance(model, feature_names)

                # Create DataFrame
                df = pd.DataFrame(
                    {
                        "feature": feature_names,
                        **{
                            f"importance_{imp_type}": values
                            for imp_type, values in importances.items()
                        },
                    }
                )

                # Sort by first importance type
                first_imp_col = f"importance_{self.importance_types[0]}"
                df = df.sort_values(first_imp_col, ascending=False)

                # Apply top_k filter
                top_k = self.top_k or len(df)
                df = df.head(top_k)

                horizon_importances[(identifier, horizon)] = df

            except Exception as e:
                logger.warning(
                    f"Failed to compute importance for model "
                    f"{identifier} (FH={horizon}): {str(e)}"
                )
                continue

        return horizon_importances

    def plot_importance(
        self,
        importance_results: Dict[Tuple[str, int], pd.DataFrame],
        figsize: Tuple[int, int] = (12, 8),
    ) -> Tuple[Figure, Dict[Tuple[str, int], Axes]]:
        """
        Plot feature importance across horizons

        Parameters
        ----------
        importance_results : Dict[Tuple[str, int], pd.DataFrame]
            Results from evaluate method
        figsize : Tuple[int, int], default=(12, 8)
            Figure size for plots
        """

        n_horizons = len(set([k[1] for k in importance_results.keys()]))
        n_models = len(set([k[0] for k in importance_results.keys()]))
        n_types = len(self.importance_types)

        num_rows = n_models * n_types
        num_cols = n_horizons

        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(figsize[0] * num_cols, figsize[1] * num_rows),
        )
        if num_cols * num_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        plots = {}
        for i, imp_type in enumerate(self.importance_types):
            for j, ((identifier, horizon), df) in enumerate(importance_results.items()):
                ax = axes[i * n_horizons + j]

                col_name = f"importance_{imp_type}"
                if col_name in df.columns:
                    from plots.evaluate import plot_feature_importance

                    # Horizontal bar plot
                    plot_feature_importance(
                        df,
                        imp_column=col_name,
                        top_n=self.top_k,
                        show_values=True,
                        ax=ax,
                    )

                    ax.set_title(f"{identifier.title()} - Horizon {horizon}")
                    plots[(identifier, horizon)] = ax

        # Add title
        fig.suptitle("Feature Importance Analysis", fontsize=18, fontweight="bold")
        plt.tight_layout()
        return fig, plots
