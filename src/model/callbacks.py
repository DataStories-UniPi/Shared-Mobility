import timeit
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from IPython.display import clear_output
from loguru import logger
from matplotlib import cm, colormaps, colors
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from optuna.study import Study, StudyDirection
from optuna.trial import FrozenTrial, TrialState

import config.paths as cfg
from config.constants import LOG_INTERVAL


class ChampionCallback:
    """Callback for Optuna to track and log champion models during optimization.

    This callback detects when a new best trial is found, logs the improvement,
    and stores the results in MLflow. It handles both the initial best model
    and subsequent improvements.

    Attributes
    ----------
    log_interval : int
        The interval at which to log the champion and its improvement.
    metric_name : str
        The name of the metric to use for optimization.
    higher_is_better : bool

    Methods
    -------
    init(log_interval, metric_name, higher_is_better)
        Initialize the ChampionCallback with the given parameters.
    call(study, trial)
        Callback to log the champion model and its improvement during optimization.

    """

    def __init__(
        self,
        log_interval: int = 10,
        metric_name: str = "sMAPE",
        higher_is_better: bool = False,
    ) -> None:
        """Initialize the ChampionCallback with the given parameters.

        Parameters
        ----------
        log_interval : int, default=10
            The interval at which to log the champion model and its improvements.
        metric_name : str, default='sMAPE'
            The name of the metric to use for optimization.
        higher_is_better : bool, default=False
            Indicator of whether a higher metric score is considered better.

        """
        self.log_interval = log_interval
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        """Optuna callback to log the champion model and its improvement.

        This callback is called after each trial and checks if current is the best.
        If it is, update the winner and calculate the improvement percentage.
        Logs the trial number, hyperparameters, best value, and improvement percentage.

        Parameters
        ----------
        study : optuna.Study
            The study object that contains the optimization results.
        trial : optuna.Trial
            The current trial object.

        Returns
        -------
        None

        """
        # Get current best value and previous best (winner)
        best_value = study.best_value
        previous_best = study.user_attrs.get("winner", None)

        # Ensure there is a valid best value
        if best_value is None:
            return

        # Ensure trial score is valid
        trial_no = trial.number + 1
        if (trial_score := trial.value) is None:
            raise ValueError(f"Invalid trial {trial_no}, (`trial_score` is None)")

        # Log every N trials to avoid excessive logging
        if trial_no > 0 and trial_no % self.log_interval == 0:
            logger.info(
                f"Trial [{trial_no}/{len(study.trials)}], "
                f"best {self.metric_name} achieved so far: {best_value:.4f}",
            )

        # Check if current trial is the new best
        if best_value and previous_best != best_value:
            study.set_user_attr("winner", best_value)  # Update the winner

            if previous_best:
                # Calculate improvement percentage
                improvement_pct = (
                    ((previous_best - best_value) / best_value) * 100
                    if previous_best != 0
                    else float("inf")
                )

                # self.log_trial(trial_no, trial.params, trial_score, improvement_pct)
                logger.info(
                    f"Trial {trial_no} achieved best {self.metric_name}: {trial_score:.4f} "
                    f"with {improvement_pct:.1%} improvement",
                )
            else:
                # self.log_trial(trial_no, trial.params, trial_score)
                logger.info(f"Initial trial achieved {self.metric_name}: {trial_score:.4f}")

    def log_trial(
        self,
        trial_no: int,
        params: dict[str, Any],
        metric_value: float,
        improvement_pct: float | None = None,
    ) -> None:
        """Log the current trial's hyperparameters and metric value to MLflow.

        The run name is set to "trial_<trial_no>" and the hyperparameters are logged
        as MLflow parameters. The metric value is logged as an MLflow metric named
        "sMAPE". If the improvement percentage is provided, it is logged as an
        MLflow metric named "improvement_pct".

        Parameters
        ----------
        trial_no : int
            The trial number.
        params : dict[str, Any]
            The hyperparameters of the current trial.
        metric_value : float
            The metric value of the current trial.
        improvement_pct : float|None, optional
            The improvement percentage of the current trial over the previous winner,
            by default None.

        Returns
        -------
        None

        """
        try:
            with mlflow.start_run(run_name=f"trial_{trial_no}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("sMAPE", metric_value)

                if improvement_pct is not None:
                    mlflow.log_metric("improvement_pct", improvement_pct)

        except Exception as e:
            logger.exception(f"Failed to log to MLflow: {e!s}")


class VisualizationCallback:
    """Optuna visualization callback for generating various plots.

    Attributes
    ----------
    log_interval : int, default=10
        Number of trials between logging.
    plot_types : list[str], default=["optimization_history", "param_importances"]
        list of plot types to generate. Valid options: "optimization_history",
        "param_importances", "param_distributions", "parallel_coordinate",
        "progress_over_time", and "param_heatmap".
    save_file : bool, default=False
        Whether to save plots to file.
    live_update : bool, default=True
        Whether to clear output before logging new results.
    version : str, default="v1"
        Version of hyperparameters used for optimization.
    start_time : float
        Time when the optimization started.
    history : list[dict]
        list of dictionaries containing trial history.

    Methods
    -------
    __call__(study, trial)
        Callback function for Optuna trials to log history and generate visualizations.
    print_summary(study, trial)
        Log the summary of the current Optuna optimization trial.
    _create_visualizations(study)
        Create a visualization dashboard for Optuna optimization results.
    plot_optimization_history(df, ax, study)
        Plot the optimization history of the given Optuna study.
    plot_param_importances(ax, study)
        Plot the parameter importance of the given Optuna study.
    plot_param_distributions(df, ax, study)
        Plot parameter distributions of the given Optuna study.
    plot_parallel_coordinate(ax, study)
        Plot parallel coordinates of the given Optuna study.
    plot_performance_over_time(df, ax, study)
        Plot performance improvement over time.
    plot_contour(df, ax, study)
        Plot the contour plot of the given Optuna study.

    """

    def __init__(
        self,
        log_interval: int = LOG_INTERVAL,
        save_file: bool = False,
        live_update: bool = True,
        version: str = "v1",
    ) -> None:
        """Initialize visualization callback.

        Parameters
        ----------
        log_interval : int, default=10
            Number of trials between logging.
        plot_types : list|Nonestr]], default=None
            list of plot types to generate. If None, all plots will be generated.
            Valid options: see `DEFAULT_PLOTS`.
        save_file : bool, default=False
            Whether to save plots to file.
        live_update : bool, default=True
            Whether to clear output before logging new results.
        version : str, default="v1"
            Version of hyperparameters used for optimization.

        Returns
        -------
        None

        """
        self.log_interval = log_interval
        self.save_file = save_file
        self.live_update = live_update
        self.version = version
        self.start_time = timeit.default_timer()

        self.plot_types = [
            "optimization_history",
            "param_importances",
            "param_distributions",
            "parallel_coordinate",
            "progress_over_time",
            "param_heatmap",
        ]

        # Storage for history
        self.history: list[dict[str, Any]] = []

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        """Log history and generate visualizations.

        This method is used as a callback function to be called on each trial during
        the optimization process. It logs the history of the trial and generates
        visualizations based on the provided plot types.

        Parameters
        ----------
        study : optuna.study.Study
            The study object that contains the optimization results.
        trial : FrozenTrial
            The trial instance currently being evaluated.

        Raises
        ------
        ValueError
            If `trial.value` is None, indicating an invalid trial.

        """
        if trial.value is None:
            raise ValueError(f"Invalid trial {trial.number + 1}: `trial.value` is None ")

        params_data = trial.params.copy()
        self.history.append(
            {
                "number": trial.number + 1,
                "value": trial.value,
                "state": trial.state,  # Convert enum to string for readability
                "datetime": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                **params_data,
            }
        )

        # Print summary at intervals
        if (trial.number + 1) % self.log_interval == 0:
            self.print_summary(study, trial)

            # Generate visualizations at controlled intervals
            if len(self.history) % max(5, self.log_interval) == 0:
                self._create_visualizations(study)

    def print_summary(self, study: Study, trial: FrozenTrial) -> None:
        """Log the summary of the current Optuna optimization trial.

        Parameters
        ----------
        study : optuna.Study
            The Optuna study object.
        trial : optuna.Trial
            The current trial object.

        Returns
        -------
        None

        """
        best_value = getattr(study, "best_value", "N/A")
        best_trial = getattr(study.best_trial, "number", "N/A")
        best_params = getattr(study, "best_params", {})

        logger.debug(
            f"\nTrial {trial.number + 1} Summary:\n"
            f"Value: {trial.value:.4f}\n"
            f"Params: {dict(trial.params.items())}\n"
            f"Best value: {best_value:.4f}\n"
            f"Best trial: {best_trial}\n"
            f"Best params: {best_params}"
        )

        # Calculate improvement rate over the last 10 trials
        if len(self.history) >= self.log_interval:
            improvement = self._calculate_improvement(self.history, study.direction)
            if improvement is not None:
                logger.info(f"Recent improvement rate: {improvement:.4%}")

        # Assume self.start_time is initialized somewhere earlier in your class
        elapsed_time = timeit.default_timer() - self.start_time  # Calculate elapsed time

        if elapsed_time <= 3600:  # Check if less than or equal to one hour (3600 seconds)
            minutes, seconds = divmod(elapsed_time, 60)
            formatted_time = f"{int(minutes):02}:{int(seconds):02}"
        else:
            hours, remainder = divmod(elapsed_time, 3600)  # Calculate hours and remaining
            minutes, seconds = divmod(remainder, 60)
            formatted_time = f"{int(hours):02}.{int(minutes):02}{int(seconds):02}"

        logger.info(f"Elapsed time: {formatted_time}")

    def _calculate_improvement(
        self,
        history: list[dict],
        direction: StudyDirection,
    ) -> float | None:
        """Compute the improvement rate based on the last 10 completed trials.

        Parameters
        ----------
        history : list[dict]
            A list of trial histories containing "value" and "state".
        direction : optuna.study.StudyDirection
            The optimization direction (minimize/maximize).

        Returns
        -------
        float | None
            The improvement rate as a decimal (e.g., 0.12 for 12% improvement),
            or None if not enough data is available.

        """
        recent_values = [
            h["value"]
            for h in history[-self.log_interval :]
            if h["state"] == TrialState.COMPLETE
        ]

        if len(recent_values) < 2:
            return None  # Not enough data

        initial, latest = recent_values[0], recent_values[-1]

        if initial == 0:
            return None  # Avoid division by zero

        return (
            (initial - latest) / initial
            if direction == StudyDirection.MINIMIZE
            else (latest - initial) / initial
        )

    def _create_visualizations(self, study: Study):
        """Create a visualization dashboard for Optuna optimization results.

        The dashboard displays the optimization history, parameter importances,
        parallel coordinate plot, contour plot, parameter distributions, and
        learning curves.

        Parameters
        ----------
        study : optuna.study.Study
            The Optuna study object containing the optimization results.

        Notes
        -----
        To save the visualization, set the `save_file` attribute to a valid path.

        Examples
        --------
        >>> callback = VisualizationCallback()
        >>> callback.version = "1.0"  # Set version to save visualization under
        >>> callback.save_file = True  # Save visualization to file
        >>> study.optimize(objective, n_trials=50, callbacks=[callback])

        """
        # Convert history to DataFrame for analysis

        df = pd.DataFrame([h for h in self.history if h["state"] == TrialState.COMPLETE])

        if len(df) < 3 or not self.plot_types:
            return  # Return early if not enough data or no plots requested

        # Define supported plots
        supported_plots = {
            "optimization_history": self.__plot_optimization_history,
            "param_importances": self.__plot_param_importances,
            "param_distributions": self.__plot_param_distributions,
            "parallel_coordinate": self.__plot_parallel_coordinate,
            "progress_over_time": self.__plot_performance_over_time,
            "param_heatmap": self.__plot_contour,
        }

        # Filter enabled plot types
        enabled_plots = [
            p
            for p in self.plot_types
            if p in supported_plots and (p != "param_importances" or len(df) >= 10)
        ]

        n_plots = len(enabled_plots)
        if n_plots == 0:
            return

        # Clear previous output if live update is enabled
        if self.live_update:
            clear_output(wait=True)
        # Create figure and grid
        fig = plt.figure(figsize=(15, 3 * n_plots))
        gs = GridSpec(4, 2, figure=fig)

        # Row 1: Optimization history (spans full width)
        ax1 = fig.add_subplot(gs[0, :])
        self.__plot_optimization_history(df, ax1, study)

        # Row 2, Col 1: Parameter importance
        ax2 = fig.add_subplot(gs[1, 0])
        self.__plot_param_importances(ax2, study)

        # Row 2, Col 2: Parallel coordinate plot
        ax3 = fig.add_subplot(gs[1, 1])
        self.__plot_parallel_coordinate(ax3, study)

        # Row 3, Col 1: Contour plot
        ax4 = fig.add_subplot(gs[2, 0])
        self.__plot_contour(df, ax4, study)

        # Row 3, Col 2: Parameter distributions
        ax5 = fig.add_subplot(gs[2, 1])
        self.__plot_param_distributions(df, ax5, study)
        # Row 4: Learning curves or slice plot
        ax6 = fig.add_subplot(gs[3, :])
        self.__plot_performance_over_time(df, ax6, study)
        # Set title for the entire figure
        fig.suptitle(
            f"Optuna Optimization Dashboard - Trial {len(self.history)}",
            fontsize=16,
            y=1.02,
        )

        plt.tight_layout()
        # Save visualization if path provided
        if self.save_file:
            root = cfg.FIGURES_DIR / "1.0" / f"{self.version}"
            Path.mkdir(root, parents=True, exist_ok=True)
            file_path = root / f"optuna_trial_{df['number'].max()}.pdf"

            plt.savefig(file_path)

        plt.show()

    def __plot_optimization_history(self, df: pd.DataFrame, ax: Axes, study: Study) -> None:
        """Plot the optimization history of the given Optuna study.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the optimization history.
        ax : matplotlib.axes.Axes
            The axes to draw the plot in.
        study : optuna.study.Study
            The Optuna study object.

        Returns
        -------
        None

        """
        sns.lineplot(
            data=df,
            x="number",
            y="value",
            alpha=0.6,
            label="Trial Value",
            marker="o",
            markersize=8,
            ax=ax,
        )

        # Add rolling median to show trend
        if len(df) >= 5:
            window = min(5, len(df) // 2)
            rolling_median = df["value"].rolling(window=window, center=True).median()
            sns.lineplot(
                x=df["number"],
                y=rolling_median,
                markers=True,
                linewidth=2,
                ax=ax,
                label=f"{window}-Trial Rolling Median",
            )
        # Mark best trial
        best_trial_idx = (
            df["value"].idxmin()
            if study.direction == optuna.study.StudyDirection.MINIMIZE
            else df["value"].idxmax()
        )

        if not pd.isna(best_trial_idx):  # Check if we found a valid index
            best_trial = df.loc[best_trial_idx]
            sns.scatterplot(
                x=[best_trial["number"]],
                y=[best_trial["value"]],
                marker="*",
                s=700,
                c="red",
                label="Best Trial",
                ax=ax,
            )

        step = max(5, len(df) // 10, len(df) // 20)
        ax.set_xticks(list(range(0, len(df) + 1, step)))
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("Objective Value")
        ax.set_title("Optimization History", fontsize=14)

        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

    def __plot_param_importances(self, ax: Axes, study: Study) -> None:
        """Plot the parameter importance of the given Optuna study.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot in.
        study : optuna.study.Study
            The Optuna study containing the optimization results.

        """
        try:
            # Get importance scores and normalize
            importances = optuna.importance.get_param_importances(study)
            if not importances:
                ax.set_title("Parameter Importance (No data)")
                return

            # Convert to DataFrame and normalize values
            importance_df = pd.DataFrame(
                {
                    "Parameter": list(importances.keys()),
                    "Importance": list(importances.values()),
                },
            ).sort_values("Importance", ascending=False)

            # Normalize importances to sum to 1
            importance_df["Importance"] /= importance_df["Importance"].sum()

            # Plot with seaborn
            sns.barplot(
                data=importance_df,
                x="Importance",
                y="Parameter",
                hue="Parameter",
                ax=ax,
                palette="Blues_d",
                legend=False,
            )

            ax.set_title("Normalized Parameter Importance", fontsize=14)
            ax.set_xlabel("Normalized Importance")
            ax.set_ylabel("Parameter")
            sns.despine(ax=ax)

            # Add percentage labels
            for i, v in enumerate(importance_df["Importance"]):
                ax.text(v + 0.01, i, f"{v * 100:.1f}%", va="center")

        except Exception as e:
            ax.set_title(f"Parameter Importance (Error: {e!s})")

    def __plot_param_distributions(self, df: pd.DataFrame, ax: Axes, study: Study) -> None:
        """Plot parameter distributions of the given Optuna study.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the optimization history.
        ax : matplotlib.axes.Axes
            The axes to draw the plot in.
        study : optuna.study.Study
            The Optuna study containing the optimization results.

        Returns
        -------
        None

        """
        if len(df) < 5:
            ax.text(
                0.5,
                0.5,
                "Not enough trials for parameter distribution plot",
                horizontalalignment="center",
                verticalalignment="center",
            )
            return

        # Find top-performing trials
        n_top = max(3, len(df) // 5)  # At least 3 trials, or 20% of total
        if study.direction == optuna.study.StudyDirection.MINIMIZE:
            top_df = df.nsmallest(n_top, "value")
        else:
            top_df = df.nlargest(n_top, "value")

        # Select most important parameter
        try:
            importances = optuna.importance.get_param_importances(study)
            top_param = max(importances.items(), key=lambda x: x[1])[0]
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Could not extract most important parameter\n({e!s}",
                horizontalalignment="center",
                verticalalignment="center",
            )

        # Extract parameter values
        try:
            # Create histogram
            sns.histplot(
                x=df[top_param].values,
                bins=10,
                alpha=0.5,
                label="All Trials",
                stat="percent",
                ax=ax,
            )
            # ax.hist(top_values, alpha=0.8, bins=5, label=f"Top {n_top} Trials")
            # ax.hist(all_values, alpha=0.5, bins=10, label="All Trials")
            sns.histplot(
                x=top_df[top_param].values,
                bins=n_top // 2,
                alpha=0.8,
                label=f"Top {n_top} Trials",
                stat="percent",
                ax=ax,
            )

            ax.set_xlabel(top_param)
            ax.set_ylabel("Percent")
            ax.set_title(f"Distribution of {top_param}", fontsize=14)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)
            # Create histograms for parameter distribution
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Could not create parameter distribution\n({e!s}, {top_param})",
                horizontalalignment="center",
                verticalalignment="center",
            )

    def __plot_parallel_coordinate(self, ax: Axes, study: Study) -> None:
        """Plot parallel coordinates of the given Optuna study.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot in.
        study : optuna.study.Study
            The Optuna study object.

        Returns
        -------
        None

        """
        # Determine if we're minimizing or maximizing
        direction = study.direction
        is_min = direction == StudyDirection.MINIMIZE

        # Get completed trials
        trials = study.get_trials(states=([TrialState.COMPLETE]))
        if len(trials) == 0:
            ax.set_title("Parallel Coordinate Plot (No complete trials)")
            return

        # Collect trial data
        params_data = [
            {
                "Objective": trial.value,
                "Trial": trial.number,
                **trial.params,
            }
            for trial in trials
        ]

        if not params_data:
            ax.set_title("Parallel Coordinate Plot (No parameter data)")
            return

        # Create DataFrame
        df = pd.DataFrame(params_data)

        # Exclude 'Trial' and 'Objective' columns from parameters
        param_cols = [col for col in df.columns if col not in ["Objective", "Trial"]]

        if not param_cols:
            ax.set_title("Parallel Coordinate Plot (No parameters to plot)")
            return

        # Select only numeric columns for normalization
        numeric_cols = [col for col in param_cols if pd.api.types.is_numeric_dtype(df[col])]

        # Normalize numeric parameter columns (vectorized approach)
        norm_df = df.sort_values("Objective", ascending=bool(is_min)).head(20)

        norm_df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (
            df[numeric_cols].max() - df[numeric_cols].min()
        )

        # Melt DataFrame for parallel coordinate plotting
        melted_df = norm_df.melt(
            id_vars=["Trial", "Objective"],
            value_vars=param_cols,
            var_name="Parameter",
            value_name="Value",
        )

        # Create colormap based on objective values
        obj_min = df["Objective"].min()
        obj_max = df["Objective"].max()

        # Create a categorical color map
        best_trial_idx = df["Objective"].idxmin() if is_min else df["Objective"].idxmax()
        best_trial_num = df.loc[best_trial_idx, "Trial"]

        # Create a palette where the best trial is blue and others are lighter shades
        objective_values = df.set_index("Trial")["Objective"].to_dict()

        obj_min = min(objective_values.values())
        obj_max = max(objective_values.values())
        eps = 1e-6  # Small value to prevent division errors

        # Precompute the colormap
        cmap = sns.color_palette("coolwarm", as_cmap=True)

        # Generate color mapping
        colors = {
            trial_num: (
                "darkblue"
                if trial_num == best_trial_num
                else (
                    cmap(
                        1 - (obj_val - obj_min) / (obj_max - obj_min + eps)
                        if is_min
                        else (obj_val - obj_min) / (obj_max - obj_min + eps)
                    )
                )
            )
            for trial_num, obj_val in objective_values.items()
        }

        # Plot all trials with lower opacity
        sns.lineplot(
            data=melted_df,
            x="Parameter",
            y="Value",
            hue="Trial",
            palette=colors,
            ax=ax,
            legend=False,
            alpha=0.6,  # Reduce opacity for other trials
            linewidth=0.8,  # Thinner lines for visibility
        )

        # Highlight best trial
        best_trial_df = melted_df[melted_df["Trial"] == best_trial_num]
        sns.lineplot(
            data=best_trial_df,
            x="Parameter",
            y="Value",
            color="darkblue",
            linewidth=2,
            label=f"Best Trial (# {best_trial_num})",
            ax=ax,
            zorder=3,
        )

        # Set labels and title
        ax.set_title("Parallel Coordinate Plot (Top 20)", fontsize=14)
        ax.set_xlabel("Parameter", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
        ax.set_ylabel("Normalized Value", fontsize=12)

        # Improve legend
        ax.legend(loc="upper right", fontsize=10)

        # Reduce grid line visibility
        ax.grid(alpha=0.3)

    def __plot_performance_over_time(self, df: pd.DataFrame, ax: Axes, study: Study) -> None:
        """Plot performance improvement over time.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the optimization history.
        ax : matplotlib.axes.Axes
            Axes to plot on.
        study : optuna.study.Study
            The Optuna study object.

        Returns
        -------
        None

        """
        try:
            df["datetime"] = pd.to_datetime(df["datetime"], unit="ns")
            df["runtime"] = df["datetime"].diff().dt.total_seconds().fillna(0)

            sns.scatterplot(
                data=df,
                x="number",
                y="value",
                hue="runtime",
                palette="coolwarm",
                ax=ax,
            )

            if len(df) >= 5:
                window = min(5, len(df) // 2)
                exponential_mean = df["value"].ewm(span=window, adjust=False).mean()
                sns.lineplot(
                    x=df["number"],
                    y=exponential_mean,
                    markers=True,
                    linewidth=2,
                    ax=ax,
                    label=f"{window}-Trial Rolling Median",
                )
            # Mark best trial
            best_trial_idx = (
                df["value"].idxmin()
                if study.direction == optuna.study.StudyDirection.MINIMIZE
                else df["value"].idxmax()
            )

            if not pd.isna(best_trial_idx):  # Check if we found a valid index
                best_trial = df.loc[best_trial_idx]
                sns.scatterplot(
                    x=[best_trial["number"]],
                    y=[best_trial["value"]],
                    marker="*",
                    s=700,
                    c="red",
                    label="Best Trial",
                    ax=ax,
                )

            norm = colors.Normalize(vmin=df["runtime"].min(), vmax=df["runtime"].max())
            sm = cm.ScalarMappable(cmap=colormaps["coolwarm"], norm=norm)
            sm.set_array([])

            plt.colorbar(sm, ax=ax, label="Duration (seconds)")

            ax.set_title("Optimization Progress", fontsize=16, weight="bold")
            ax.set_xlabel("Number", fontsize=12)
            ax.set_ylabel("Score", fontsize=12)
            ax.set_xticks(range(0, len(df) + 1, max(1, len(df) // 5, len(df) // 10)))

            ax.get_legend().remove()
            ax.grid(axis="y", alpha=0.4)

            sns.despine(ax=ax)

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Could not create performance over time plot\n({e!s})",
                horizontalalignment="center",
                verticalalignment="center",
            )

    def __plot_contour(self, df: pd.DataFrame, ax: Axes, study: Study) -> None:
        """Plot the contour plot of the given Optuna study.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the optimization history.
        ax : matplotlib.axes.Axes
            The axes to draw the plot in.
        study : optuna.study.Study
            The Optuna study object containing the optimization results.

        Returns
        -------
        None

        """
        ax.set_title("Contour Plot (Top 2 Parameters)", fontsize=14)

        # Only proceed if we have enough trials
        if len(df) < 10:
            ax.text(
                0.5,
                0.5,
                "Not enough trials for contour plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            print("Not enough trials for contour plot")
            return

        # Get parameter importance to identify top 2 parameters
        importances = optuna.importance.get_param_importances(study)
        importance_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        if len(importance_items) < 2:
            ax.text(
                0.5,
                0.5,
                "Need at least 2 parameters\nfor contour plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            print("Need at least 2 parameters for contour plot")
            return

        param1, param2 = importance_items[0][0], importance_items[1][0]

        # Extract parameter values
        param1_values = np.array(df[param1])
        param2_values = np.array(df[param2])
        objective_values = np.array(df["value"])

        if len(param1_values) < 5:
            ax.text(
                0.5,
                0.5,
                "Not enough data points\nwith both parameters",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            print("Not enough data points with both parameters")
            return

        # Create 2D grid for contour plot
        param1_min, param1_max = param1_values.min(), param1_values.max()
        param2_min, param2_max = param2_values.min(), param2_values.max()

        if param1_min == param1_max or param2_min == param2_max:
            ax.text(
                0.5,
                0.5,
                "Parameter range too small\nfor contour plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            print("Parameter range too small for contour plot")
            return

        # Create grid values
        grid_size = 150  # Increased resolution for smoother interpolation
        x = np.linspace(param1_min, param1_max, grid_size)
        y = np.linspace(param2_min, param2_max, grid_size)
        X, Y = np.meshgrid(x, y)

        from scipy.interpolate import griddata

        # Interpolation for contour
        Z = griddata(
            (param1_values, param2_values),
            objective_values,
            (X, Y),
            method="cubic",
            fill_value=np.nan,
        )

        # Fixing the contour not spanning the whole figure
        contour = ax.contourf(X, Y, Z, levels=15, cmap="coolwarm", extend="both")

        # Adjust layout to prevent cropping
        plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)

        # Add colorbar with proper positioning
        cbar = plt.colorbar(contour, ax=ax, fraction=0.05, pad=0.02)
        cbar.set_label("Objective Value", fontsize=12)

        # Plot sampled points
        ax.scatter(
            param1_values,
            param2_values,
            c="white",
            s=30,
            alpha=0.7,
            edgecolors="black",
        )

        # Highlight the best parameter combination
        best_idx = (
            np.argmin(objective_values)
            if study.direction == optuna.study.StudyDirection.MINIMIZE
            else np.argmax(objective_values)
        )

        ax.scatter(
            param1_values[best_idx],
            param2_values[best_idx],
            c="red",
            s=200,
            marker="*",
            edgecolors="black",
            label="Best",
        )

        ax.set_xlabel(param1, fontsize=12)
        ax.set_ylabel(param2, fontsize=12)
        ax.legend()
