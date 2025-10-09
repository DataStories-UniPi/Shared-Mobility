import gc
from itertools import product
from statistics import mean, stdev
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from utils.helper import create_crowd_levels, create_mask, load_hparams, split_X_y
from utils.models import MLDataConfig, TaskType

from .core import Estimator, timer
from .importance import FeatureImportanceAnalyzer
from .models import BenchmarkResult, ImportanceType, OutputFormat, Split
from .performance import PerformanceBenchmarker


class ModelBenchmarker:
    """
    Main benchmarking orchestrator that integrates timing, feature importance,
    and performance analysis
    """

    def __init__(
        self,
        model_factory,
        target_col: str | List[str],
        group_col: Optional[str] = None,
        task_type: TaskType = TaskType.REGRESSION,
        *,
        n_runs: int = 3,
        warmup_runs: int = 1,
        data_loader: Optional[Callable] = None,
        per_output_model: bool = False,
    ):
        """
        Initialize comprehensive model benchmarker

        Parameters
        ----------
        model_factory : Callable
            Factory function/class for creating models
        best_params : Dict[str, Any]
            Best hyperparameters for the model
        target_col : str
            Target column name
        task_type : str, default='reg'
            Type of ML task ('reg' or 'clf')
        n_runs : int, default=3
            Number of timing runs per combination
        warmup_runs : int, default=1
            Number of warmup runs (not timed)
        data_loader : Optional[Callable], default=None
            Custom data loading function with signature (identifier, horizon, split, **kwargs)
        """
        self.model_factory = model_factory
        self.target_col = target_col
        self.group_col = group_col
        self.task_type = task_type
        self.n_runs = n_runs
        self.warmup_runs = warmup_runs
        self.data_loader = data_loader
        self.per_output_model = per_output_model

        # Initialize analyzers
        self.results = {}
        self.feature_analyzer: Optional[FeatureImportanceAnalyzer] = None
        self.performance_benchmarker: Optional[PerformanceBenchmarker] = None

        self._data_cache: Dict[Tuple[str, int], Dict[str, Tuple]] = {}

    def setup_feature_analysis(
        self,
        importance_types: List[ImportanceType] = [ImportanceType.WEIGHT],
        output_format: str = OutputFormat.RAW,
        top_k: Optional[int] = None,
    ):
        """
        Setup feature importance analysis

        Parameters
        ----------
        importance_types : List[str], default=['weight']
            Types of importance to compute
        output_format : str, default='raw'
            Format for importance values
        top_k : Optional[int], default=None
            Number of top features to return
        """
        self.feature_analyzer = FeatureImportanceAnalyzer(
            task_type=self.task_type,
            target_col=self.target_col,
            importance_types=importance_types,
            output_format=output_format,
            top_k=top_k,
        )

    def setup_performance_analysis(
        self,
        regression_metrics: List[str] = ["mae", "rmse", "mape", "r2"],
        classification_metrics: List[str] = ["accuracy", "f1", "precision", "recall"],
        quantiles: List[float] = [0.25, 0.5, 0.75],
        group_col: Optional[str] = None,
    ):
        """
        Setup performance benchmarking

        Args:
            regression_metrics: Regression metrics to compute
            classification_metrics: Classification metrics to compute
            quantiles: Quantiles to compute for each metric
            group_col (optional): Column name for group-level analysis
        """
        self.performance_benchmarker = PerformanceBenchmarker(
            task_type=self.task_type,
            target_col=self.target_col,
            regression_metrics=regression_metrics,
            classification_metrics=classification_metrics,
            quantiles=quantiles,
            group_col=group_col,
        )

    def _create_model(self, **hparams) -> Estimator:
        """Create a fresh model instance"""
        factory = self.model_factory(
            self.task_type,
            num_output=1,
            per_output=self.per_output_model,
        )
        params = {
            "early_stopping_rounds": 10,
            "verbosity": 1,
            "device": "cuda",
            **hparams,
        }

        if self.task_type == TaskType.REGRESSION:
            params.update(
                {
                    "objective": "reg:squarederror",
                    "eval_metric": ["rmse"],
                }
            )
        elif self.task_type == TaskType.CLASSIFICATION:
            params.update(
                {
                    "objective": "multi:softmax",
                    "num_class": 3,
                    "eval_metric": "mlogloss",
                }
            )
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        return factory.build_model("xgb", estimator_params=params)

    def _load_data(
        self,
        identifier: str,
        horizon: int,
        split: Optional[Split] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.Series] | pd.DataFrame:
        """
        Load data using custom loader or default implementation

        Parameters
        ----------
        identifier : str
            Data identifier (e.g., city name)
        horizon : int
            Forecast horizon
        split : str
            Data split ('train', 'validation', 'test')

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Features and target data
        """
        key = (identifier, horizon)
        if key not in self._data_cache:
            self._data_cache[key] = {}

        if split not in self._data_cache[key]:
            logger.debug(f"{split} not found in cache, loading data...")
            if self.data_loader is not None:

                # At this point we are sure that `window` and `suffix` are defined
                config = MLDataConfig(
                    fh=horizon,
                    window=kwargs["window"],
                    dataset=identifier,
                    suffix=kwargs["suffix"],
                    target=self.target_col,
                )
                data = self.data_loader(
                    config=config,
                    prefix=kwargs.get("prefix", None),
                    dropna=True,
                    return_X_y=False,
                )
            else:
                # TODO: Implement default data loader
                logger.warning("No data loader specified, using default implementation")
                raise NotImplementedError

            self._data_cache[key][split] = data
        return self._data_cache[key][split]

    def _benchmark_single_run(
        self,
        identifier: str,
        horizon: int,
        X_train,
        y_train,
        X_test,
        y_test,
        train_limit: Optional[int] = None,
    ) -> Tuple[Optional[Estimator], BenchmarkResult]:
        """
        Benchmark a single identifier-horizon combination

        Parameters
        ----------
        identifier : str
            Data identifier
        horizon : int
            Forecast horizon
        train_limit : int
            Maximum training samples

        Returns
        -------
        BenchmarkResult
            Benchmark result with timing and prediction data
        """
        try:
            # Limit training size if specified
            if train_limit and len(X_train) > train_limit:
                logger.debug(f"Reducing training size to {X_train.shape[0]:,} samples")
                X_train = X_train.iloc[:train_limit]
                y_train = y_train.iloc[:train_limit]

            if self.task_type == TaskType.CLASSIFICATION:
                logger.debug("Building class labels from demand values")

                for col in self.target_col:

                    # Create class labels
                    out, bins = create_crowd_levels(y_train, 3, col, self.group_col)

                    # Store class labels
                    y_train[col] = pd.concat(out.values()).to_frame()

                    # Mask test data with the same bins
                    y_test[col] = create_mask(y_test, bins, col)

                    logger.debug(f"Class labels built for {col}")
                    logger.debug(f"{y_train[col].head()}")

            # Load hyperparameters
            hparams = load_hparams(identifier, self.task_type, horizon)

            # Create fresh model
            model = self._create_model(**hparams)

            # Time training
            with timer() as train_timer:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_train, y_train)],
                    verbose=25,
                )
            train_time = train_timer()

            pred_kwargs = {}

            # Get best iteration if available
            if hasattr(model, "best_iteration"):
                best_iteration = model.best_iteration
                logger.debug(f"Best iteration: {best_iteration}")
                pred_kwargs = {"iteration_range": (0, best_iteration)}

            # Time prediction and store results
            with timer() as pred_timer:
                y_pred = model.predict(X_test, **pred_kwargs)

                if self.task_type == TaskType.REGRESSION:
                    np.maximum(y_pred, 0)
                    y_pred = np.round(y_pred).astype(int)
            predict_time = pred_timer()

            return model, BenchmarkResult(
                identifier=identifier,
                horizon=horizon,
                target_col=self.target_col,
                train_time=train_time,
                predict_time=predict_time,
                total_time=train_time + predict_time,
                train_size=len(X_train),
                test_size=len(X_test),
                predictions=y_pred,
                actuals=y_test.values if hasattr(y_test, "values") else y_test,
            )

        except Exception as e:
            return None, BenchmarkResult(
                identifier=identifier,
                horizon=horizon,
                target_col=self.target_col,
                error=str(e),
            )
        finally:
            # Clean up memory
            gc.collect()

    def benchmark_combinations(
        self,
        identifiers: List[str],
        horizons: List[int],
        *,
        include_feature_importance: bool = False,
        include_performance_metrics: bool = False,
        train_limit: int | None = None,
        **load_kwargs,
    ) -> Dict[str, Any]:
        """
        Comprehensive benchmarking across identifier-horizon combinations

        Parameters
        ----------
        identifiers : List[str]
            List of data identifiers to test
        horizons : List[int]
            List of forecast horizons to test
        train_limit : int, default=2000000
            Maximum training samples per run
        include_feature_importance : bool, default=False
            Whether to compute feature importance analysis
        include_performance_metrics : bool, default=False
            Whether to compute performance metrics

        Returns
        -------
        Dict[str, Any]
            Comprehensive results including timing, importance, and performance data
        """
        model_runs = self.n_runs + self.warmup_runs
        total_runs = len(identifiers) * len(horizons) * model_runs
        logger.info(
            f"Benchmarking {len(identifiers)} identifiers × {len(horizons)} horizons"
            f" × {model_runs} runs - Total runs: {total_runs}"
        )

        # Store models for feature importance analysis
        trained_models = {} if include_feature_importance else None
        feature_names = None
        X_test_data: Dict[Tuple[str, int], pd.DataFrame] = {}

        # Extract windows and process them for compatibility
        windows = load_kwargs.pop("windows", 1)
        if isinstance(windows, int):
            windows = [windows] * len(identifiers) * len(horizons)
        elif isinstance(windows, list):
            if len(windows) != len(identifiers) * len(horizons):
                raise ValueError(
                    "`windows` must have the same length as identifiers × horizons"
                )

        # Run benchmark
        for i, (identifier, horizon) in enumerate(product(identifiers, horizons)):
            logger.info(f"📍 Testing {identifier=} with {horizon=}")

            # Load data
            kwargs = {
                "window": windows[i],
                "suffix": load_kwargs.get("suffix", ""),
            }
            data = self._load_data(identifier, horizon, **kwargs)

            X_train, y_train, X_test, y_test = split_X_y(
                data.drop(columns=self.target_col),
                data[self.target_col],
                test_size=0.1,
            )

            logger.debug(f"Data sizes: {X_train.shape=}  | {X_test.shape=}")
            logger.debug(f"Target sizes: {y_train.shape=}| {y_test.shape=}")

            # Warmup runs (not timed)
            for warmup in range(self.warmup_runs):
                logger.info(f"Warmup run [{warmup + 1}/{self.warmup_runs}]")
                _, result = self._benchmark_single_run(
                    identifier,
                    horizon,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    train_limit,
                )
                if result.error:
                    logger.info(f"❌ Error during warmup: {result.error}")
                    break

            # Actual benchmark runs
            run_times = []
            for run in range(self.n_runs):
                logger.info(f"Benchmark run [{run + 1}/{self.n_runs}]")
                model, result = self._benchmark_single_run(
                    identifier,
                    horizon,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    train_limit,
                )

                logger.debug(f"{identifier=} | {horizon=}")
                if (identifier, horizon) not in self.results:
                    self.results[(identifier, horizon)] = []
                self.results[(identifier, horizon)].append(result)

                if result.error:
                    logger.info(f"❌ Error during run {run + 1}: {result.error}")
                    break

                run_times.append(result.total_time)
                logger.info(
                    f"✅ Total time: {result.total_time:.2f}s "
                    f"(train: {result.train_time:.2f}s, pred: {result.predict_time:.4f}s)"
                )

                # Store model and data for analysis (only from last successful run)
                if run == self.n_runs - 1:  # Last run
                    if include_feature_importance:
                        if self.feature_analyzer is None:
                            logger.warning(
                                "Feature Importance Analyzer not initialized. Skipping..."
                            )
                        else:
                            trained_models[(identifier, horizon)] = model

                        if feature_names is None:
                            feature_names = list(X_test.columns)

                    if include_performance_metrics:
                        if self.performance_benchmarker is None:
                            logger.warning(
                                "Performance Benchmarker not initialized. Skipping..."
                            )
                        else:
                            X_test_data[(identifier, horizon)] = X_test
            self.clear_cache()

            # Summary statistics for this combination
            if run_times:
                avg_time = mean(run_times)
                std_time = stdev(run_times) if len(run_times) > 1 else 0
                logger.info(f"📊 Average: {avg_time:.2f}s ± {std_time:.2f}s")

        # Compile results
        results_dict = {
            "timing_summary": self.get_summary_dataframe(),
            "raw_results": self.results,
        }

        # Feature importance analysis
        if include_feature_importance and self.feature_analyzer and trained_models:
            logger.info("🔍 Computing feature importance analysis")
            importance_results = self.feature_analyzer.evaluate(trained_models, feature_names)
            results_dict["feature_importance"] = importance_results

            # Generate importance plots
            if importance_results:
                _, plots = self.feature_analyzer.plot_importance(importance_results)
                results_dict["importance_plots"] = plots

        # Performance metrics analysis
        if include_performance_metrics and self.performance_benchmarker:
            logger.info("📊 Computing performance metrics")
            performance_results = self.performance_benchmarker.evaluate(
                self.results, X_test_data
            )
            results_dict["performance_metrics"] = performance_results

            # Generate performance plots
            if not performance_results.empty:
                perf_fig = self.performance_benchmarker.plot_performance(performance_results)
                heatmap_fig = self.performance_benchmarker.create_heatmap(performance_results)
                results_dict["performance_plots"] = {
                    "distribution": perf_fig,
                    "heatmap": heatmap_fig,
                }

        return results_dict

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Convert timing results to summary DataFrame

        Returns
        -------
        pd.DataFrame
            Summary of timing results with statistics
        """
        if not self.results:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(
            [
                {
                    "identifier": r.identifier,
                    "horizon": r.horizon,
                    "run_id": run_id + 1,
                    "target_col": r.target_col,
                    "train_time": r.train_time,
                    "predict_time": r.predict_time,
                    "total_time": r.total_time,
                    "train_size": r.train_size,
                    "test_size": r.test_size,
                    "error": r.error,
                }
                for result_list in self.results.values()
                for run_id, r in enumerate(result_list)
            ]
        )

        # Group by combination and calculate statistics
        summary = df.groupby(["identifier", "horizon"]).agg(
            {
                "train_time": ["mean", "std"],
                "predict_time": ["mean", "std"],
                "total_time": ["mean", "std"],
                "train_size": "first",
                "test_size": "first",
                "error": lambda x: x.dropna().iloc[0] if x.dropna().any() else None,
            }
        )

        # Flatten column names
        summary.columns = ["_".join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()

        return summary

    def clear_cache(self):
        """Clear the data cache to free memory."""
        self._data_cache.clear()
        logger.debug("Cache cleared")
