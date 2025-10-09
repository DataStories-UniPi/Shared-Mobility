from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils.validation import check_is_fitted

from config.constants import TEMPORAL_COLUMNS, TIME_COLUMN

from .models import ForecastConfig, PowerTransformConfig, TargetEncoderConfig


def create_graph_representation(config: ForecastConfig):
    """
    Create pipeline for extracting graph features.

    Args:
        config: Forecast configuration containing city of interest.

    Returns:
        Pipeline with graph feature extractor.

    Raises:
        ValueError: If city is not specified in the configuration.
    """
    if config.city is not None:
        from components import GraphExtractor

        return Pipeline([("graph_features", GraphExtractor(config.city))])
    raise ValueError("City must be specified for graph feature extraction.")


def create_time_preprocessor(config: ForecastConfig):
    """Create time preprocessing pipeline."""
    from copy import deepcopy

    try:
        from components import RBFTransformer, TimeExtractor
    except ImportError as e:
        raise ImportError(f"Required components not available: {e}")

    calendar_extractor = TimeExtractor(
        deepcopy(config.time_features),
        config.time_periods,
        config.transit_patterns,
        holiday_country=config.country_code,
    )

    if config.time_periods:
        config.time_features.append("time_period")
        config.num_kernels.append(2)
        config.input_ranges.append((0, 4))

    rbf_encoder = ColumnTransformer(
        transformers=[
            (
                "rbf",
                RBFTransformer(
                    config.time_features,
                    config.num_kernels,
                    config.input_ranges,
                ),
                config.time_features,
            )
        ],
        sparse_threshold=0,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        [
            ("extract", calendar_extractor),
            ("rbf", rbf_encoder),
        ]
    )


def create_temporal_preprocessor(config: ForecastConfig):
    """Create temporal feature engineering pipeline."""
    try:

        from components import (
            DerivativeTransformer,
            FourierTransformer,
            TemporalExtractor,
            TrafficAdjuster,
        )

    except ImportError as e:
        raise ImportError(f"Required components not available: {e}")

    extractor = TemporalExtractor(config.lags, config.windows, config.rolling_stats)
    fourier = FourierTransformer(
        n_harmonics=config.fourier_harmonics,
        smooth=True,
        window=config.fourier_window,
    )

    transformer_list = [
        ("extract", extractor),
        ("derivative", DerivativeTransformer(orders=config.diff_orders)),
        ("fourier", fourier),
    ]
    if config.quantiles is not None:
        adjuster = TrafficAdjuster(quantiles=config.quantiles, use_diff=config.use_diff)
        transformer_list.append(("adjust", adjuster))

        logger.debug("Using quantile adjustment")

    return FeatureUnion(transformer_list)


def make_exclusive_selector(selector_fn, exclude=[]):
    return lambda X: [col for col in selector_fn(X) if col not in exclude]


class DemandForecaster(TransformerMixin, BaseEstimator):

    def __init__(
        self,
        config: ForecastConfig,
        scaler: Optional[TransformerMixin] = None,
        numeric_columns: Optional[Iterable[str] | selector] = None,
        chunk_size: int = 10000,  # For memory management
    ):
        self.config = config
        self.scaler = scaler
        self.numeric_columns = numeric_columns or self._get_columns(config=None)
        self.chunk_size = chunk_size

        # State management
        self._pipeline = None
        self._feature_names = None

    def _build_graph_representation(self):
        """Build the graph representation of the model."""
        return create_graph_representation(self.config)

    def _build_preprocessor(self):
        """Build the preprocessing pipeline with error handling."""

        from components import GroupTransformer

        time_preprocessor = create_time_preprocessor(self.config)
        temporal_preprocessor = create_temporal_preprocessor(self.config)

        preprocessor = ColumnTransformer(
            [
                ("time", time_preprocessor, [TIME_COLUMN]),
                ("temporal", temporal_preprocessor, [*TEMPORAL_COLUMNS]),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

        return GroupTransformer(preprocessor, self.config.group_col)

    def _build_postprocessor(self):
        """Build the postprocessing pipeline."""
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer

        from components import TargetEncoder

        post_transformers = []
        power_columns = None
        target_columns = None

        # Add PowerTransformer if enabled
        if self.config.power_transform_config.enabled:
            power_transformer = PowerTransformer(
                method=self.config.power_transform_config.method,
                standardize=self.config.power_transform_config.standardize,
                copy=self.config.power_transform_config.copy,
            )

            # Determine columns for power transformation
            power_columns = self._get_columns(self.config.power_transform_config)

            if power_columns is not None:
                power_pipeline = Pipeline([("power", power_transformer)])

                if self.config.power_transform_config.standardize:
                    power_pipeline.steps.append(("scale", self.scaler))

                post_transformers.append(("target_pipeline", power_pipeline, power_columns))
            else:
                logger.warning("PowerTransformer enabled but no valid columns specified")

        # Add TargetEncoder if enabled
        if self.config.target_encoder_config.enabled:
            target_encoder = TargetEncoder(
                prior_weight=self.config.target_encoder_config.prior_weight,
                min_samples=self.config.target_encoder_config.min_samples,
                smoothing_strategy=self.config.target_encoder_config.method,
            )

            # Determine columns for power transformation
            target_columns = self._get_columns(self.config.target_encoder_config)

            if target_columns is not None:
                target_pipeline = Pipeline([("target", target_encoder)])

                if self.config.target_encoder_config.standardize:
                    target_pipeline.steps.append(("scale", self.scaler))

                post_transformers.append(("target_pipeline", target_pipeline, target_columns))

            else:
                logger.warning("TargetEncoder enabled but no valid columns specified")

        if self.scaler is not None:
            # Apply scaler only to numeric columns that are not transformed by
            # power or target encoders
            exclude_columns = []
            if power_columns and not isinstance(power_columns, selector):
                exclude_columns.extend(power_columns)
            if target_columns and not isinstance(target_columns, selector):
                exclude_columns.extend(target_columns)

            scaler_columns = make_exclusive_selector(
                self.numeric_columns,
                exclude=exclude_columns,
            )
            post_transformers.append(("scaler", self.scaler, scaler_columns))

        # Add encoding transformers if configured
        if hasattr(self.config, "encoding_config"):
            encoding_config = self.config.encoding_config

            # Add ordinal encoder if columns are specified
            if encoding_config.ordinal_columns:
                ordinal_encoder = OrdinalEncoder(
                    handle_unknown=encoding_config.handle_unknown_ordinal,
                    unknown_value=encoding_config.unknown_value,
                )
                post_transformers.append(
                    ("ordinal", ordinal_encoder, encoding_config.ordinal_columns)
                )

            # Add onehot encoder if columns are specified
            if encoding_config.onehot_columns:
                onehot_encoder = OneHotEncoder(
                    handle_unknown=encoding_config.handle_unknown_onehot,
                    categories="auto",
                    sparse_output=False,  # Pandas output does not support sparse data
                    drop="if_binary",
                    max_categories=encoding_config.max_categories,
                )
                post_transformers.append(
                    ("onehot", onehot_encoder, encoding_config.onehot_columns)
                )

        if not post_transformers:
            # Return identity transformer if no postprocessing needed
            from sklearn.preprocessing import FunctionTransformer

            return FunctionTransformer(validate=False)

        return ColumnTransformer(
            transformers=post_transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

    def _get_columns(
        self,
        config: Optional[PowerTransformConfig | TargetEncoderConfig] = None,
    ) -> selector | Iterable[str] | None:
        """
        Determine which columns should be power transformed.

        Returns
        -------
        columns : selector, list, or None
            Columns to apply power transformation to
        """
        if config is None:
            # Use a function instead of lambda for better serialization with joblib
            def get_numeric_columns(X):
                if hasattr(X, "columns"):
                    return [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
                return []

            return selector(dtype_include="number")

        # If explicitly specified in power_transform_config config, use those
        if config.columns is not None:
            return config.columns

        # Otherwise, use numeric columns if available
        if self.numeric_columns is not None:
            return self.numeric_columns

        # For box-cox, we need to be more careful about column selection
        if isinstance(config, PowerTransformConfig) and config.method == "box-cox":
            logger.warning(
                "Box-Cox method requires positive values. "
                "Consider using yeo-johnson or specifying columns explicitly."
            )
            return None

    def _validate_power_transform_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and preprocess data for power transformation.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to validate

        Returns
        -------
        pd.DataFrame
            Validated data, potentially with adjustments for power transform
        """
        if not self.config.power_transform_config.enabled:
            return X

        X_copy = X.copy() if self.config.power_transform_config.copy else X

        # Get columns that will be power transformed
        power_columns = self._get_columns(self.config.power_transform_config)

        if power_columns is None:
            return X_copy

        # Handle Box-Cox specific requirements
        if self.config.power_transform_config.method == "box-cox":
            if isinstance(power_columns, selector):
                # Get actual column names from selector
                numeric_cols = power_columns(X_copy)
            else:
                numeric_cols = power_columns

            for col in numeric_cols:
                if col in X_copy.columns:
                    min_val = X_copy[col].min()
                    if min_val <= 0:
                        logger.warning(
                            f"Column '{col}' has non-positive values (min: {min_val}). "
                            "Box-Cox requires positive values. Consider switching to"
                            "yeo-johnson."
                        )

                        raise ValueError(
                            f"Box-Cox transformation cannot be applied to column '{col}' "
                            f"with non-positive values. Use yeo-johnson method instead."
                        )

        return X_copy

    @property
    def pipeline_(self):
        """Lazy property for pipeline access."""
        return self._build_pipeline()

    def _build_pipeline(self):
        """Lazy pipeline construction with proper error handling."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            graph_representation = self._build_graph_representation()
            preprocessor = self._build_preprocessor()
            postprocessor = self._build_postprocessor()

            steps = [
                ("graph", graph_representation),
                ("preprocessor", preprocessor),
                ("postprocessor", postprocessor),
            ]

            # Create pipeline with caching support
            self._pipeline = Pipeline(
                steps=steps,
                verbose=True,  # Use a default value for now
            )

            return self._pipeline

        except Exception as e:
            logger.error(f"Pipeline construction failed: {e}")
            raise

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        """
        Fit the pipeline with proper validation and chunked processing.
        """
        # Input validation
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if TIME_COLUMN not in X.columns:
            raise ValueError(f"X must contain {TIME_COLUMN} column")

        # ? Memory-efficient processing for large datasets
        # if len(X) > self.chunk_size:
        #     logger.info(f"Large dataset detected ({len(X)} rows). Processing in chunks.")
        #     return self._fit_chunked(X, y)

        try:
            self.pipeline_.fit(X, y)
            self._is_fitted = True

            # Cache feature names
            if hasattr(self.pipeline_.named_steps["postprocessor"], "get_feature_names_out"):
                self._feature_names = self.pipeline_.named_steps[
                    "postprocessor"
                ].get_feature_names_out()

            return self

        except Exception as e:
            logger.error(f"Fitting failed: {e}")
            self._is_fitted = False
            raise

    # ? Fit pipeline using chunked processing for memory efficiency.
    # def _fit_chunked(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
    #     # Implementation would depend on specific chunking strategy
    #     # This is a placeholder for the chunked fitting logic
    #     raise NotImplementedError("Chunked fitting not yet implemented")

    def transform(self, X: pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """Transform data without prediction."""
        check_is_fitted(self, "_is_fitted")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        result = self.pipeline_.transform(X)

        return result

    def get_feature_names_out(self) -> np.ndarray:
        """Get output feature names."""
        check_is_fitted(self, "_is_fitted")

        if self._feature_names is None:
            raise AttributeError("Feature names not available")

        return self._feature_names

    def save(self, path: Path):
        """Save the fitted pipeline with all components for reproducibility."""
        check_is_fitted(self, "_is_fitted")

        import dill

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as file:
            dill.dump(
                {
                    "pipeline": self._pipeline,
                    "config": self.config,
                    "feature_names": self._feature_names,
                },
                file,
            )

    def load(self, path: Path):
        """Load a saved pipeline."""
        import dill

        with path.open("rb") as f:
            data = dill.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"The file {path} does not contain the expected payload.")

        self._pipeline = data["pipeline"]
        self.config = data["config"]
        self._feature_names = data.get("feature_names", getattr(self, "_feature_names", None))

        self._is_fitted = True

        return self

    def delete_cache(self):
        """
        Delete the cache directory.

        This method will delete the cache directory specified in the configuration.
        It is useful for cleaning up temporary files and memory.
        """
        self.config.cache_config.delete_cache_directory()
