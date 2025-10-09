import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple

import numpy as np
from joblib import Memory
from loguru import logger
from sklearn.compose import make_column_selector as selector

from config.constants import GROUP_COLUMN


@dataclass
class EncodingConfig:
    """Configuration for encoding strategies."""

    ordinal_columns: Optional[Iterable[str] | selector] = None
    onehot_columns: Optional[Iterable[str] | selector] = None

    # Ordinal encoding
    handle_unknown_ordinal: Literal["error", "use_encoded_value"] = "use_encoded_value"
    unknown_value: Optional[float] = np.nan
    encoded_missing_value: int = -1

    # One-hot encoding
    handle_unknown_onehot: Literal["infrequent_if_exist", "error", "ignore"] = "error"
    max_categories: Optional[int] = None


@dataclass
class PowerTransformConfig:
    """Configuration for PowerTransformer."""

    enabled: bool = False
    method: Literal["yeo-johnson", "box-cox"] = "yeo-johnson"
    standardize: bool = True
    copy: bool = True
    columns: Optional[Iterable[str] | selector] = None


@dataclass
class TargetEncoderConfig:
    """Configuration for TargetEncoder."""

    enabled: bool = False
    prior_weight: float = 10.0
    min_samples: int = 1
    standardize: bool = True
    method: Literal["additive", "exponential"] = "additive"
    columns: Optional[Iterable[str] | selector] = None


@dataclass
class CacheConfig:
    """Configuration class for cache management."""

    enabled: bool = False
    directory: Optional[Path] = None
    size_limit: str = "1G"  # joblib Memory size limit
    verbose: int = 0  # Verbosity level for cache operations

    _memory: Optional[Memory] = field(default=None, init=False, repr=False)
    _temp_dir: Optional[str] = field(default=None, init=False, repr=False)

    def setup_memory(self) -> Optional[Memory]:
        """Setup joblib Memory instance."""
        if not self.enabled:
            return None

        try:
            cache_dir = Path(self.directory or "cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using cache directory: {cache_dir}")

            self._memory = Memory(
                location=str(cache_dir),
                verbose=self.verbose,
                bytes_limit=self.size_limit,
            )
        except RuntimeError as re:
            raise RuntimeError("Cache directory creation failed") from re
        except Exception as e:
            logger.warning(f"Failed to create cache directory: {e}")
            return None

        return self._memory

    def delete_cache_directory(self) -> bool:
        """Delete the cache directory."""
        if not self.directory or not Path(self.directory).exists():
            return False

        try:
            shutil.rmtree(self.directory)
            logger.info(f"Deleted cache directory: {self.directory}")

            # Reset memory and temp dir
            self._memory = None
            self._temp_dir = None if self._temp_dir != self.directory else None

            return True
        except Exception as e:
            logger.error(f"Failed to delete cache directory: {e}")
            return False

    def get_cache_info(self) -> dict:
        """
        Get information about the current cache configuration and usage.

        Returns
        -------
        dict
            Dictionary containing cache information
        """
        info = {
            "cache_enabled": self._memory is not None,
            "cache_directory": self.directory,
            "cache_size_mb": 0,
            "cache_files": 0,
            "size_limit": self.size_limit,
            "verbose": self.verbose,
        }

        if self._memory is not None:
            cache_dir = Path(self._memory.location)
            info["cache_directory"] = str(cache_dir)

            if cache_dir.exists():
                try:
                    cache_files = list(cache_dir.rglob("*"))
                    info["cache_files"] = len([f for f in cache_files if f.is_file()])

                    total_size = sum(f.stat().st_size for f in cache_files if f.is_file())
                    info["cache_size_mb"] = round(total_size / (1024 * 1024), 2)

                except Exception as e:
                    logger.warning(f"Failed to get cache size info: {e}")

        return info

    def clear_cache(self) -> bool:
        """
        Clear all cached data without deleting the cache directory.

        Returns
        -------
        bool
            True if cache cleared successfully, False otherwise
        """
        if self._memory is None:
            logger.warning("No cache configured")
            return False

        try:
            self._memory.clear(warn=False)
            logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

    def __del__(self):
        """Cleanup temp directory on destruction."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary cache: {e}")


@dataclass
class ForecastConfig:
    """Configuration class to encapsulate forecaster parameters."""

    city: Optional[str] = None
    group_col: str | List[str] = GROUP_COLUMN
    time_features: List[str] = field(default_factory=lambda: ["hour", "month"])
    num_kernels: List[int] = field(default_factory=lambda: [12, 24])
    input_ranges: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 23), (1, 12)])
    time_periods: bool = False
    transit_patterns: bool = False
    country_code: Optional[str] = None
    lags: List[int] = field(default_factory=list)
    windows: List[int] = field(default_factory=list)
    rolling_stats: List[str] = field(default_factory=list)
    fourier_harmonics: int = 30
    fourier_window: int = 12
    quantiles: Optional[List[float]] = None
    use_diff: bool = False
    diff_orders: List[int] = field(default_factory=list)

    power_transform_config: PowerTransformConfig = field(default_factory=PowerTransformConfig)
    target_encoder_config: TargetEncoderConfig = field(default_factory=TargetEncoderConfig)

    # Encoding configuration as a nested config
    encoding_config: EncodingConfig = field(default_factory=EncodingConfig)

    # Cache configuration as a nested config
    cache_config: CacheConfig = field(default_factory=CacheConfig)

    def __post_init__(self):
        """Validate configuration parameters."""
        if not self.time_features:
            raise ValueError("time_features cannot be empty")

        if not self.group_col:
            raise ValueError("group_col must be specified")

        if self.num_kernels and len(self.num_kernels) != len(self.input_ranges):
            raise ValueError("num_kernels and input_ranges must have same length")

        if not self.lags:
            raise ValueError("lags cannot be empty")

        if not self.windows:
            raise ValueError("windows cannot be empty")

        if not self.rolling_stats:
            raise ValueError("rolling_stats cannot be empty")

        if not self.diff_orders:
            raise ValueError("diff_orders cannot be empty")
