import gc
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import psutil
from loguru import logger

warnings.filterwarnings("ignore")


@dataclass
class ColumnMapping:
    """Configuration class for column name mappings"""

    # Core columns - start_time and end_time are required
    start_time: str = "StartTime"
    end_time: str = "EndTime"
    start_station_id: str = "StartStationId"
    end_station_id: str = "EndStationId"

    # Optional columns that might be present
    trip_id: Optional[str] = None
    user_id: Optional[str] = None
    bike_id: Optional[str] = None
    trip_duration: Optional[str] = None  # If already calculated

    def to_dict(self) -> Dict[str, Any]:
        """Convert mapping to dictionary for logging"""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_station_id": self.start_station_id,
            "end_station_id": self.end_station_id,
            "trip_id": self.trip_id,
            "user_id": self.user_id,
            "bike_id": self.bike_id,
            "trip_duration": self.trip_duration,
        }

    def get_required_columns(self) -> List[str]:
        """Get list of required columns"""
        return [self.start_time, self.end_time, self.start_station_id, self.end_station_id]

    def get_optional_columns(self) -> List[str]:
        """Get list of optional columns (excluding None values)"""
        return [
            col
            for col in [self.trip_id, self.user_id, self.bike_id, self.trip_duration]
            if col
        ]


@dataclass
class CleaningConfig:
    """Configuration class for data cleaning parameters"""

    # Temporal thresholds
    min_trip_duration_minutes: float = 0.5
    max_trip_duration_minutes: float = 1440.0  # 24 hours
    min_round_trip_duration_minutes: float = 3.0

    # Statistical outlier detection
    outlier_method: Literal["iqr", "z_score", "isolation_forest"] = "iqr"
    z_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    isolation_forest_contamination: float = 0.1

    # Memory management
    chunk_size: int = 10000
    memory_threshold_gb: float = 8.0
    enable_memory_monitoring: bool = True

    # Column mapping configuration
    column_mapping: ColumnMapping = field(default_factory=ColumnMapping)

    # Validation
    enable_validation: bool = True
    validation_sample_size: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        return {
            "min_trip_duration_minutes": self.min_trip_duration_minutes,
            "max_trip_duration_minutes": self.max_trip_duration_minutes,
            "min_round_trip_duration_minutes": self.min_round_trip_duration_minutes,
            "outlier_method": self.outlier_method,
            "z_threshold": self.z_threshold,
            "iqr_multiplier": self.iqr_multiplier,
            "isolation_forest_contamination": self.isolation_forest_contamination,
            "chunk_size": self.chunk_size,
            "memory_threshold_gb": self.memory_threshold_gb,
            "enable_memory_monitoring": self.enable_memory_monitoring,
            "enable_validation": self.enable_validation,
            "validation_sample_size": self.validation_sample_size,
            "column_mapping": self.column_mapping.to_dict(),
        }


@dataclass
class SaveConfig:
    """Configuration for saving cleaned data"""

    format: Literal["parquet", "csv", "pickle", "feather"] = "parquet"
    compression: Literal["gzip", "zstd"] = "gzip"
    path: str | Path = "cleaned_bike_data"
    create_backup: bool = True
    include_metadata: bool = True
    partition_cols: Optional[List[str]] = None

    def get_file_extension(self) -> str:
        """Get appropriate file extension"""
        extensions = {
            "parquet": ".parquet",
            "csv": ".csv",
            "pickle": ".pkl",
            "feather": ".feather",
        }
        return extensions[self.format]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        return {
            "file_format": self.format,
            "compression": self.compression,
            "include_metadata": self.include_metadata,
            "partition_cols": self.partition_cols,
        }


class MemoryMonitor:
    """Memory monitoring utility for large dataset processing"""

    def __init__(self, threshold_gb: float = 8.0):
        self.threshold_gb = threshold_gb
        self.process = psutil.Process()

    def get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB"""
        return self.process.memory_info().rss / (1024**3)

    def check_memory_threshold(self) -> bool:
        """Check if memory usage exceeds threshold"""
        current_usage = self.get_memory_usage_gb()
        return current_usage > self.threshold_gb

    def log_memory_status(self):
        """Log current memory status"""
        current_usage = self.get_memory_usage_gb()
        available = psutil.virtual_memory().available / (1024**3)

        logger.info(f"Memory Usage: {current_usage:.2f} GB | Available: {available:.2f} GB")

        if current_usage > self.threshold_gb:
            logger.warning(
                f"Memory usage ({current_usage:.2f} GB) exceeds "
                f"threshold ({self.threshold_gb} GB)"
            )

    def force_garbage_collection(self):
        """Force garbage collection to free memory"""
        logger.info("Forcing garbage collection...")
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        self.log_memory_status()
