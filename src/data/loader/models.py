"""
Configuration and output structures for bike trip data loader.

Example:
    config = LoaderConfig(
        chunk_size=5000,
        max_workers=8,
        parse_dates=True
    )

    result = LoadResult(
        data=df,
        month="2021-01",
        rows_loaded=15000,
        success=True
    )
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import pandas as pd


class S3DataLoaderError(Exception):
    """Base exception for errors in the S3 data loader."""

    pass


class S3TransientError(S3DataLoaderError):
    """Exception raised for transient errors that may be retried."""

    pass


class S3PermanentError(S3DataLoaderError):
    """Exception raised for permanent errors that should not be retried."""

    pass


@dataclass
class LoaderConfig:
    """Configuration for bike trip data loader operations."""

    chunk_size: int = field(default=10000)
    max_workers: int = field(default=4)
    cache_enabled: bool = field(default=True)
    parse_dates: bool = field(default=True)
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = field(default="numpy_nullable")
    starttime_columns: str | List[str] = field(default_factory=lambda: ["starttime"])
    yearmonth_column: str = field(default="yearmonth")

    def __post_init__(self):
        """Validate configuration values."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")


@dataclass
class LoadResult:
    """Result of a data loading operation."""

    data: Optional[pd.DataFrame]
    month: str
    rows_loaded: int
    success: bool
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None


@dataclass
class DataSummary:
    """Summary information for a month's data sources."""

    month: str
    source_count: int
    csv_count: int
    estimated_rows: Optional[int] = None
