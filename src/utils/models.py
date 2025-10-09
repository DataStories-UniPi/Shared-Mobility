from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import List, Optional, Protocol

from config import paths
from config.constants import TARGET_COLUMN

TargetType = str | List[str]
IndexType = str | List[str]


class Trainable(Protocol):

    def fit(self, X, y, **kwargs): ...

    def predict(self, X, **kwargs): ...


class TaskType(StrEnum):
    """Enumeration for task types"""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class FileFormat(StrEnum):
    """Enumeration for file formats"""

    CSV = "csv"
    PARQUET = "parquet"
    SQL = "sql"
    JSON = "json"
    FEATHER = "feather"


@dataclass(kw_only=True)
class MLDataConfig:
    """Configuration for loading processed data."""

    fh: int
    window: int
    dataset: str
    suffix: str
    target: TargetType = field(default_factory=lambda: TARGET_COLUMN)
    extension: Optional[str] = field(default="parquet.gzip")

    def __post_init__(self):
        self.base_dir: Path = paths.PROCESSED_DATA_DIR / self.dataset


EstimatorType = str | Trainable | Callable[..., Trainable]
