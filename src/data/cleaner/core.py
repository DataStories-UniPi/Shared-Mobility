from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from .models import CleaningConfig, MemoryMonitor


class DataCleaner(ABC):

    def __init__(
        self,
        df: pd.DataFrame,
        clean_config: Optional[CleaningConfig] = None,
    ):

        self.df = df.copy()
        self.original_df = df.copy()  # Keep original for comparison
        self.original_shape = df.shape
        self.clean_config = clean_config or CleaningConfig()

        # Initialize memory monitoring
        self.memory_monitor_ = (
            MemoryMonitor(self.clean_config.memory_threshold_gb)
            if self.clean_config.enable_memory_monitoring
            else None
        )

        # Tracking attributes
        self.cleaning_log_ = []

    @abstractmethod
    def clean_data(self) -> pd.DataFrame:
        """
        Execute the complete data cleaning pipeline

        This method should be overridden by subclasses to perform specific
        data cleaning tasks.

        Returns:
            Cleaned data
        """
        raise NotImplementedError

    def log_cleaning_step(self, step_name: str, removed_count: int, reason: str):
        """Log each cleaning step with enhanced details"""
        step_info = {
            "step": step_name,
            "removed_count": removed_count,
            "reason": reason,
            "remaining_rows": len(self.df),
            "removal_percentage": round(removed_count / self.original_shape[0], 2),
            "timestamp": datetime.now().isoformat(),
        }

        self.cleaning_log_.append(step_info)

        logger.info(f"🧹 {step_name}: Removed {removed_count:,} records ({reason})")
        if removed_count > 0:
            logger.info(
                f"📊 Retention: {len(self.df):,} records remaining "
                f"({len(self.df) / self.original_shape[0]:.1%})"
            )

        if self.memory_monitor_:
            self.memory_monitor_.log_memory_status()

    @abstractmethod
    def save_cleaned_data(self, custom_path: Optional[str | Path] = None):
        """
        Save cleaned data with configurable format and compression

        Args:
            custom_path: Custom save path (overrides config)

        Returns:
            None
        """
        raise NotImplementedError
