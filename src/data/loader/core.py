"""
Abstract base classes and protocols for bike trip data loading.

This module defines the core interfaces for data loading components that can handle
various source types (directories, ZIP files) and provide chunked or bulk loading
capabilities.

Example:
    class BikeDataLoader(DataLoader):
        def load_data(self) -> pd.DataFrame:
            # Implementation here
            pass
"""

import re
from abc import ABC, ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

from .models import LoaderConfig, LoadResult, S3PermanentError, S3TransientError


class DataLoader(ABC):
    """
    Abstract base class for bike trip data loaders.

    Provides the interface for discovering, loading, and processing bike trip data
    from various source types with support for chunked processing.
    """

    def __init__(self, root_dir: Path, config: Optional[LoaderConfig] = None) -> None:
        """
        Initialize the data loader.

        Args:
            root_dir: Root directory containing data sources
            config: Configuration object (uses defaults if None)

        Raises:
            FileNotFoundError: If root directory doesn't exist
            NotADirectoryError: If root_dir is not a directory
        """
        if not root_dir.exists():
            logger.warning(f"Root directory not found: {root_dir} . Creating directory.")
            root_dir.mkdir(parents=True)
        if not root_dir.is_dir():
            logger.error(f"Root path is not a directory: {root_dir}")
            raise NotADirectoryError

        self.root_dir = root_dir
        self.config = config or LoaderConfig()

        # Cache for loaded data - only stores if cache_enabled
        self._data_cache: Dict[str, pd.DataFrame] = {}

    @abstractmethod
    def load_data(self) -> Dict[str, LoadResult]:
        """Load data from sources and yield chunks as Pandas DataFrames."""
        raise NotImplementedError

    @abstractmethod
    @lru_cache
    def get_data_summary(self) -> pd.DataFrame:
        """Get summary information about discovered data sources."""
        raise NotImplementedError


class S3DataLoader(DataLoader, metaclass=ABCMeta):
    """
    Abstract base class for bike data loaders that fetch data from S3 URLs.

    This class provides common functionality for downloading and processing
    bike trip data from S3 sources with configurable URL patterns.
    """

    def __init__(
        self,
        root_dir: Path,
        bucket_name: Optional[str],
        config: Optional[LoaderConfig] = None,
    ):
        """Initialize the S3BikeDataLoader.

        Args:
            root_dir: Root directory containing data sources
            bucket_name: Name of the S3 bucket
            config (optional, default None): Configuration object (uses defaults if None)
        """
        super().__init__(root_dir, config)

        self.bucket_name = bucket_name

    @abstractmethod
    def _discover_sources(self) -> None:
        """Discover and map all valid data sources to month keys."""
        raise NotImplementedError

    def _fetch_file(
        self,
        s3_client,
        bucket_name: str,
        key: str,
        file_path: Path,
        max_attempts: int = 5,
        retry_wait_min: int = 2,
        retry_wait_max: int = 10,
    ) -> None:
        """
        Download a single file from S3 with retry logic.

        This function handles both transient and permanent errors, with detailed
        logging for each attempt. It uses exponential backoff for retries.

        Args:
            s3_client: Configured boto3 S3 client
            bucket_name: Name of the S3 bucket
            key: Key (path) of the file in the bucket
            file_path: Local path where the file will be saved
            max_attempts: Maximum number of retry attempts
            retry_wait_min: Minimum wait time between retries (exponential backoff)
            retry_wait_max: Maximum wait time between retries

        Raises:
            S3PermanentError: For permanent errors that should not be retried
            S3TransientError: For transient errors after all retries are exhausted
        """

        def custom_before_sleep(retry_state):
            logger.warning(
                f"Transient error. Retrying {retry_state.attempt_number}/{max_attempts}..."
            )

        retry_config = {
            "wait": wait_exponential(multiplier=1, min=retry_wait_min, max=retry_wait_max),
            "stop": stop_after_attempt(max_attempts),
            "retry": retry_if_exception_type((ClientError, BotoCoreError)),
            "before_sleep": custom_before_sleep,
        }

        @retry(**retry_config)
        def download_file() -> None:
            try:
                logger.debug(f"Attempting to download {key} from {bucket_name}")
                s3_client.download_file(bucket_name, key, str(file_path))
                logger.info(f"Successfully downloaded {key} to {file_path}")
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code")

                match error_code:
                    case "404" | "NoSuchKey" | "AccessDenied":
                        # These are permanent errors that shouldn't be retried
                        raise S3PermanentError(
                            f"Permanent error downloading {key}: {str(e)}"
                        ) from e
                    case "Throttling" | "RequestLimitExceeded" | "5xx":
                        # These are transient errors that can be retried
                        logger.warning(
                            f"Transient error downloading {key} (will retry): {str(e)}"
                        )
                        raise S3TransientError(str(e)) from e
                    case _:
                        # Other ClientErrors that aren't clearly transient or permanent
                        logger.error(f"Unexpected ClientError for {key}: {str(e)}")
                        raise

        try:
            download_file()
        except S3PermanentError as e:
            logger.error(str(e))
            raise
        except S3TransientError as e:
            logger.error(f"All retry attempts failed for {key}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading {key}: {str(e)}")
            raise

    def fetch(
        self,
        save_dir: Optional[Path] = None,
        *,
        filenames: Optional[List[str]] = None,
        pattern: Optional[str] = None,
        max_workers: Optional[int] = None,
        max_attempts: int = 5,
        retry_wait_min: int = 2,
        retry_wait_max: int = 10,
    ) -> None:
        """
        Fetch bike trip data from the S3 bucket using boto3.

        Downloads ZIP archives matching either:
        - The specified filenames (if provided)
        - A custom regex pattern (if provided)
        - The class's default filename pattern (otherwise)

        Files are downloaded concurrently with progress tracking and retry logic
        for transient errors.

        Args:
            save_dir: Directory where downloaded files will be saved.
            filenames: Optional list of specific filenames to download.
            pattern: Optional regex pattern to select files.
            max_workers (defaults to config.max_workers): Maximum number of concurrent
                downloads.
            max_attempts: Maximum number of retry attempts for transient errors.
            retry_wait_min: Minimum wait time between retries (exponential backoff).
            retry_wait_max: Maximum wait time between retries.

        Returns:
            None

        Raises:
            S3PermanentError: For permanent errors that should not be retried
            S3TransientError: For transient errors after all retries are exhausted
        """
        if not self.bucket_name:
            raise ValueError("Bucket name is not set. Please provide a bucket name.")

        save_dir = save_dir or self.root_dir

        if save_dir and not save_dir.exists():
            logger.info(f"Creating directory: {save_dir}")
            save_dir.mkdir(parents=True)

        # Initialize boto3 S3 client with unsigned configuration
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

        # List objects in the bucket that match our pattern
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix="")

        # Determine which files to download based on provided parameters
        matching_files = []

        # If specific filenames are provided, use those directly
        if filenames:
            # Check if the files exist in the bucket
            existing_files = set()
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        if any(re.search(rf"{filename}\b", key) for filename in filenames):
                            existing_files.add(key)

            # Only add files that actually exist in the bucket
            matching_files.extend(existing_files)
        elif pattern:

            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        if re.search(pattern, key):
                            matching_files.append(key)

        if not matching_files:
            logger.warning("No matching files found in the S3 bucket")
            return

        # Sort files for consistent processing order
        matching_files.sort()

        # Set up progress tracking
        workers = max_workers or self.config.max_workers
        total_files = len(matching_files)
        logger.info(f"Found {total_files} files to download")

        # Download files concurrently with progress bars and retry logic
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Create a progress bar for overall progress
            with tqdm(total=total_files, desc="Overall Progress") as overall_pbar:
                # Submit all download tasks
                future_to_file = {}
                for key in matching_files:
                    try:
                        file_path = save_dir / Path(key).name
                        future = executor.submit(
                            self._fetch_file,
                            s3,
                            self.bucket_name,
                            key,
                            file_path,
                            max_attempts,
                            retry_wait_min,
                            retry_wait_max,
                        )
                        future_to_file[future] = (key, file_path)
                    except Exception as e:
                        logger.error(f"Error submitting download for {key}: {str(e)}")

                # Process completed tasks with progress tracking
                for future in as_completed(future_to_file):
                    key, file_path = future_to_file[future]
                    try:
                        # Update overall progress
                        overall_pbar.update(1)

                        # Log per-file progress
                        logger.info(f"Downloaded {key} to {file_path}")
                    except Exception as e:
                        logger.error(f"Error processing download for {key}: {str(e)}")
        logger.success("All files downloaded successfully")
        self._discover_sources()
        logger.success("All files downloaded successfully")
        self._discover_sources()
        logger.success("All files downloaded successfully")
        self._discover_sources()
