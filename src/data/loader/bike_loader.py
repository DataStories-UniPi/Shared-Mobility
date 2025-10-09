import os
import re
import timeit
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
from loguru import logger

from .core import S3DataLoader
from .models import DataSummary, LoaderConfig, LoadResult


class BikeDataLoader(S3DataLoader):
    """
    Production-ready loader for Divvy bike trip data from multiple source types.

    Supports efficient loading from directories with CSV files and ZIP archives
    with configurable chunking and parallel processing capabilities.
    """

    def __init__(
        self,
        root_dir: Path,
        bucket_name: Optional[str] = None,
        config: Optional[LoaderConfig] = None,
    ):
        """Initialize the BikeDataLoader."""
        # Initialize the parent class
        super().__init__(root_dir, bucket_name, config)

        self._month_sources: Dict[str, List[Path]] = defaultdict(list)
        self._discover_sources()

    def _discover_sources(self) -> None:
        """Discover and map all valid data sources to month keys."""
        self._month_sources.clear()

        for item in self.root_dir.iterdir():
            if item.is_dir():
                # Process year directories with month subdirectories
                for subitem in item.iterdir():
                    month_key = self._extract_from_path(subitem)
                    if month_key:
                        self._month_sources[month_key].append(subitem)
            elif item.suffix.lower() == ".zip":
                # Process direct ZIP files
                month_key = self._extract_from_path(item)
                if month_key:
                    self._month_sources[month_key].append(item)

        logger.info(
            f"Discovered {len(self._month_sources)} months: "
            f"{sorted(self._month_sources.keys())}"
        )

    def _extract_from_path(self, path: Path) -> Optional[str]:
        """
        Extract month key (YYYY-MM) from path using multiple strategies.

        Supports patterns like:
        - 2016/1_January/ → 2016-01
        - 201601-tripdata.zip → 2016-01
        """
        name = path.stem

        # Strategy Numbered month folders (1_January, 2_February, etc.)
        if "_" in name and name.split("_")[0].isdigit():
            try:
                parent = path.parent.name
                if parent.startswith("20") and len(parent) >= 4:
                    year = parent[:4]
                    month_num = int(name.split("_")[0])
                    if 1 <= month_num <= 12:
                        return f"{year}-{month_num:02d}"
            except (ValueError, AttributeError):
                pass

        # Strategy 2: YYYYMM prefix in filename
        match = re.match(r"^(\d{6})", name)
        if match:
            yyyymm = match.group(1)
            year, month = yyyymm[:4], yyyymm[4:]
            try:
                month_int = int(month)
                if 1 <= month_int <= 12:
                    return f"{year}-{month_int:02d}"
            except ValueError:
                logger.error(f"Invalid month number in {name} (YYYYMM: {yyyymm})")

        return None

    def load_data(
        self,
        months: Optional[List[str]] = None,
        *,
        sort: Optional[str] = None,
        merge: bool = True,
        max_workers: Optional[int] = None,
    ) -> Dict[str, LoadResult]:
        """
        Load data for specified months in parallel.

        Args:
            max_workers: Maximum worker threads (defaults to config value)
            months: List[str] - Month keys (YYYY-MM format) to load.
                If None, loads all available months.
            sort: Optional column name to sort the DataFrame by after loading

        Returns:
            Dictionary mapping month keys to load results
        """
        workers = max_workers or self.config.max_workers
        results: Dict[str, LoadResult] = {}

        # Get the list of months to process
        if months is not None:
            # Validate that all requested months exist
            invalid_months = [m for m in months if m not in self._month_sources]
            if invalid_months:
                logger.warning(f"Requested months not found: {invalid_months}")
            month_keys = sorted(set(months) & set(self._month_sources.keys()))
        else:
            # Load all available months
            month_keys = sorted(self._month_sources.keys())

        if not month_keys:
            logger.info("No months to process")
            return results

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_month = {
                executor.submit(self._load_month_with_timing, month, sort): month
                for month in month_keys
            }

            progress_iter = as_completed(future_to_month)

            for future in progress_iter:
                month = future_to_month[future]
                results[month] = future.result()
        if merge:
            # Merge all loaded data into a single DataFrame
            df = pd.concat([result.data for result in results.values() if result.success])
            results["all"] = LoadResult(
                data=df,
                month="all",
                rows_loaded=len(df),
                success=True,
                processing_time_seconds=sum(
                    result.processing_time_seconds
                    for result in results.values()
                    if result.success
                ),
            )
        return results

    def _load_month_with_timing(
        self,
        month_key: str,
        sort: Optional[str] = None,
    ) -> LoadResult:
        """Load a month's data with timing and error handling."""
        start_time = timeit.default_timer()

        try:
            df = self._load_month_dataframe(month_key, sort)
            processing_time = timeit.default_timer() - start_time

            logger.info(f"Loaded {len(df)} rows for {month_key} in {processing_time:.2f}s")
            return LoadResult(
                data=df,
                month=month_key,
                rows_loaded=len(df),
                success=True,
                processing_time_seconds=processing_time,
            )
        except Exception as e:
            processing_time = timeit.default_timer() - start_time
            error_msg = f"Failed to load {month_key}: {str(e)}"
            logger.error(error_msg)

            return LoadResult(
                data=None,
                month=month_key,
                rows_loaded=0,
                success=False,
                error_message=error_msg,
                processing_time_seconds=processing_time,
            )

    def _load_month_dataframe(
        self, month_key: str, sort: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load entire month as a single DataFrame.

        Args:
            month_key: Month identifier in YYYY-MM format
            sort: Optional column name to sort the DataFrame by

        Returns:
            Complete DataFrame for the specified month

        Raises:
            ValueError: If no data is found for the month
        """
        chunks = list(self.load_month_chunks(month_key))
        if not chunks:
            raise ValueError(f"No data found for month: {month_key}")

        df = pd.concat(chunks, ignore_index=True)

        # Apply sorting if a column name was provided
        if sort is not None and sort in df.columns:
            df = df.sort_values(by=sort).reset_index(drop=True)

        return df

    def load_month_chunks(self, month_key: str) -> Iterator[pd.DataFrame]:
        """
        Load data for a specific month as DataFrame chunks.

        Args:
            month_key: Month identifier in YYYY-MM format

        Yields:
            Standardized DataFrame chunks

        Raises:
            KeyError: If month_key is not found
        """
        sources = self._month_sources.get(month_key)
        if not sources:
            raise KeyError(f"No data sources found for month: {month_key}")

        for source_path in sources:
            yield from self._read_source_chunks(source_path)

    def _read_source_chunks(self, source_path: Path) -> Iterator[pd.DataFrame]:
        """Read chunks from a single source (directory or ZIP file)."""
        try:
            logger.debug(f"Reading source: {source_path.stem}")
            if source_path.is_dir():
                yield from self._read_directory_chunks(source_path)

            match source_path.suffix.lower():
                case ".zip":
                    yield from self._read_zip_chunks(source_path)
                case ".csv":
                    yield from self._read_csv_chunks(source_path)
                case _:
                    logger.warning(f"Unsupported source type: {source_path}")
        except Exception as e:
            logger.error(f"Error reading source {source_path}: {e}")
            raise

    def _read_zip_chunks(self, zip_path: Path) -> Iterator[pd.DataFrame]:
        """Read chunks from ZIP file containing CSV files."""
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Get the first CSV file (assuming there's only one relevant CSV)
                csv_files = [
                    f
                    for f in zf.namelist()
                    if f.lower().endswith(".csv") and "__MACOSX" not in f
                ]
                if not csv_files:
                    logger.warning(f"No CSV files found in {zip_path}")
                    return

                for csv_name in csv_files:
                    logger.debug(f"Processing CSV file from ZIP: {csv_name}")

                    try:
                        # Try to read directly like in the example script
                        with zf.open(csv_name) as csv_stream:
                            yield from self._read_csv_chunks(csv_stream)
                    except Exception as e:
                        logger.error(f"Error reading CSV from ZIP: {e}")
                        raise
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid ZIP file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing ZIP file: {e}")
            raise

    def _read_directory_chunks(self, dir_path: Path) -> Iterator[pd.DataFrame]:
        """Read chunks from directory containing CSV files."""
        csv_files = sorted(dir_path.glob("*.csv"))
        for csv_file in csv_files:
            # Exclude __MACOSX/ files
            if "__MACOSX" not in str(csv_file).split(os.sep):
                yield from self._read_csv_chunks(csv_file)

    def _read_csv_chunks(self, file_or_stream: Any) -> Iterator[pd.DataFrame]:
        """Read and standardize chunks from CSV file or stream."""
        # We'll handle date parsing in the standardize_chunk method
        # For now, just read the data without specific date parsing
        try:
            # Try reading with automatic encoding detection first (most common)
            try:
                for chunk in pd.read_csv(
                    file_or_stream,
                    chunksize=self.config.chunk_size,
                    parse_dates=False,  # Don't parse dates here
                    dtype_backend=self.config.dtype_backend,
                    # Let pandas detect encoding automatically
                ):
                    yield self.standardize_chunk(chunk)
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                logger.warning(f"Encoding detection failed or file is not a valid CSV: {e}")
                try:
                    # As a last resort, try with latin1 which can decode any byte sequence
                    for chunk in pd.read_csv(
                        file_or_stream,
                        chunksize=self.config.chunk_size,
                        parse_dates=False,
                        dtype_backend=self.config.dtype_backend,
                        encoding="latin1",
                    ):
                        # Check if we got valid data before yielding
                        if not chunk.empty and any(chunk.columns):
                            yield self.standardize_chunk(chunk)
                        else:
                            logger.error("No columns parsed from file after latin1 fallback")
                            break
                except Exception as inner_e:
                    logger.error(f"Fallback to latin1 also failed: {inner_e}")
                    raise
        except Exception as e:
            logger.error(f"Error processing CSV chunks: {e}")
            raise

    def standardize_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Standardize chunk columns and add derived fields."""
        # Normalize column names
        chunk.columns = [col.lower().strip() for col in chunk.columns]

        # Parse dates for all starttime columns if configured to do so
        if self.config.parse_dates:
            for col in list(chunk.columns):
                if any(
                    col == existing_col.lower().strip()
                    for existing_col in self.config.starttime_columns
                ):
                    chunk[col] = pd.to_datetime(chunk[col], errors="coerce")

        # Add yearmonth column if missing
        if self.config.yearmonth_column not in chunk.columns:
            self._add_yearmonth_column(chunk)

        return chunk

    def _add_yearmonth_column(self, chunk: pd.DataFrame) -> None:
        """Add yearmonth column derived from any of the starttime columns."""
        # Check all possible start time columns
        for starttime_col in self.config.starttime_columns:
            if starttime_col not in chunk.columns:
                continue

            # Ensure datetime type
            if not pd.api.types.is_datetime64_any_dtype(chunk[starttime_col]):
                chunk[starttime_col] = pd.to_datetime(chunk[starttime_col], errors="coerce")

            # Add period-based month column using the first valid start time column
            if pd.api.types.is_datetime64_any_dtype(chunk[starttime_col]):
                chunk[self.config.yearmonth_column] = (
                    chunk[starttime_col].dt.to_period("M").astype(str)
                )
            return  # Exit after processing the first valid column

    @lru_cache(maxsize=1)
    def get_data_summary(self):
        """
        Get summary of all discovered data sources.

        Returns:
            DataFrame with columns: month, source_count, csv_count
        """
        summaries = []

        for month_key, sources in self._month_sources.items():
            csv_count = sum(self._count_csv_files(source) for source in sources)

            summaries.append(
                DataSummary(
                    month=month_key,
                    source_count=len(sources),
                    csv_count=csv_count,
                )
            )

        summary_df = pd.DataFrame([summary.__dict__ for summary in summaries])
        return summary_df.sort_values("month").reset_index(drop=True)

    def _count_csv_files(self, source_path: Path) -> int:
        """Count CSV files in a source (directory or ZIP)."""
        try:
            if source_path.suffix.lower() == ".zip":
                with zipfile.ZipFile(source_path, "r") as zf:
                    return len(
                        [
                            f
                            for f in zf.namelist()
                            if f.lower().endswith(".csv") and "__MACOSX" not in f
                        ]
                    )
            elif source_path.is_dir():
                # Exclude __MACOSX/ files
                csv_files = [
                    f
                    for f in source_path.glob("*.csv")
                    if "__MACOSX" not in str(f).split(os.sep)
                ]
                return len(csv_files)
            return 0
        except Exception as e:
            logger.warning(f"Could not count CSV files in {source_path}: {e}")
            return 0
