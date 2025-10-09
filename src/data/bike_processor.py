from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from loguru import logger


@dataclass(frozen=True)
class BikeTripsDataProcessorConfig:
    """Configuration for the BikeTripsDataProcessor.

    This class holds configuration parameters for the data processing pipeline.
    It ensures that all required settings are provided and validates them.

    Attributes:
        freq: The frequency at which to aggregate the data (e.g., '1h', '1D').
             Must be a valid pandas offset string.
        fill_missing: Whether to fill missing time periods with zeros.
        required_columns: List of required columns in the input DataFrame.
    """

    freq: str = field(
        default="1h",
        metadata={"description": "Frequency for data aggregation"},
    )
    fill_missing: bool = field(
        default=True,
        metadata={"description": "Flag to fill missing time periods"},
    )
    engine: Literal["pyarrow", "fastparquet"] = field(
        default="pyarrow",
        metadata={"description": "Engine for reading Parquet files"},
    )

    # Define required columns as a class-level constant
    REQUIRED_COLUMNS: Tuple[str, ...] = (
        "StartTime",
        "EndTime",
        "StartStationId",
        "EndStationId",
        "duration_minutes",
    )

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        # Validate frequency is a valid pandas offset string
        try:
            pd.tseries.frequencies.to_offset(self.freq)
        except ValueError as e:
            raise ValueError(f"Invalid frequency '{self.freq}': {e}")


class BikeTripsDataProcessor:
    """A class to process bike trip data into time series format with configurable
    aggregation windows.

    Transforms raw trip data into hourly (or other configurable intervals)
    aggregated data showing departures, arrivals, and average duration per station.
    """

    def __init__(
        self,
        source_file: Path,
        usecols: Optional[List[str]] = None,
        config: Optional[BikeTripsDataProcessorConfig] = None,
    ):
        """Initialize the DataProcessor.

        Args:
            source_file: The path to the source file containing the raw trip data.
            usecols (optional, default None): Columns to read from the source file.
            config (optional, default None): Configuration for the processor.

        Returns:
            None
        """
        self.df = pd.read_parquet(source_file, engine="pyarrow", columns=usecols)
        self.config = config or BikeTripsDataProcessorConfig()

        self.processed_data = None
        self.data_info_ = {}

    def process_trips(self) -> pd.DataFrame:
        """Process raw trip data into time series format.

        Returns:
            Processed time series with columns:
                - timestamp: Time period start
                - station_id: Station identifier
                - departures: Number of trips starting from this station
                - arrivals: Number of trips ending at this station
                - avg_duration_minutes: Average duration of trips ending at this station
        """
        # Input validation
        self._validate_input()

        # Convert timestamps to datetime and extract time periods
        df_processed = self._prepare_timestamps(self.df)

        # Process departures and arrivals separately for efficiency
        departures_df = self._process_departures(df_processed)
        arrivals_df = self._process_arrivals(df_processed)

        # Merge and create complete time series
        result_df = self._merge_and_complete_timeseries(departures_df, arrivals_df)

        # Store processing info
        self._store_processing_info(self.df, result_df)

        self.processed_data = result_df
        return result_df

    def _validate_input(self) -> None:
        """Validate input DataFrame has required columns.

        Raises:
            ValueError: If any required columns are missing or if the DataFrame is empty.
        """
        missing_cols = [
            col for col in self.config.REQUIRED_COLUMNS if col not in self.df.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if len(self.df) == 0:
            raise ValueError("Input DataFrame is empty")

    def _prepare_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert timestamps and create time period columns.

        Args:
            df: Input DataFrame with 'StartTime' and 'EndTime' columns.

        Returns:
            DataFrame with additional columns:
                - start_dt: Datetime with UTC timezone
                - end_dt: Datetime with UTC timezone
                - start_period: Time period start (floor to nearest interval)
                - end_period: Time period end (floor to nearest interval)
        """
        # Check for datetime64[ns] type to avoid redundant conversions
        if not pd.api.types.is_datetime64_any_dtype(df["StartTime"]):
            df["start_dt"] = pd.to_datetime(df["StartTime"], utc=True)
        else:
            df["start_dt"] = df["StartTime"].dt.tz_localize("UTC", ambiguous="NaT")

        if not pd.api.types.is_datetime64_any_dtype(df["EndTime"]):
            df["end_dt"] = pd.to_datetime(df["EndTime"], utc=True)
        else:
            df["end_dt"] = df["EndTime"].dt.tz_localize("UTC", ambiguous="NaT")

        # Create time period columns (floor to nearest interval)
        df["start_period"] = df["start_dt"].dt.floor(self.config.freq)
        df["end_period"] = df["end_dt"].dt.floor(self.config.freq)

        logger.info("Converted timestamps and created time period columns")

        return df

    def _process_departures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process departure counts by station and time period.

        Args:
            df: DataFrame with 'start_period' and 'StartStationId' columns.

        Returns:
            DataFrame with departure counts per station and time period.
        """
        departures = (
            df.groupby(["start_period", "StartStationId"])
            .size()
            .reset_index(name="departures")
            .rename(columns={"start_period": "timestamp", "StartStationId": "station_id"})
        )

        logger.info("Computed station-wise departure counts")
        return departures

    def _process_arrivals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process arrival counts and average duration by station and time period.

        Args:
            df: DataFrame with 'end_period', 'EndStationId', and 'duration_minutes' columns.

        Returns:
            DataFrame with arrival counts and average duration per station and time period.
        """
        arrivals = (
            df.groupby(["end_period", "EndStationId"])
            .agg(
                arrivals=("duration_minutes", "count"),
                avg_duration_minutes=("duration_minutes", "mean"),
            )
            .round(2)
            .reset_index()
            .rename(columns={"end_period": "timestamp", "EndStationId": "station_id"})
        )

        logger.info("Computed station-wise arrival counts and average arrival duration")
        return arrivals

    def _merge_and_complete_timeseries(
        self,
        departures_df: pd.DataFrame,
        arrivals_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge departures and arrivals, create complete time series.

        Args:
            departures_df (pd.DataFrame): DataFrame with departure counts.
            arrivals_df (pd.DataFrame): DataFrame with arrival counts and average duration.

        Returns:
            pd.DataFrame: Merged DataFrame with complete time series data.
        """
        # Outer join on timestamp and station_id
        merged = pd.merge(
            departures_df,
            arrivals_df,
            on=["timestamp", "station_id"],
            how="outer",
        ).fillna(0)

        if self.config.fill_missing:
            # Create full index
            all_stations = sorted(
                set(departures_df["station_id"]).union(set(arrivals_df["station_id"]))
            )
            min_time = merged["timestamp"].min()
            max_time = merged["timestamp"].max()
            time_range = pd.date_range(start=min_time, end=max_time, freq=self.config.freq)

            # Create MultiIndex
            full_index = pd.MultiIndex.from_product(
                [time_range, all_stations], names=["timestamp", "station_id"]
            )
            # Reindex
            merged.set_index(["timestamp", "station_id"], inplace=True)
            result = merged.reindex(full_index, fill_value=0).reset_index()
        else:
            result = merged

        logger.info("Merged arrivals and departures")

        # Ensure correct types, sort by timestamp
        result = (
            result.astype({"departures": int, "arrivals": int, "avg_duration_minutes": float})
            .sort_values(["timestamp", "station_id"])
            .reset_index(drop=True)
        )

        return result

    def filter_by_daily_arrivals(
        self,
        *,
        arrivals_col: str = "arrivals",
        ts_col: str = "timestamp",
        min_daily_arrivals: int,
    ) -> pd.DataFrame:
        """
        Remove all rows whose calendar‑day total of arrivals is less than a certain threshold

        Args:
            df : Input DataFrame. Must contain at least `ts_col` (datetime‑like)
                and `arrivals_col` (numeric) columns.
            arrivals_col : Name of the column storing the per‑record arrival count.
                Defaults to `arrivals`.
            ts_col : Name of the column storing the timestamp. The column may be a string,
                `datetime64[ns]`, or a timezone‑aware `datetime`. It will be coerced
                to UTC and truncated to the date component.
            min_daily_arrivals : Days whose summed `arrivals` are strictly less than this
                value will be discarded.

        Returns:
            A view of `df` that only contains rows belonging to days
            meeting the threshold.

        Raises:
            ValueError : If required columns are missing or if `min_daily_arrivals`
                is negative.

        Notes:
            The function is idempotent – calling it twice with the same `df` and
                `min_daily_arrivals` yields the same result.
            It works with arbitrarily large datasets as long as they fit in memory.
            For out‑of‑core workloads replace the groupby with a dask or Spark aggregation.
        """
        ts_series = pd.to_datetime(self.processed_data[ts_col], utc=True, errors="coerce")
        if ts_series.isna().any():
            logger.warning(
                f"Found {ts_series.isna().sum()} malformed timestamps; "
                "those rows will be excluded.",
            )
        self.processed_data = self.processed_data.assign(date=ts_series.dt.date)

        daily_sums = (
            self.processed_data.groupby(["station_id", "date"])
            .agg(daily_arrivals=(arrivals_col, "sum"))
            .reset_index()
        )

        avg_daily_arrivals = (
            daily_sums.groupby("station_id")
            .agg(
                avg_daily_arrivals=("daily_arrivals", "mean"),
                median_daily_arrivals=("daily_arrivals", "median"),
            )
            .reset_index()
        )

        # Extract stations that do not meet the daily threshold
        low_demand_stations = avg_daily_arrivals.loc[
            avg_daily_arrivals["avg_daily_arrivals"] < min_daily_arrivals, "station_id"
        ].unique()

        # Filter original rows
        filtered = self.processed_data.loc[
            ~self.processed_data["station_id"].isin(low_demand_stations)
        ].drop(columns="date")
        logger.info(
            f"Filtered out {len(low_demand_stations):,} stations -→ {len(filtered):,} rows "
            f"remain ({len(filtered) / len(self.processed_data):.1%})"
        )
        self.processed_data = filtered
        return filtered.reset_index(drop=True)

    def save_data(
        self,
        output_filename: Path | str,
        compression: Literal["gzip", "brotli", "lz4", "zstd"] = "gzip",
    ) -> None:
        """Save a DataFrame to a Parquet file.

        This method saves the processed data to a Parquet file in the interim data directory.
        Output file name is appended with the sample frequency, parquet format and
        compression method.

        Args:
            output_filename: The filename for the output file.
            compression (optional, default "gzip"): Compression method.

        Raises:
            Exception: If there's an error during saving.


        """
        output_file = Path(f"{output_filename}_resampled_{self.config.freq}")
        try:
            if self.processed_data is None:
                logger.error("No data to save - dataframe is empty")
                return

            self.processed_data.to_parquet(
                output_file.with_suffix(f".parquet.{compression}"),
                compression=compression,
                engine="pyarrow",
                index=False,
            )

            logger.success(f"Data saved to {output_file.parent.name}/{output_file.name}")

        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def _store_processing_info(
        self,
        original_df: pd.DataFrame,
        result_df: pd.DataFrame,
    ) -> None:
        """Store information about the processing.

        Args:
            original_df (pd.DataFrame): The original input DataFrame.
            result_df (pd.DataFrame): The processed result DataFrame.
        """
        self.data_info_ = {
            "original_trips": len(original_df),
            "unique_stations": len(result_df["station_id"].unique()),
            "time_periods": len(result_df["timestamp"].unique()),
            "date_range": {
                "start": result_df["timestamp"].min(),
                "end": result_df["timestamp"].max(),
            },
            "total_departures": result_df["departures"].sum(),
            "total_arrivals": result_df["arrivals"].sum(),
            "frequency": self.config.freq,
            "fill_missing": self.config.fill_missing,
        }

    def get_processing_info(self) -> Dict[str, Any]:
        """Get information about the last processing operation.

        Returns:
            Dict[str, Any]: Dictionary containing processing information.
        """
        return self.data_info_.copy()

    def get_station_summary(self) -> pd.DataFrame:
        """Get summary statistics per station.

        Returns:
            pd.DataFrame: DataFrame with summary statistics for each station.

        Raises:
            ValueError: If no processed data is available.
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run process_trips() first.")

        summary = (
            self.processed_data.groupby("station_id")
            .agg(
                {
                    "departures": ["sum", "mean", "max"],
                    "arrivals": ["sum", "mean", "max"],
                    "avg_duration_minutes": "mean",
                },
            )
            .round(2)
        )

        # Flatten column names
        summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
        summary = summary.reset_index()

        return summary

    def get_time_summary(self) -> pd.DataFrame:
        """Get summary statistics per time period.

        Returns:
            pd.DataFrame: DataFrame with summary statistics for each time period.

        Raises:
            ValueError: If no processed data is available.
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run process_trips() first.")

        summary = (
            self.processed_data.groupby("timestamp")
            .agg({"departures": "sum", "arrivals": "sum", "avg_duration_minutes": "mean"})
            .round(2)
            .reset_index()
        )

        return summary
