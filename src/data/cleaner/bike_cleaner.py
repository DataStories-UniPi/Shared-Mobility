import calendar
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import janitor
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from .core import DataCleaner
from .models import CleaningConfig, SaveConfig

warnings.filterwarnings("ignore")


class BikeTripsDataCleaner(DataCleaner):
    """
    Enhanced bike data cleaning workflow with pyjanitor integration,
    configurable saving, and professional logging
    """

    def __init__(
        self,
        df: pd.DataFrame,
        clean_config: Optional[CleaningConfig] = None,
        save_config: Optional[SaveConfig] = None,
    ):

        super().__init__(df, clean_config)

        self.save_config = save_config or SaveConfig()
        self.column_mapping = self.clean_config.column_mapping
        self.validation_results = {}

        self.metadata_ = {
            "cleaning_timestamp": datetime.now().isoformat(),
            "original_shape": self.original_shape,
            "config": self.clean_config.to_dict(),
        }

        logger.info(
            f"🚀 Initialized {self.__class__.__name__} with "
            f"{self.original_shape[0]:,} records"
        )
        logger.info(f"📋 Column mapping: {self.column_mapping.to_dict()}")
        logger.info(f"📋 Cleaning configuration: {self.clean_config.to_dict()}")

    def _validate_and_standardize_columns(self) -> pd.DataFrame:
        """
        Validate required columns exist and standardize column names

        Parameters:
        - df: Original DataFrame

        Returns:
        - DataFrame with standardized column names
        """
        logger.info("🔍 Validating and standardizing column names...")

        # Check if required columns exist
        required_columns = self.column_mapping.get_required_columns()
        missing_required = [col for col in required_columns if col not in self.df.columns]

        if missing_required:
            available_cols = list(self.df.columns)
            logger.error(f"❌ Missing required columns: {missing_required}")
            logger.info(f"📋 Available columns: {available_cols}")

            # Try to suggest potential mappings
            suggestions = self._suggest_column_mappings(self.df.columns.tolist())
            if suggestions:
                logger.info("💡 Suggested column mappings:")
                for standard_name, suggested_col in suggestions.items():
                    logger.info(f"  • {standard_name} -> {suggested_col}")

            raise ValueError(
                f"Missing required columns: {missing_required}. "
                f"Available columns: {available_cols}"
            )

        # Create standardized DataFrame
        standardized_df = self.df.copy()

        # Rename columns to standardized names
        column_rename_map = {}

        # Required columns
        if self.column_mapping.start_time != "StartTime":
            column_rename_map[self.column_mapping.start_time] = "StartTime"

        if self.column_mapping.end_time != "EndTime":
            column_rename_map[self.column_mapping.end_time] = "EndTime"

        if self.column_mapping.start_station_id != "StartStationId":
            column_rename_map[self.column_mapping.start_station_id] = "StartStationId"

        if self.column_mapping.end_station_id != "EndStationId":
            column_rename_map[self.column_mapping.end_station_id] = "EndStationId"

        # Optional columns
        if self.column_mapping.trip_id and self.column_mapping.trip_id != "TripId":
            if self.column_mapping.trip_id in self.df.columns:
                column_rename_map[self.column_mapping.trip_id] = "TripId"

        if self.column_mapping.user_id and self.column_mapping.user_id in self.df.columns:
            column_rename_map[self.column_mapping.user_id] = "UserId"

        if self.column_mapping.bike_id and self.column_mapping.bike_id in self.df.columns:
            column_rename_map[self.column_mapping.bike_id] = "BikeId"

        if (
            self.column_mapping.trip_duration
            and self.column_mapping.trip_duration in self.df.columns
        ):
            column_rename_map[self.column_mapping.trip_duration] = "ExistingTripDuration"

        # Apply column renaming
        if column_rename_map:
            standardized_df = standardized_df.rename(columns=column_rename_map)
            logger.info(f"📝 Renamed columns: {column_rename_map}")

        # Check what columns we have available for optional operations
        has_trip_id = "TripId" in standardized_df.columns
        logger.info(f"📊 Optional columns available: TripId={has_trip_id}")

        if not has_trip_id:
            logger.warning(
                "⚠️ No TripId column available - will skip TripId-dependent operations"
            )

        return standardized_df

    def _suggest_column_mappings(self, available_columns: List[str]) -> Dict[str, str]:
        """
        Suggest potential column mappings based on common naming patterns

        Parameters:
        - available_columns: List of available column names

        Returns:
        - Dictionary of suggested mappings
        """
        suggestions = {}

        # Common patterns for different columns
        patterns = {
            "start_time": [
                "start_time",
                "starttime",
                "start_datetime",
                "departure_time",
                "trip_start_time",
                "begin_time",
                "started_at",
            ],
            "end_time": [
                "end_time",
                "endtime",
                "stoptime",
                "end_datetime",
                "arrival_time",
                "trip_end_time",
                "finish_time",
                "ended_at",
            ],
            "start_station_id": [
                "start_station_id",
                "startstation",
                "start_station",
                "origin_station_id",
                "from_station_id",
                "departure_station_id",
            ],
            "end_station_id": [
                "end_station_id",
                "endstation",
                "end_station",
                "destination_station_id",
                "to_station_id",
                "arrival_station_id",
            ],
            "trip_id": [
                "trip_id",
                "tripid",
                "ride_id",
                "journey_id",
                "trip_number",
                "id",
            ],
        }

        available_lower = [col.lower() for col in available_columns]

        for standard_name, pattern_list in patterns.items():
            for pattern in pattern_list:
                if pattern.lower() in available_lower:
                    # Find the original column name (with correct case)
                    original_col = available_columns[available_lower.index(pattern.lower())]
                    suggestions[standard_name] = original_col
                    break

        return suggestions

    def _has_trip_id(self) -> bool:
        """Check if TripId column is available"""
        return "TripId" in self.df.columns

    def remove_irrelevant_columns(self, columns: List[str]):
        """Remove columns that are not relevant to the analysis"""
        logger.info("🧹 Removing irrelevant columns...")

        columns_to_remove = [col for col in columns if col not in self.df.columns]

        self.df = self.df.drop(columns=columns_to_remove)

        self.log_cleaning_step(
            "Remove Irrelevant Columns",
            len(columns_to_remove),
            "Irrelevant",
        )

    def convert_datetime_columns(self):
        """Convert time columns to datetime with enhanced error handling"""
        logger.info("📅 Converting datetime columns...")

        # Pandas conversion with multiple formats
        datetime_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
        ]

        for col in ["StartTime", "EndTime"]:
            if col in self.df.columns:
                original_col = col.lower().replace("time", "_time")
                if original_col in self.df.columns:
                    col = original_col

                for fmt in datetime_formats:
                    try:
                        self.df[col] = pd.to_datetime(self.df[col], format=fmt)
                        logger.success(f"✅ Converted {col} using format {fmt}")
                        break
                    except Exception:
                        continue
                else:
                    # Final fallback to pandas automatic parsing
                    self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
                    logger.info(f"🔄 Used automatic parsing for {col}")

        return self.df

    def remove_missing_critical_data(self) -> pd.DataFrame:
        """Remove records with missing critical information using janitor"""
        logger.info("🔍 Checking for missing critical data...")

        initial_count = len(self.df)

        if self._has_trip_id():
            self.df = self.df.dropna(subset=["TripId"])

            if self._is_numeric("TripId"):
                self.df = self.df.astype({"TripId": int})

        self.df = (
            self.df.dropna(subset=["StartTime", "EndTime"], how="all")
            .dropna(subset=["StartStationId", "EndStationId"], how="any")
            .remove_empty()  # type: ignore
        )

        total_removed = initial_count - len(self.df)
        if total_removed > 0:
            self.log_cleaning_step(
                "Missing Critical Data",
                total_removed,
                "Missing TripId, timestamps, or station IDs",
            )

        return self.df

    def remove_duplicate_trips(self):
        """Remove duplicate trip records using janitor's deduplication"""
        logger.info("🔄 Checking for duplicate trips...")

        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        exact_dupes_removed = initial_count - len(self.df)

        # Handle TripId duplicates only if TripId column exists
        if self._has_trip_id():
            logger.info("📋 Checking for TripId duplicates")

            # Check the data type of TripId column
            is_numeric = self._is_numeric("TripId")

            if is_numeric:
                # Convert to integer for numeric IDs
                self.df = self.df.astype({"TripId": int})

            # Drop duplicates regardless of type
            self.df = self.df.drop_duplicates(subset=["TripId"], keep="first")
            tripid_dupes_removed = initial_count - exact_dupes_removed - len(self.df)
        else:
            logger.info("⏭️ Skipping TripId duplicate check (column not available)")
            tripid_dupes_removed = 0

        total_removed = initial_count - len(self.df)
        if total_removed > 0:
            reason = f"Exact duplicates: {exact_dupes_removed}"
            if tripid_dupes_removed > 0:
                reason += f", TripId duplicates: {tripid_dupes_removed}"

            self.log_cleaning_step("Duplicate Trips", total_removed, reason)

        return self.df

    def _is_numeric(self, col: str):
        """Check if a column is numeric"""
        df_col = self.df[col]
        return pd.api.types.is_numeric_dtype(df_col) or pd.api.types.is_integer_dtype(df_col)

    def calculate_trip_duration(self):
        """Calculate trip duration with enhanced validation"""
        logger.info("⏱️ Calculating trip durations...")

        # Use janitor's add_column for method chaining
        duration = (self.df["EndTime"] - self.df["StartTime"]).dt.total_seconds()
        self.df: pd.DataFrame = self.df.assign(
            duration_seconds=duration,
            duration_minutes=duration // 60,
            duration_hours=duration // 3600,
        )

        # Log duration statistics
        duration_stats = self.df["duration_minutes"].apply(["mean", "median", "min", "max"])
        logger.info(
            f"📊 Duration stats: Mean={duration_stats['mean']:.2f}min, "
            f"Median={duration_stats['median']:.2f}min, "
            f"Range=[{duration_stats['min']:.2f}, {duration_stats['max']:.2f}]min"
        )

    def remove_temporal_anomalies(self) -> pd.DataFrame:
        """Remove trips with temporal issues using enhanced logic"""
        logger.info("🕐 Checking for temporal anomalies...")

        initial_count = len(self.df)
        negative_duration_trips = self.df.filter_on("EndTime < StartTime")

        # Swap start <-> end times (possible false entry)
        negative_duration_trips[["EndTime", "StartTime"]] = negative_duration_trips[
            ["StartTime", "EndTime"]
        ]

        # Assign back to original df
        self.df.loc[negative_duration_trips.index, ["StartTime", "EndTime"]] = (
            negative_duration_trips[["StartTime", "EndTime"]]
        )

        self.calculate_trip_duration()

        self.df = (
            self.df.filter_on("duration_seconds >= 0")
            .filter_on(f"duration_minutes >= {self.clean_config.min_trip_duration_minutes}")
            .filter_on(f"duration_minutes <= {self.clean_config.max_trip_duration_minutes}")
        )

        total_removed = initial_count - len(self.df)
        if total_removed > 0:
            self.log_cleaning_step(
                "Temporal Anomalies",
                total_removed,
                f"Duration outside [{self.clean_config.min_trip_duration_minutes}, "
                f"{self.clean_config.max_trip_duration_minutes}] minutes",
            )

        return self.df

    def remove_spatial_anomalies(self):
        """Remove trips with spatial/station issues using enhanced validation"""
        logger.info("🗺️ Checking for spatial anomalies...")

        initial_count = len(self.df)

        # Handle suspicious round trips
        round_trip_mask = self.df["StartStationId"] == self.df["EndStationId"]
        short_round_trips = round_trip_mask & (
            self.df["duration_minutes"] < self.clean_config.min_round_trip_duration_minutes
        )

        self.df = self.df[~short_round_trips]

        total_removed = initial_count - len(self.df)
        if total_removed > 0:
            self.log_cleaning_step(
                "Spatial Anomalies",
                total_removed,
                "Invalid station IDs or suspicious round trips",
            )

        return self.df

    def remove_statistical_outliers(self) -> pd.DataFrame:
        """Remove statistical outliers using multiple methods"""

        logger.info(
            f"📊 Removing statistical outliers using {self.clean_config.outlier_method}"
        )

        initial_count = len(self.df)

        if self.clean_config.outlier_method == "iqr":
            # Interquartile Range method
            Q1, Q3 = self.df["duration_minutes"].quantile([0.25, 0.75])

            IQR = Q3 - Q1

            lower_bound = Q1 - self.clean_config.iqr_multiplier * IQR
            upper_bound = Q3 + self.clean_config.iqr_multiplier * IQR

            self.df = self.df.filter_on(f"duration_minutes >= {lower_bound}").filter_on(
                f"duration_minutes <= {upper_bound}"
            )

            logger.info(f"📊 IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}] minutes")

        elif self.clean_config.outlier_method == "z_score":
            # Z-score method
            mean_duration = self.df["duration_minutes"].mean()
            std_duration = self.df["duration_minutes"].std()

            z_scores = np.abs((self.df["duration_minutes"] - mean_duration) / std_duration)
            outlier_mask = z_scores <= self.clean_config.z_threshold

            self.df = self.df[outlier_mask]

            logger.info(f"📊 Z-score threshold: {self.clean_config.z_threshold}")

        elif self.clean_config.outlier_method == "isolation_forest":
            # Isolation Forest method (requires scikit-learn)
            try:
                from sklearn.ensemble import IsolationForest

                # Prepare features for isolation forest
                features = ["duration_minutes", "StartStationId", "EndStationId"]
                feature_data = (
                    self.df[features].copy().assign(StartHour=self.df["StartTime"].dt.hour)
                )

                # Handle categorical features
                feature_data["StartStationId"] = (
                    feature_data["StartStationId"].astype("category").cat.codes
                )
                feature_data["EndStationId"] = (
                    feature_data["EndStationId"].astype("category").cat.codes
                )

                # Fit isolation forest
                iso_forest = IsolationForest(
                    contamination=self.clean_config.isolation_forest_contamination,
                    random_state=42,
                    n_jobs=-1,
                )

                outlier_labels = iso_forest.fit_predict(feature_data)
                outliers = outlier_labels == -1

                self.df = self.df[~outliers]

                logger.info("📊 Used Isolation Forest with contamination=0.1")

            except ImportError:
                logger.warning("⚠️ sklearn not available, falling back to IQR method")
                self.clean_config.outlier_method = "iqr"
                return self.remove_statistical_outliers()

        removed_count = initial_count - len(self.df)
        if removed_count > 0:
            self.log_cleaning_step(
                "Statistical Outliers",
                removed_count,
                f"Using {self.clean_config.outlier_method} method",
            )

        return self.df

    def add_derived_features(self) -> pd.DataFrame:
        """Add useful derived features using janitor's add_column"""
        logger.info("➕ Adding derived features...")

        self.df = (
            self.df.assign(
                StartHour=self.df["StartTime"].dt.hour,
                StartMonth=self.df["StartTime"].dt.month,
                StartYear=self.df["StartTime"].dt.year,
                StartDayOfWeek=self.df["StartTime"].dt.dayofweek,
                IsRoundTrip=(self.df["StartStationId"] == self.df["EndStationId"]).astype(
                    int
                ),
            )
            .bin_numeric(
                from_column_name="StartHour",
                to_column_name="TimeCategory",
                bins=[0, 6, 12, 18, 24],
                labels=["Night", "Morning", "Afternoon", "Evening"],
                include_lowest=True,
            )  # type: ignore
            .bin_numeric(
                from_column_name="duration_minutes",
                to_column_name="DurationCategory",
                bins=[0, 15, 30, 60, float("inf")],
                labels=["Short", "Medium", "Long", "Very Long"],
                include_lowest=True,
            )
        )

        logger.success("✅ Added derived features successfully")
        return self.df

    def validate_cleaned_data(self) -> bool:
        """Comprehensive validation of cleaned data"""
        logger.info("🔍 Validating cleaned data...")

        validations = {
            **(
                {"no_missing_trip_ids": self.df["TripId"].notna().all()}
                if self._has_trip_id()
                else {}
            ),
            "no_missing_timestamps": self.df[["StartTime", "EndTime"]].notna().all().all(),
            "no_negative_durations": (self.df["duration_minutes"] >= 0).all(),
            "duration_within_bounds": self.df["duration_minutes"]
            .between(
                self.clean_config.min_trip_duration_minutes,
                self.clean_config.max_trip_duration_minutes,
            )
            .all(),
            "chronological_order": (self.df["EndTime"] >= self.df["StartTime"]).all(),
            "no_duplicates": not self.df.duplicated().any(),
            "reasonable_round_trips": (
                ~(
                    (self.df["StartStationId"] == self.df["EndStationId"])
                    & (
                        self.df["duration_minutes"]
                        < self.clean_config.min_round_trip_duration_minutes
                    )
                )
            ).all(),
        }

        all_passed = True
        for check_name, passed in validations.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"🔍 {check_name}: {status}")
            if not passed:
                all_passed = False

        if all_passed:
            logger.success("✅ All validation checks passed!")
        else:
            logger.error("❌ Some validation checks failed!")

        return all_passed

    def save_cleaned_data(
        self,
        custom_path: Optional[str | Path] = None,
        columns: Optional[List[str]] = None,
        keep_original: bool = False,
    ) -> None:
        """
        Save cleaned data with configurable format and compression

        Args:
            custom_path: Custom save path (overrides config)
            columns: Specific columns to include in the saved file
            keep_original: If True, restore original column names before saving
        """
        from config import paths

        def _save_metadata(data_path: Path):
            """Save cleaning metadata alongside the data"""
            if self.df is None or self.df.empty:
                logger.error("No data to save - dataframe is empty")
                return

            metadata = {
                "cleaning_timestamp": datetime.now().isoformat(),
                "original_shape": self.original_shape,
                "final_shape": self.df.shape,
                "cleaning_config": {
                    "min_trip_duration_minutes": self.clean_config.min_trip_duration_minutes,
                    "max_trip_duration_minutes": self.clean_config.max_trip_duration_minutes,
                    "outlier_method": self.clean_config.outlier_method,
                    "chunk_size": self.clean_config.chunk_size,
                },
                "cleaning_log": self.cleaning_log_,
                "validation_results": self.validation_results,
                "data_types": self.df.dtypes.to_dict(),
                "column_stats": {
                    "duration_minutes": {
                        "mean": float(self.df["duration_minutes"].mean()),
                        "median": float(self.df["duration_minutes"].median()),
                        "std": float(self.df["duration_minutes"].std()),
                        "min": float(self.df["duration_minutes"].min()),
                        "max": float(self.df["duration_minutes"].max()),
                    }
                },
            }

            # Create metadata directory if it doesn't exist
            meta_dir = paths.DATA_DIR / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)

            meta_path = (
                meta_dir
                / f"{data_path.parent.name}_{data_path.name.split('.')[0]}.metadata.json"
            )

            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.info(f"Metadata saved to {meta_path.parent.name}/{meta_path.stem}")

        if columns is not None and keep_original:
            raise ValueError("Specify either columns or keep_original, but not both.")

        # Create a copy of the dataframe for saving to avoid modifying the original
        df_to_save = self.df.copy()

        # Manually select columns to keep
        if columns is not None:
            df_to_save = df_to_save[columns]
        # Keep original column names if requested
        elif keep_original and hasattr(self, "original_df"):
            # Map standardized column names back to original names
            column_mapping = {}
            for col in self.df.columns:
                # Find the original name that maps to this standardized name
                for orig_col, std_col in self.column_mapping.to_dict().items():
                    if std_col == col:
                        column_mapping[col] = orig_col

            df_to_save = df_to_save.rename(columns=column_mapping)

        # Convert station ID columns to numeric if possible, otherwise keep as string
        # Column names are guaranteed to be in the dataframe at this point
        for col in ["StartStationId", "EndStationId"]:
            if col in df_to_save.columns:
                try:
                    # Try to convert to numeric, stop if it fails
                    df_to_save[col] = pd.to_numeric(df_to_save[col], errors="raise")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to numeric: {e}")

        # Determine save path
        save_path = Path(custom_path) if custom_path else Path(self.save_config.path)
        save_path = save_path.with_suffix(self.save_config.get_file_extension())

        # Create directory if it doesn't exist
        if not Path.exists(save_path.parent):
            Path.mkdir(save_path.parent, parents=True, exist_ok=True)
            logger.debug(f"Created directory {save_path.parent.stem}")

        logger.info(f"Saving cleaned data in {self.save_config.format} format")

        if self.save_config.compression:
            logger.info(f"Compressing data with {self.save_config.compression} compression")
            name, ext = save_path.name.split(".")
            save_path = save_path.with_name(f"{name}.{ext}.{self.save_config.compression}")

        try:
            # Save based on format
            match self.save_config.format:
                case "parquet":

                    df_to_save.to_parquet(
                        save_path,
                        compression=self.save_config.compression,
                        partition_cols=self.save_config.partition_cols,
                        index=False,
                    )
                case "csv":
                    df_to_save.to_csv(
                        save_path,
                        compression=self.save_config.compression,
                        index=False,
                    )
                case "pickle":
                    df_to_save.to_pickle(save_path, compression=self.save_config.compression)
                case "feather":
                    df_to_save.to_feather(save_path, compression=self.save_config.compression)

            logger.success(f"Data saved to {save_path.parent.name}/{save_path.stem}")

            # Save metadata if requested
            if self.save_config.include_metadata:
                _save_metadata(save_path)

        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

    def visualize_cleaning_impact(self, fig, ax):
        """Create visualizations showing cleaning impact"""

        # Trip duration distribution
        sns.histplot(
            x=self.df["duration_seconds"] // 60,
            bins=30,
            alpha=0.7,
            stat="percent",
            edgecolor="black",
            ax=ax[0, 0],
        )
        # Trips by hour of day
        hourly_counts = self.df["StartHour"].value_counts().sort_index()
        sns.barplot(x=hourly_counts.index, y=hourly_counts.values, alpha=0.7, ax=ax[0, 1])

        # Day of week pattern
        dow_counts = self.df["StartDayOfWeek"].value_counts().sort_index()
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        sns.barplot(
            x=list(range(7)),
            y=[dow_counts.get(i, 0) for i in range(7)],
            alpha=0.7,
            ax=ax[1, 0],
        )

        ax[1, 0].set_xticks(range(7))
        ax[1, 0].set_xticklabels(dow_labels)

        # Round trip vs one-way
        trip_types = ["One-way", "Round Trip"]
        trip_counts = [
            len(self.df) - self.df["IsRoundTrip"].sum(),
            self.df["IsRoundTrip"].sum(),
        ]
        ax[1, 1].pie(trip_counts, labels=trip_types, autopct="%1.1f%%", startangle=90)

    def plot_trip_characteristics(self, fig, ax):
        if self.df is None or self.df.empty:
            logger.error("No data available for visualization")
            return

        fig.suptitle("Bike Share Data Insights Dashboard", fontsize=16, fontweight="bold")

        # 1. Trip Duration Distribution (Top Left)
        sns.histplot(
            x=self.df["duration_minutes"],
            bins=30,
            ax=ax[0, 0],
            edgecolor="black",
        )
        ax[0, 0].axvline(
            self.df["duration_minutes"].mean(),
            color="k",
            ls="--",
            label=f"Mean ({self.df['duration_minutes'].mean():.2f}min)",
        )
        ax[0, 0].axvline(
            self.df["duration_minutes"].median(),
            color="g",
            ls="--",
            label=f"Median ({self.df['duration_minutes'].median():.2f}min)",
        )

        # 2. Trip Count by Hour (Top Right)
        hourly_counts = self.df.groupby("StartHour").size().reset_index(name="TripCount")
        peak_hour = hourly_counts.at[hourly_counts["TripCount"].idxmax(), "StartHour"]

        sns.barplot(data=hourly_counts, x="StartHour", y="TripCount", ax=ax[0, 1])
        ax[0, 1].annotate(
            f"Peak: {peak_hour}:00",
            xy=(peak_hour, hourly_counts["TripCount"].max()),
            xytext=(peak_hour + 1, hourly_counts["TripCount"].max() * 0.9),
            arrowprops=dict(arrowstyle="->", color="red"),
            color="black",
        )
        ax[0, 1].set_xticks(range(0, 24))

        # 3. Monthly Trip Trends by Year (Bottom Left)
        pivot_data = (
            self.df.groupby(["StartYear", "StartMonth"])
            .size()
            .reset_index(name="TripCount")
            .pivot(index="StartMonth", columns="StartYear", values="TripCount")
        )

        pivot_data.plot(ax=ax[1, 0], marker="o", linewidth=2)
        ax[1, 0].set_xticks(range(1, 13))
        ax[1, 0].set_xticklabels(calendar.month_abbr[1:])

        # 4. Global Usage Patterns (Bottom Right)
        heatmap_data = (
            self.df.groupby(["StartDayOfWeek", "StartHour"]).size().unstack(fill_value=0)
        )

        sns.heatmap(
            heatmap_data,
            ax=ax[1, 1],
            cmap="YlOrRd",
            linewidths=0.1,
            cbar_kws={"label": "Number of Trips"},
            square=False,
        )

        ax[1, 1].set_yticklabels(calendar.day_abbr[:], rotation=0)

        # Highlight peak cell
        max_loc = np.unravel_index(np.argmax(heatmap_data.values), heatmap_data.shape)
        ax[1, 1].text(
            max_loc[1] + 0.5,
            max_loc[0] + 0.5,
            "★",
            ha="center",
            va="center",
            fontsize=16,
            color="white",
            weight="bold",
        )

    def generate_cleaning_report(self) -> Dict[str, Any]:
        """Generate comprehensive cleaning report"""
        logger.info("📊 Generating cleaning report...")

        x_old, y_old = self.original_shape
        x_new, y_new = self.df.shape
        report = {
            "summary": {
                "original_shape": f"({x_old:,} x {y_old})",
                "final_shape": f"({x_new:,} x {y_new})",
                "total_removed": x_old - x_new,
                "retention_rate": x_new / x_old,
                "cleaning_steps": len(self.cleaning_log_),
            },
            "cleaning_steps": self.cleaning_log_,
            "data_quality": {
                "trip_duration_stats": self.df["duration_minutes"].describe().to_dict(),
                "unique_trips": self.df["TripId"].nunique() if self._has_trip_id() else 0,
                "unique_start_stations": self.df["StartStationId"].nunique(),
                "unique_end_stations": self.df["EndStationId"].nunique(),
                "round_trips_count": self.df["IsRoundTrip"].sum(),
                "round_trips_percentage": self.df["IsRoundTrip"].mean(),
            },
            "temporal_patterns": {
                "date_range": {
                    "start": self.df["StartTime"].min().isoformat(),
                    "end": self.df["StartTime"].max().isoformat(),
                },
                "busiest_hour": self.df["StartHour"].mode().iloc[0],
                "busiest_day_of_week": self.df["StartDayOfWeek"].mode().iloc[0],
            },
            "config_used": self.clean_config.to_dict(),
            "metadata": self.metadata_,
        }

        print("\n" + "=" * 60)
        print("🧹 ENHANCED DATA CLEANING REPORT")
        print("=" * 60)

        print(
            f"Original dataset shape: {report['summary']['original_shape']}\n"
            f"Final dataset shape: {report['summary']['final_shape']}\n"
            f"Total records removed: {report['summary']['total_removed']:,}\n"
            f"Retention rate: {report['summary']['retention_rate']:.1%}"
        )

        print("\n📋 Cleaning Steps Summary:")
        for step in report["cleaning_steps"]:
            print(
                f"  • {step['step']}: -{step['removed_count']:,} records "
                f"({step['removal_percentage']:.1%})"
            )

        print("\n📊 Final Dataset Quality Metrics:")
        duration_stats = report["data_quality"]["trip_duration_stats"]
        print(
            f"  • Trip Duration (min): Mean={duration_stats['mean']:.1f}, "
            f"Median={duration_stats['50%']:.1f}, "
            f"Range=[{duration_stats['min']:.1f}, {duration_stats['max']:.1f}]"
            f"  • Unique Trips: {report['data_quality']['unique_trips']:,}\n"
            f"  • Unique Stations: Start={report['data_quality']['unique_start_stations']}, "
            f"End={report['data_quality']['unique_end_stations']}"
            f"  • Round Trips: {report['data_quality']['round_trips_count']:,} "
            f"({report['data_quality']['round_trips_percentage']:.1%})"
        )
        print("\n🕐 Temporal Patterns:")
        print(
            f"  • Date Range: {report['temporal_patterns']['date_range']['start'][:10]} to "
            f"{report['temporal_patterns']['date_range']['end'][:10]}"
            f"  • Busiest Hour: {report['temporal_patterns']['busiest_hour']}:00"
        )

        logger.success("✅ Cleaning report generated successfully")
        return report

    def clean_data(self) -> pd.DataFrame:
        """
        Execute the complete enhanced data cleaning pipeline

        Returns:
            Cleaned DataFrame
        """
        logger.info("🚀 STARTING DATA CLEANING WORKFLOW")
        try:
            # Memory monitoring
            if self.memory_monitor_:
                self.memory_monitor_.log_memory_status()

            self.df = self._validate_and_standardize_columns()

            # Execute cleaning steps in order
            self.df = self.convert_datetime_columns()
            self.df = self.remove_missing_critical_data()
            self.df = self.remove_duplicate_trips()
            self.df = self.remove_temporal_anomalies()
            self.df = self.remove_spatial_anomalies()
            self.df = self.remove_statistical_outliers()
            self.df = self.add_derived_features()

            # Validate cleaned data
            validation_passed = self.validate_cleaned_data()

            if not validation_passed:
                logger.warning("⚠️ Some validation checks failed - please review the data")

            # Generate final report
            self.generate_cleaning_report()

            logger.success(f"✅ Data cleaning completed! Final shape: {self.df.shape}")

            # Memory monitoring
            if self.memory_monitor_:
                self.memory_monitor_.force_garbage_collection()

            return self.df

        except Exception as e:
            logger.error(f"Error occurred during data cleaning: {e}")
            raise
