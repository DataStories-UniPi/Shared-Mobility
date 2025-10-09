"""Functions for data preprocessing tasks."""

from typing import List, Literal, Optional

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from config.constants import FH, GROUP_COLUMN, TARGET_COLUMN, TIME_COLUMN


def create_forecasting_targets(
    data: pd.DataFrame,
    group_col: str = GROUP_COLUMN,
    target_columns: Optional[str | List[str]] = None,
    max_horizon: int = FH,
    min_horizon: int = 1,
    merge_format: Literal["long", "wide"] = "long",
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Create time series forecasting targets by shifting specified columns forward.

    This function creates multiple forecasting horizons for time series data by
    shifting target columns forward in time, grouped by a specified identifier.
    Each horizon represents how many time steps ahead to predict.

    Args:
        data : Input time series data with columns to be shifted as targets. Must contain
        the time column (assumed to be 'timestamp') and the group column
        group_col : Column name containing station identifiers for grouping.
        target_columns (default=["inbound", "outbound"]): List of column names to create
            targets for.
        max_horizon : Maximum forecasting horizon.
        min_horizon : Minimum forecasting horizon.
        merge_format (default='long') : Output format:
            - "long": One row per horizon (includes a 'horizon' column).
            - "wide": One row per group/time, with separate target columns per horizon.
        dropna : (default=True) Whether to drop rows containing NaN values after shifting.

    Returns:
        DataFrame with original data plus target columns for each horizon.
        Format depends on `merge_format`:
        - "long": Includes a 'horizon' column; rows repeated per horizon.
        - "wide": Target columns suffixed with horizon (e.g., target_inbound_h1).

    Raises:
        ValueError
            If required columns are missing from the input data.
        KeyError
            If specified target columns don't exist in the data.

    Examples:
        >>> data = pd.DataFrame({
        ...     'station_id': [1, 1, 1, 2, 2, 2],
        ...     'timestamp': pd.date_range('2023-01-01', periods=6, freq='H')[:3].tolist(),
        ...     'inbound': [10, 20, 30, 15, 25, 35],
        ...     'outbound': [5, 15, 25, 8, 18, 28]
        ... })
        >>> result = create_forecasting_targets(data, group_col='station_id', horizons=[1, 2])
        >>> print(result[['station_id', 'inbound', 'target_inbound', 'horizon']].head())
        station_id  inbound  target_inbound  horizon
        0           1       10            20.0        1
        1           1       20            30.0        1
        2           1       10            30.0        2
        3           1       20             NaN        2
    """
    # Set defaults
    target_columns = target_columns or TARGET_COLUMN
    horizons = list(range(min_horizon, max_horizon + 1))

    # Validate inputs
    missing_cols = [col for col in [group_col, TIME_COLUMN] if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in data: {missing_cols}")

    missing_targets = [col for col in target_columns if col not in data.columns]
    if missing_targets:
        raise KeyError(f"Target columns not found in data: {missing_targets}")

    if not horizons:
        raise ValueError("Horizons list cannot be empty.")

    def process_group_with_horizons(group: pd.DataFrame) -> List[pd.DataFrame]:
        """Apply shifting for all horizons on a single group."""
        group = group.sort_values(TIME_COLUMN).copy()

        result_dfs = []
        for h in horizons:
            df_h = group.copy()
            df_h["horizon"] = h

            for col in target_columns:
                df_h[f"target_{col}"] = group[col].shift(-h).astype("Int64")
            result_dfs.append(df_h)

        return result_dfs

    # Apply processing per group
    processed_groups = []
    groups = data.groupby(group_col)

    for _, group_df in tqdm(groups, "Creating target forecasts", groups.ngroups):
        group_results = process_group_with_horizons(group_df)
        processed_groups.extend(group_results)

    # Combine all results
    result = pd.concat(processed_groups, ignore_index=True)

    # Drop rows with NaN in any target (if requested)
    if dropna:
        target_cols = [f"target_{col}" for col in target_columns]
        result = result.dropna(subset=target_cols)

    # Convert targets to int safely (if possible)
    for col in target_columns:
        target_col = f"target_{col}"
        if target_col in result.columns:
            # Only cast if no NaNs and compatible
            if result[target_col].notna().all():
                result[target_col] = result[target_col].astype(int)
            else:
                result[target_col] = result[target_col].astype("Int64")  # nullable int

    # Handle output format
    if merge_format == "wide":
        # Pivot target columns per horizon
        pivot_cols = [f"target_{col}" for col in target_columns]
        # Widen using pivot or column stacking
        result = result.pivot_table(
            index=[c for c in result.columns if c not in ["horizon"] + pivot_cols],
            columns="horizon",
            values=pivot_cols,
            aggfunc="first",
        ).reset_index()

        # Flatten column names
        if isinstance(result.columns, pd.MultiIndex):
            new_cols = []
            for a, b in result.columns:
                if a == "":
                    new_cols.append(b)
                elif b == "":
                    new_cols.append(a)
                else:
                    new_cols.append(f"{a}_h{b}")
            result.columns = new_cols
        result = result.reset_index(drop=True)

    # Keep horizon as regular column in "long" format
    elif merge_format == "long":
        result = result.reset_index(drop=True)

    if len(horizons) == 1:
        result = result.drop("horizon", axis=1)

    return result


def points_in_boundaries(
    df: pd.DataFrame,
    city_boundaries: gpd.GeoDataFrame,
    ts_col: Optional[str] = "ts",
) -> pd.DataFrame:
    """
    Join a DataFrame of points with a GeoDataFrame of city boundaries.

    Args:
        df: DataFrame of points with columns 'latitude' and 'longitude'.
        city_boundaries: GeoDataFrame of city boundaries with columns 'City' and 'name'.
        ts_col (optional): Column name in df to use as timestamp, by default 'ts'.

    Returns:
        Joined DataFrame with columns 'district_id', 'timestamp', 'city', and geometry.
    """
    df = df.drop_duplicates().reset_index(drop=True)
    df = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"], crs=4326),
        crs=4326,
    )  # type: ignore

    df_left = (
        pd.DataFrame(
            data=df.sindex.query(city_boundaries.geometry, predicate="intersects").T,
            columns=["district_id", "point_id"],
        )
        .reset_index(drop=True)
        .set_index("district_id")
        .join(city_boundaries)
    )

    df_right = (
        df.iloc[df_left["point_id"]][ts_col]
        .reset_index()
        .rename(
            columns={
                "index": "point_id",
                ts_col: "timestamp",
            }
        )
    )

    return pd.merge(df_left, df_right, on="point_id").rename(
        columns={
            "City": "city",
            "name": "district_id",
        }
    )


def build_timeseries(data: pd.DataFrame):
    """
    Build a timeseries DataFrame of crowd counts for each district and city.

    Args:
        data : DataFrame containing  `district_id`, `city`, `timestamp`, and `point_id`.

    Returns:
        DataFrame with columns `district_id`, `timestamp`, `crowd`, and `city`,
        where `crowd` represents the count of points for each timestamp.
    """

    ts = (
        data.groupby(by=["district_id", "city", "timestamp"])
        .agg({"point_id": "count"})
        .rename({"point_id": "crowd"}, axis=1)
        .sort_values(by="timestamp")
        .reset_index()
    )

    return ts[["district_id", "timestamp", "crowd", "city"]]
