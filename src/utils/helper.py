import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, overload

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from xgboost import XGBModel

from config import paths
from config.constants import (DATE_FORMAT, GROUP_COLUMN, TARGET_COLUMN,
                              TIME_COLUMN)
from utils.models import IndexType, MLDataConfig, TargetType, TaskType


@overload
def create_crowd_levels(
    data: pd.DataFrame,
    num_classes: int,
    target: TargetType,
    group_col: None = None,
) -> Tuple[pd.Series, dict[str, np.ndarray]]: ...


@overload
def create_crowd_levels(
    data: pd.DataFrame,
    num_classes: int,
    target: TargetType,
    group_col: str,
) -> Tuple[dict[str, pd.Series], dict[str, np.ndarray]]: ...


def create_crowd_levels(
    data: pd.DataFrame,
    num_classes: int,
    target: TargetType,
    group_col: Optional[str] = None,
) -> Tuple[pd.Series | Dict[str, pd.Series], Dict[str, np.ndarray]]:
    """
    Discretize data into quantile-based bins, either globally or per group.

    Args:
        data : The input DataFrame to discretize.
        num_classes : The number of bins to use for discretization.
        target : The column name of the target variable to discretize.
        group_col : The column name of the group variable to discretize per group.
            If None, the data is discretized globally.

    Returns:
        A tuple containing:
        - levels: A Series of bin labels for each data point, or a dict of such Series
          if group_col is given.
        - crowd_bins: A dict of bin edges for each group, or a single array if group_col
          is None.
    """

    GROUP_BINS: Dict[int, List[float]] = {
        3: [0, 0.3, 0.7, 1],
        5: [0, 0.25, 0.4, 0.85, 1],
    }

    def group_crowd_levels(data: pd.Series) -> Tuple[pd.Series, np.ndarray]:
        """
        Discretize data into quantile-based bins and return labels and bin edges.

        Args:
            data : The input series to discretize.

        Returns:
            A tuple containing:
            - labels: A Series of bin labels for each data point.
            - bins: An array of bin edges.

        """

        # Discretize using qcut with ranking to handle ties
        labels = pd.qcut(data.rank(method="first"), q=GROUP_BINS[num_classes], labels=False)

        # Get the actual value bins for each label
        bin_df = pd.DataFrame({target: data, "label": labels})
        bin_ranges = (
            bin_df.groupby("label")[target].agg(["min", "max"]).reset_index(drop=True)
        )

        # Flatten into bin edges, remove duplicates, prepend 0 and append inf
        edges = sorted(set(bin_ranges.values.flatten()))
        bins = np.concatenate(([-1], edges[1:-1], [np.inf])).tolist()

        return labels.astype(np.uint8), bins

    if group_col:
        groups = data.index.get_level_values(group_col).unique()

        all_levels, all_crowd_bins = {}, {}
        for group in groups:
            mask = data.index.get_level_values(group_col) == group
            levels, bins = group_crowd_levels(data[mask][target])

            all_levels[group] = levels
            all_crowd_bins[group] = bins

    else:
        levels, bins = group_crowd_levels(data[target])
        all_levels = {target: levels}
        all_crowd_bins = {target: bins}

    return all_levels, all_crowd_bins


def create_mask(
    y: pd.DataFrame,
    bins: dict,
    column: TargetType = TARGET_COLUMN,
    log_distribution: bool = False,
) -> pd.Series:
    """
    Create a mask for y based on bin edges for each district.

    Args:
        y : The test data with district_id in the index.
        bins : A dictionary with district_id as keys and bin edges as values.
        column (optional) : The column to bin, by default "target_crowd".
        log_distribution (default False) : Whether to log the distribution of bins for each
            district.

    Returns:
        The mask for y.
    """

    def bin_and_label(group_df):
        group_id = group_df.name
        boundaries = bins[group_id]

        if not np.all(np.diff(boundaries) > 0):
            raise ValueError(f"Bin edges for group '{group_id}' must be strictly increasing.")

        values = group_df[column]
        bin_indices = np.digitize(values, bins=boundaries, right=True) - 1

        # Create readable labels
        binned = pd.Categorical.from_codes(
            bin_indices,
            categories=list(range(len(boundaries) - 1)),
            ordered=True,
        )

        if log_distribution:
            counts = pd.Series(binned).value_counts().sort_index()
            logger.debug(f"Group '{group_id}' bin distribution:\n{counts.to_string()}")

        return pd.DataFrame({column: binned}, index=group_df.index)

    if "district_id" not in y.index.names:
        raise ValueError("y must have 'district_id' in the index.")

    test_series = y.groupby(level="district_id", as_index=False).apply(bin_and_label)
    return pd.Series(test_series.values.flatten(), index=y.index)


@overload
def load_data(config: MLDataConfig,
    *,
    prefix: Optional[str] = None,
    dropna: bool = False,
    return_X_y: Literal[False],
    index: Optional[IndexType] = None,
    **kwargs,) -> pd.DataFrame: ...

@overload
def load_data(config: MLDataConfig,
    *,
    prefix: Optional[str] = None,
    dropna: bool = False,
    return_X_y: Literal[True],
    index: Optional[IndexType] = None,
    **kwargs,) -> Tuple[pd.DataFrame, pd.DataFrame | pd.Series]: ...

def load_data(
    config: MLDataConfig,
    *,
    prefix: Optional[str] = None,
    dropna: bool = False,
    return_X_y: bool = False,
    index: Optional[IndexType] = None,
    **kwargs,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame | pd.Series]:
    """Load processed time series data from disk.

    Loads dataset based on split, forecast horizon, window size, and optional city.
    Supports returning feature/target splits and index setting.

    Args:
        fh : Forecast horizon (in minutes or steps), used in filename as `h{fh}`.
        window : Lookback window size, used in filename as `w{window}`.
        base_dir (optional): Base directory containing processed data. Defaults
            to `paths.PROCESSED_DATA_DIR`. Must be provided if `city` is used unless a global
            config exists.
        target (optional): Column name(s) to treat as target(s). Required if
            `return_X_y=True`.
        prefix (optional): Prefix to append to the filename, usually the version of
            the data (e.g., 'train', 'full').
        suffix (optional): Suffix to append to the filename, usually the split or a variation
            of the data (e.g., 'v1', 'scaled').
        dropna (default False): If True, drops rows with NaN values.
        return_X_y (default False): If True, returns tuple (X, y) where X is features and
            y is target(s).
        index (optional): Column(s) to set as index after loading.
        extension (default "parquet.gzip"): File extension (e.g., 'parquet.gzip', 'parquet').
        **kwargs : Additional keyword arguments to pass to `pd.read_parquet`.

    Returns:
        - If `return_X_y=False`: Returns the full DataFrame.
        - If `return_X_y=True`: Returns (X, y) as a tuple.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If `return_X_y=True` but `target` is not specified.

    Examples:
        >>> X, y = load_data(
        ...     config,
        ...     target="inflow",
        ...     return_X_y=True,
        ...     dropna=True
        ... )
    """
    # Handle prefix and suffix
    suffix_str = f"_{config.suffix}" if config.suffix else ""
    prefix_str = f"{prefix}_" if prefix else ""

    filename = f"{prefix_str}h{config.fh}_w{config.window}{suffix_str}"

    if config.extension is not None:
        filename = f"{filename}.{config.extension}"

    # Validate target
    if return_X_y and config.target is None:
        raise ValueError("target must be specified when return_X_y=True")

    # Build filename
    logger.debug(
        f"Loading {prefix_str}{suffix_str[1:]} data "
        f"({config.fh} horizon, {config.window} window)"
    )

    file_path = config.base_dir / filename

    # Validate file existence
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path.resolve()}")

    # Load data
    logger.debug(f"Reading file: {file_path.name}")
    data = pd.read_parquet(file_path, **kwargs)

    if (filters := kwargs.get("filters")) is not None:
        # `filters` is a list of tuples (column, operator, value)
        # source: https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html
        for column, op, _ in filters:
            if op == "=":
                logger.debug(f"Filtering out {column} due to equality constraint")
                if column in data.columns:
                    data = data.drop(columns=column)
                elif column in data.index.names:
                    data = data.droplevel(column, axis=0)

    if dropna:
        initial_len = len(data)
        data = data.dropna()
        final_len = len(data)
        logger.debug(f"Dropped {initial_len - final_len} rows containing NaN values")

    if index:
        data = data.set_index(index)
        logger.debug(f"  Set index: {index}")

    samples, features = data.shape
    log_message = (
        f"Loaded {samples:,} samples ({features - len(config.target)} features) from "
        f"{file_path.parent.name}/{file_path.stem}"
    )
    if return_X_y:
        if not config.target:
            raise ValueError("Argument `target` must be specified when `return_X_y=True`")

        X = data.drop(columns=config.target, errors="ignore")
        y = data[config.target].copy()

        logger.info(f"{log_message}; returned X and y")
        return X, y

    logger.info(log_message)
    return data


def get_last_backup_timestamp() -> float | None:
    """
    Retrieve the timestamp of the last backup from a file.

    This function reads the timestamp of the last backup from a text file
    located at `paths.BACKUPS_DIR/last_backup.txt`. The timestamp is expected
    to be in the format specified by `paths.DATE_FORMAT`.

    Returns
    -------
    float | None
        The UTC timestamp of the last backup if the file exists, otherwise
        None if the file is not found.
    """

    try:
        with open(f"{paths.BACKUPS_DIR}/last_backup.txt", "r", encoding="utf-8") as f:
            obj = f.read().strip()
            return (
                datetime.strptime(obj, DATE_FORMAT).replace(tzinfo=timezone.utc).timestamp()
            )
    except FileNotFoundError:
        return None  # No previous backups


def update_last_backup_timestamp() -> str:
    """
    Update the timestamp of the last backup in a file.

    This function writes the current UTC timestamp to a text file located
    at `paths.BACKUPS_DIR/last_backup.txt`. The timestamp is written in the
    format specified by `paths.DATE_FORMAT`.

    Returns
    -------
    str
        The current UTC timestamp as a string.
    """
    with open(f"{paths.BACKUPS_DIR}/last_backup.txt", "w", newline="") as f:
        f.write(datetime.now(timezone.utc).strftime(DATE_FORMAT))

    return f.read()


def load_models(
    city: Literal["Amsterdam", "Rotterdam"],
    method: Literal["reg", "classif"],
    model_file_extension: str = ".joblib",
) -> dict[Tuple[int, int], XGBModel]:
    """
    Loads models from a compressed archive based on city and method.

    Parameters
    ----------
    city : str
        City name to locate the models.
    method : Literal["reg", "classif"]
        Method type ("reg" for regression, "classif" for classification).
    model_dir : Path
        Path to the directory containing the models.zip file.
    model_file_extension : str, optional
        File extension for model files (default is ".joblib").

    Returns
    -------
    dict[Tuple[int, int], XGBModel]
        dictionary mapping (district, forecast horizon) to loaded models.
    """
    models = {}

    models_dir = paths.MODEL_DIR / f"{city}-{method}"

    if not models_dir.exists() or not models_dir.is_dir():
        print(f"Model directory not found: {models_dir}")
        return models

    for district in models_dir.iterdir():
        if not district.is_dir():
            continue

        for model_path in district.iterdir():
            if model_path.suffix != model_file_extension:
                continue

            match = re.split(r"_(\d+)", model_path.stem)
            if len(match) < 3:
                print(f"Skipping invalid model file: {model_path}")
                continue

            try:
                district_id, fh = int(match[0]), int(match[1])
                model = joblib.load(model_path)
                models[(district_id, fh)] = model

            except ValueError as e:
                print(f"Failed to parse model filename {model_path.name}: {e}")
            except Exception as e:
                print(f"Failed to load model {model_path.name}: {e}")

    return models


def temporal_split(
    df: pd.DataFrame,
    target_col: str | List[str],
    return_X_y: bool = True,
    time_col: str = TIME_COLUMN,
    test_size: float = 0.3,
) -> (
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    | Tuple[pd.DataFrame, pd.DataFrame]
):
    """
    Split a dataframe into training and testing sets using a temporal split.

    The split is done by selecting a cutoff time and splitting the data at that time.
    The first `test_size` percentage of the data is used for testing, and the remaining
    data is used for training.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to split.
    time_col : str
        The column name of the datetime column.
    target_col : str
        The column name of the target variable.
    test_size : float, optional
        The proportion of the data to include in the test split, by default 0.2.
    return_X_y : bool, optional
        If True, split the data into X and y, by default False.

    Returns
    -------
    Union[
        Tuple[pd.DataFrame, pd.DataFrame],
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ]
        If return_X_y is False, returns (train, test).
        If return_X_y is True, returns (X_train, X_test, y_train, y_test).

    """
    if not (0 < test_size < 1):
        raise ValueError("test_size must be a float between 0 and 1.")

    df = df.copy()

    cutoff_time = df[time_col].quantile(1 - test_size)

    # Split the data
    df_train = df.loc[df[time_col] < cutoff_time]
    df_test = df.loc[df[time_col] >= cutoff_time]

    if return_X_y:
        X_train, y_train = df_train.drop(target_col, axis=1), df_train[target_col]
        X_test, y_test = df_test.drop(target_col, axis=1), df_test[target_col]

        return X_train, X_test, y_train, y_test
    return df_train, df_test


def load_meta(
    path: Path,
    columns: Optional[List[str]] = None,
    crs=4326,
):
    """Load station metadata from CSV with optional column selection and geometry parsing.

    Args:
        path: Path to CSV file containing station metadata.
        columns: List of column names to load. If None, all columns are loaded.
                 Must include 'geometry' if present, as it's required for spatial parsing.
        crs: Coordinate Reference System to assign to the GeoDataFrame. Default is EPSG:4326.

    Returns:
        GeoDataFrame with geometry column parsed from WKT and standard column renaming
        applied.

    Raises:
        ValueError: If 'geometry' column is missing from the data or `columns` list.
        FileNotFoundError: If the file at `path` does not exist.
        KeyError: If specified columns (other than geometry) are missing.
    """
    # Configure read options
    kwargs = {"filepath_or_buffer": path}

    if columns is not None:
        if "geometry" not in columns:
            raise ValueError(
                "Column 'geometry' required for spatial data (must be included in 'columns')"
            )
        kwargs["usecols"] = columns  # type: ignore

    df = pd.read_csv(**kwargs)

    if "geometry" not in df.columns:
        raise ValueError("CSV must contain a 'geometry' column with WKT-formatted strings")

    # Rename only the known standard columns if they exist
    rename_columns = {
        "StationId": "station_id",
        "GroupStationId": "group_id",
        "ExpectedStationSize": "capacity",
    }
    df_renamed = df.rename(columns=rename_columns)

    # Filter rename map to only those present
    actual_renames = {k: v for k, v in rename_columns.items() if k in df.columns}
    df_renamed = df.rename(columns=actual_renames)

    # Parse geometry from WKT
    try:
        geometry = gpd.GeoSeries.from_wkt(df_renamed["geometry"], crs=crs)
    except Exception as e:
        raise ValueError(f"Invalid WKT in 'geometry' column: {e}") from e

    # Construct GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df_renamed,
        geometry=geometry,
        crs=crs,
    )

    return gdf


def assert_equal(
    X_fit,
    X_transform,
    errors: Literal["coerce", "raise"] = "coerce",
):
    """Assert that fit and transform groups are equal.

    Raises an error if there is any difference between the groups in X_fit
    and X_transform. Each value in X_fit[GROUP_COLUMN] should exist in
    X_transform[GROUP_COLUMN] and vice versa.
    """
    diff = set(X_fit[GROUP_COLUMN].unique()) ^ set(X_transform[GROUP_COLUMN].unique())
    if diff != set():
        err_message = (
            f"'fit' and 'transform' groups expected to be identical, "
            f"found {len(diff)} differences: {diff}."
        )
        if errors == "coerce":
            logger.warning(err_message + " Coercing to common groups")

            X_fit = X_fit.loc[~X_fit[GROUP_COLUMN].isin(diff)].copy()
            X_transform = X_transform.loc[~X_transform[GROUP_COLUMN].isin(diff)].copy()

            return assert_equal(X_fit, X_transform)
        else:
            raise ValueError(err_message)

    logger.info("Groups in 'fit' and 'transform' are identical.")
    return X_fit, X_transform


def split_X_y(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    *,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    time_index: bool = True,
    time_col: str = TIME_COLUMN,
    sort: bool = False,
) -> Tuple:
    """
    Split features and target into train, validation, and test sets based on timestamp index.

    This function performs time-series-aware splitting using the 'timestamp' level of a
        MultiIndex. It ensures chronological order is preserved (no future leakage) by
        sorting timestamps and splitting based on proportions from the end.

    The data is split as:
        - Test: most recent (test_size)
        - Validation: just before test (val_size, if provided)
        - Train: all earlier data

    Args:
        X: Feature DataFrame with MultiIndex containing 'timestamp' level.
        y: Target Series with MultiIndex containing 'timestamp' level.
        test_size: Proportion of data to reserve for testing (between 0 and 1).
        val_size: Optional proportion for validation (must be < 1 - test_size).

    Returns:
        Tuple containing:
            - X_train, y_train: Training data
            - X_val, y_val: Validation data (if val_size provided)
            - X_test, y_test: Test data

    Raises:
        ValueError: If test_size or val_size are invalid.
        KeyError: If 'timestamp' level is not in the index.
        IndexError: If resulting splits would be empty.

    Example:
        >>> X_train, y_train, X_val, y_val, X_test, y_test = split_X_y(
        >>>     X, y, test_size=0.2, val_size=0.1)
        >>> # Train: 70%, Val: 10%, Test: 20% (chronologically ordered)

    Notes:
        - Assumes MultiIndex with a timestamp level of datetime (or comparable type).
        - Splits are made in temporal order: oldest -> train, middle -> val, newest -> test.
        - Uses actual timestamp values, not quantiles, to avoid distribution assumptions.
    """

    # Extract timestamp level
    try:
        if time_index:
            timestamps = X.index.get_level_values(time_col)
        else:
            timestamps = X[time_col]
        ts_series = pd.Series(timestamps)
    except KeyError:
        raise KeyError(f"Index must have a '{TIME_COLUMN}' level, got {X.index}")

    # Ensure alignment between X and y
    if not X.index.equals(y.index):
        raise ValueError("X and y must have identical indices")

    if sort:
        # Sort by timestamp to ensure correct order
        sorted_idx = ts_series.sort_values().index
        X = X.loc[sorted_idx]
        y = y.loc[sorted_idx]
        ts_series = ts_series.loc[sorted_idx]

    # Get cutoff positions based on proportions from the end
    test_cutoff = ts_series.quantile(1 - test_size)
    val_cutoff = (
        ts_series.quantile(1 - val_size - test_size) if val_size is not None else test_cutoff
    )

    # Split X and y
    X_train = X.loc[timestamps <= val_cutoff]
    y_train = y.loc[timestamps <= val_cutoff]

    X_test = X.loc[timestamps > test_cutoff]
    y_test = y.loc[timestamps > test_cutoff]

    if val_size is not None:
        mask = (timestamps > val_cutoff) & (timestamps <= test_cutoff)
        X_val = X.loc[mask]
        y_val = y.loc[mask]

        return X_train, y_train, X_val, y_val, X_test, y_test

    return X_train, y_train, X_test, y_test


def load_hparams(dataset: str, task: TaskType, fh: int) -> Dict[str, Any]:
    """
    Load hyperparameters from a JSON file.

    If the file does not exist, or the JSON is malformed, or the file does not
    contain the specified dataset/task/fh combination, the function will return a
    default set of hyperparameters.

    The JSON file is expected to have the following structure:
    {
        "dataset1": {
            "task1": {
                "fh1": {...},
                "fh2": {...}
            },
            "task2": {...}
        },
        "dataset2": {...}
    }

    Args:
        dataset: The name of the dataset.
        task: The name of the task.
        fh: The forecast horizon.

    Returns:
        A dictionary containing the hyperparameters for the specified combination.
    """
    import json

    model_path = paths.MODEL_DIR / "hparams.json"
    try:
        with open(model_path, "r", encoding="utf-8") as file:
            hparams = json.load(file)

        if not isinstance(hparams, dict):
            raise ValueError(
                f"Expected a dictionary in {model_path}, got {type(hparams).__name__}"
            )

        logger.debug(f"Loaded hyperparameters from {model_path.name}")
        return hparams[dataset][task.value][str(fh)]

    except FileNotFoundError:
        logger.exception(f"Error: The file {model_path} was not found.")
    except json.JSONDecodeError:
        logger.exception(f"Error: Failed to decode JSON from the file {model_path}.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}")

    logger.warning(
        f"Could not find hyperparameters for {dataset=}, {task=}, {fh=}. "
        f"Returning default values."
    )
    return {
        "n_estimators": 350,
        "learning_rate": 0.15,
        "max_depth": 6,
        "colsample_bytree": 0.9,
        "min_child_weight": 6,
        "gamma": 0.47,
        "alpha": 0.001,
    }
    }
    }
