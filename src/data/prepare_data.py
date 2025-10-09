import warnings
from typing import Optional

import pandas as pd
from loguru import logger
from sklearn import set_config
from sklearn.preprocessing import MinMaxScaler

from config import paths
from config.constants import (
    GROUP_COLUMN,
    INVALID_DISTRICTS,
    TARGET_COLUMN,
    TEMPORAL_COLUMNS,
    TIME_COLUMN,
)
from data.preprocess import create_forecasting_targets
from model.ggbdp import CacheConfig, DemandForecaster, ForecastConfig
from utils.helper import assert_equal, split_X_y

warnings.filterwarnings("ignore")
set_config(transform_output="pandas")

TIME_FEATURES = ["hour", "dayofweek", "month", "quarter"]
NUM_KERNELS = [6, 3, 6, 4]
INPUT_RANGES = [(0, 23), (0, 6), (1, 12), (1, 4)]
DATASETS = {
    "rotterdam": "rotterdam_20250718",
    "amsterdam": "amsterdam_20250718",
    "hague": "hague_20250718",
    "citi": "tripdata_cleaned_resampled_1h",
    "citi_r": "region_demand",
}


def main(
    dataset,
    fh,
    groupby,
    time_periods,
    transit,
    holiday_country,
    lags,
    windows,
    rolling,
    use_diff,
    suffix,
):
    # Load sorted dataframe
    if (dataset_name := DATASETS.get(dataset, None)) is None:
        raise ValueError(f"Dataset {dataset} not found. Valid datasets: {DATASETS.keys()}.")

    dataset_name = f"{dataset_name}_{fh}min" if dataset_name == "citi_r" else dataset_name

    data = pd.read_parquet(paths.INTERIM_DATA_DIR / f"{dataset_name}.parquet.gzip")
    data_cleaned = data.loc[~data[GROUP_COLUMN].isin(INVALID_DISTRICTS[dataset])]

    # Define the forecast horizon based on input.
    # If fh is a list, split into min and max else set fh_max = fh and start from 1
    match len(fh):
        case 1:
            fh_min, fh_max = 1, fh[0]
        case 2:
            fh_min, fh_max = fh
        case _:
            raise ValueError("fh must be a list of length 1 or 2.")

    # Display basic info
    print(
        f"Number of rows: {data_cleaned.shape[0]:,}\n"
        f"Number of groups: {data_cleaned[GROUP_COLUMN].nunique()}\n"
        f"Time range: "
        f"{pd.to_datetime(data_cleaned[TIME_COLUMN].min(), unit='s')} --> "
        f"{pd.to_datetime(data_cleaned[TIME_COLUMN].max(), unit='s')}"
    )

    # Ensure correct dtype safely
    df = create_forecasting_targets(
        data_cleaned,
        GROUP_COLUMN,
        TEMPORAL_COLUMNS,
        max_horizon=fh_max,
        min_horizon=fh_min,
        merge_format="long",
    )

    # Ensure that the test set is 10%
    X_train, y_train, X_test, y_test = split_X_y(
        df.drop(TARGET_COLUMN, axis=1),
        df[TARGET_COLUMN],
        test_size=0.1,
        time_index=False,
    )

    # Extract the `burn-in` set as 10% of the training set
    X_burn, _, X_train, y_train = split_X_y(
        X_train,
        y_train,
        test_size=0.9,
        time_index=False,
    )

    # Ensure full coverage between splits
    X_burn, X_train = assert_equal(X_burn, X_train, errors="coerce")
    X_burn, X_test = assert_equal(X_burn, X_test, errors="coerce")

    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]

    cache_config = CacheConfig(
        enabled=True,
        directory=paths.MODEL_DIR / "cache" / dataset,
        verbose=0,
    )

    config = ForecastConfig(
        dataset,
        group_col=groupby,
        time_features=TIME_FEATURES,
        num_kernels=NUM_KERNELS,
        input_ranges=INPUT_RANGES,
        time_periods=time_periods,
        transit_patterns=transit,
        country_code=holiday_country,
        lags=lags,
        windows=windows,
        rolling_stats=rolling,
        fourier_harmonics=30,
        fourier_window=max(*lags, *windows),
        quantiles=[0, 0.3, 0.7, 1],
        use_diff=use_diff,
        diff_orders=[24, 168],
        cache_config=cache_config,
    )

    # Create forecaster with dependency injection
    forecaster = DemandForecaster(
        config=config,
        scaler=MinMaxScaler(),
        chunk_size=10000,
    )

    _ = forecaster.fit_transform(X_burn)
    X_train_trans: pd.DataFrame = forecaster.transform(X_train)  # type: ignore
    X_test_trans: pd.DataFrame = forecaster.transform(X_test)  # type: ignore

    window = max(windows[dataset])

    suffix = f"_{suffix}" if suffix else ""
    forecaster.save(paths.MODEL_DIR / f"ggbdp_{dataset}_h{fh}_w{window}{suffix}.pkl")

    X_trans = pd.concat(
        [X_train_trans, X_test_trans],
        axis=0,
        keys=["train", "test"],
        names=["split"],
    )
    y = pd.concat(
        [y_train, y_test],
        axis=0,
        keys=["train", "test"],
        names=["split"],
    )

    save_data(X_trans, y, fh, window=max(windows), suffix=suffix)


def save_data(X, y, fh, *, window, suffix, dir_name: Optional[str] = None):

    save_root = paths.PROCESSED_DATA_DIR
    if dir_name is not None:
        save_root = save_root / dir_name.lower()

    if not save_root.exists():
        save_root.mkdir(parents=True)
        logger.debug(f"Created directory {save_root.name}")

    # Construct save path
    save_path = save_root / f"h{fh}_w{window}_{suffix}"

    tmp = pd.concat([X, y.set_index(X.index)], axis=1)

    if (nan_cols := [col for col in tmp.columns if tmp[col].isna().all()]) != []:
        logger.warning(f"NaN columns found: {nan_cols}. Dropping...")
        tmp = tmp.drop(columns=nan_cols)

    assert tmp.shape[0] > 0, "Empty DataFrame found"
    tmp.to_parquet(save_path, partition_cols=["split"], compression="gzip")

    logger.info(f"Full set ({tmp.shape[0]:,} rows) saved to {save_path.stem}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare data for ML training")

    parser.add_argument("-d", "--dataset", help="Select source dataset", type=str)
    parser.add_argument("-f", "--fh", help="Select fh (range, optional)", nargs="+", type=int)
    parser.add_argument("-s", "--suffix", help="Select suffix", type=str, default="v1")
    parser.add_argument(
        "-g",
        "--groupby",
        help="Group by column(s) during feature extraction",
        nargs="+",
        type=str,
    )
    parser.add_argument("--time_periods", help="Include time periods", action="store_true")
    parser.add_argument("--transit", help="Include transit patterns", action="store_true")
    parser.add_argument("-l", "--lags", help="Include lags", nargs="+", type=int)
    parser.add_argument("-w", "--windows", help="Include windows", nargs="+", type=int)
    parser.add_argument("-H", "--holiday_country", help="Include holiday country", type=str)
    parser.add_argument("-r", "--rolling", help="Include rolling stats", nargs="+", type=str)
    parser.add_argument("--diff", help="Difference ts during adjustment", action="store_true")

    args = parser.parse_args()
    return {k: v for k, v in vars(args).items() if v is not None}


if __name__ == "__main__":
    from pprint import pprint

    kwargs = parse_args()
    logger.debug(f"Calling main with arguments: {kwargs=}")
    pprint(kwargs, indent=2)
    main(**kwargs)
