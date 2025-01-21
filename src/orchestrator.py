import json
import shutil
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from xgboost import XGBClassifier, XGBRegressor

from config import BENCHMARKS_DIR, MODEL_DIR, N_SPLITS, RAW_DATA_DIR
from database import _find_neighbors, extract_neighbors
from evaluate import evaluate, mean_absolute_percentage_error
from features import transform_data
from helper import (
    create_crowd_levels,
    create_mask,
    group_feature_importances,
    parse_name,
    replace_substrings,
)
from logger import configure_logger
from preprocess import OutlierDetector

MODEL_MAP = {
    "reg": XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        seed=42,
    ),
    "classif": XGBClassifier(
        objective="multi:softmax",
        tree_method="hist",
        num_class=3,
        n_jobs=-1,
        seed=42,
    ),
}

INVALID_COLS = {
    "Rotterdam": [
        "Bedrijventerrein Schieveen",
        "Botlek",
        "Hoek van Holland",
        "Hoogvliet",
        "Pernis",
        "Rivium",
        "Vondelingenplaat",
        "Waalhaven",
    ],
    "Amsterdam": [
        "Driemond",
        "Kadoelen",
        "Nellestein",
        "Nieuwendammerdijk/Buiksloterdijk",
        "Spieringhorn",
        "Tuindorp Buiksloot",
        "Tuindorp Nieuwendam",
        "Waterland",
    ],
}

PARAM_SPACE = {
    "n_estimators": Integer(100, 500),
    "max_depth": Integer(3, 7),
    "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
    "gamma": Real(0, 2, prior="uniform"),
}
scorer = {
    "reg": make_scorer(mean_absolute_percentage_error, greater_is_better=False),
    "classif": make_scorer(f1_score, average="macro", labels=[0, 1, 2], pos_label=[0]),
}


def main(
    city: Literal["Amsterdam", "Rotterdam"],
    method: Literal["reg", "classif"],
    device: Literal["cpu", "gpu"],
):
    logger.info(f"Invalidated districts for {city}: {INVALID_COLS[city]}")

    df = pd.read_csv(
        RAW_DATA_DIR / f"{city.lower()}.csv",
        header=0,
        parse_dates=["timestamp"],
    ).drop(columns=INVALID_COLS[city])

    # Detect the 10 most crowded districts
    top_k = (
        df.loc[:, df.columns[1:].tolist()]
        .mean()
        .reset_index()
        .rename(columns={0: "average", "index": "district_id"})
        .nlargest(10, "average")["district_id"]
        .tolist()
    )

    # Construct Adjacency Matrix
    all_neighbors = {district: _find_neighbors(city, district) for district in top_k}

    # Remove outliers from the data
    detector = OutlierDetector(lower=25, upper=75, thresh=1.5)
    for col in df.columns[1:]:
        outliers_mask = detector._detect_outliers(df[col])
        df[col] = df.loc[~outliers_mask, col]

    benchmarks = {}
    feature_importances = defaultdict(list)
    cv = TimeSeriesSplit(n_splits=N_SPLITS)

    MAPPING = {"/": "_en_", " ": "_", "- ": "_"}
    FH = [5, 15, 30, 60]
    N_MODELS = len(all_neighbors.keys()) * len(FH)

    crowd_bins = json.load(open(MODEL_DIR / f"crowd_bins_{city.lower()}.json", "r"))

    TIME_LABEL = time.strftime("%Y%m%d-%H%M")
    CV_LABEL = "CV" if cv is not None else "split"
    EXPERIMENT_ID = f"{TIME_LABEL}-{city.lower()}-{method}-{CV_LABEL}"
    EXPERIMENT_DIR = BENCHMARKS_DIR / EXPERIMENT_ID

    for i, (fh, (district, neighbor_set)) in enumerate(product(FH, all_neighbors.items())):

        model_label = parse_name(district, MAPPING)
        Path.mkdir(MODEL_DIR / EXPERIMENT_ID / model_label, parents=True, exist_ok=True)

        logger.info(f"Training model [{i + 1}/{N_MODELS}]")
        logger.info(f"Model Details: {district} | Horizon: {fh}")

        df_min, neighbors = extract_neighbors(df, neighbor_set)
        df_min = df_min.rename(columns={district: "target"})

        df_min[f"target_{fh}"] = df_min["target"].shift(-fh)
        df_min = df_min.dropna()

        X_train, X_test, y_train, y_test = train_test_split(
            df_min.drop(columns=f"target_{fh}"),
            df_min[f"target_{fh}"],
            test_size=0.3,
            shuffle=False,
            random_state=42,
        )
        print(X_train.head())

        X_train, y_train = transform_data(X_train, y_train, "timestamp")
        X_test, y_test = transform_data(X_test, y_test, "timestamp")

        if method == "classif":
            y_train, bins = create_crowd_levels(y_train, n_bins=3, target=district)
            y_test = create_mask(y_test, bins)
            crowd_bins[district] = bins

        model = MODEL_MAP[method].set_params(device=device)

        try:
            logger.info("Tuning hyperparameters")
            baeys_cv = BayesSearchCV(
                model,
                PARAM_SPACE,
                n_iter=30,
                cv=cv,
                scoring=scorer[method],
                n_jobs=1,
                random_state=42,
            )

            start_time = time.time()
            baeys_cv.fit(X_train, y_train)
            tune_time = time.time() - start_time
            logger.info(f"Local tuning time: {tune_time:.3f} seconds")

            best_params = baeys_cv.best_params_
            model.set_params(**best_params)

            logger.debug("Training")
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            MODEL_ID = f"{model_label}_{fh}"
            model_path = MODEL_DIR / EXPERIMENT_ID / model_label / f"{MODEL_ID}.joblib"
            joblib.dump(model, model_path)

            logger.info(f"Model saved at root folder: {EXPERIMENT_DIR.name}")
            logger.debug(f"Leaf path: {model_label}/{MODEL_ID}")
            logger.debug("Evaluating performance")

            start_time = time.time()
            y_pred = model.predict(X_test)
            inference_time = time.time() - start_time
            result = evaluate(y_test, y_pred, method)

            result["Tuning"] = tune_time
            result["Train"] = train_time
            result["Inference"] = inference_time

            benchmarks[(district, fh)] = result
        except Exception as e:
            logger.error(f"Cannot evaluate model on test set | Skipping model | {e}")
            shutil.rmtree(MODEL_DIR / EXPERIMENT_ID / model_label)
            continue

        logger.info("Extracting feature importances")
        names = replace_substrings(X_train.columns.tolist(), neighbors, district)
        feature_importances[fh].extend(
            zip(
                [fh] * len(names),
                names,
                model.feature_importances_,
            )
        )

    try:
        Path.mkdir(EXPERIMENT_DIR, parents=True, exist_ok=True)

        logger.info("Saving benchmarks")
        benchmarks = pd.DataFrame(
            [
                {"district": district, "fh": fh, **metrics}
                for (district, fh), metrics in benchmarks.items()
            ]
        )
        benchmarks.to_csv(EXPERIMENT_DIR / "benchmarks.csv")

        logger.info("Saving feature importances")
        grouped_importances = group_feature_importances(FH, feature_importances)
        grouped_importances.to_csv(EXPERIMENT_DIR / "feature-importances.csv")

        if method == "classif":
            with open(MODEL_DIR / f"crowd_bins_{city.lower()}.json", "w") as f:
                json.dump(crowd_bins, f)
    except Exception as e:
        logger.error(f"Could not save files | Aborting | {e}")
        shutil.rmtree(EXPERIMENT_DIR)


if __name__ == "__main__":
    logger = configure_logger(log_file="orchestrator")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--city",
        type=str,
        help="Select city to run experiment on",
        choices=["Amsterdam", "Rotterdam"],
        default="Rotterdam",
    )
    parser.add_argument(
        "--method",
        type=str,
        help="Select the model variation",
        choices=["reg", "classif"],
        default="reg",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Select whether you want to use CPU or GPU",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if v is not None}

    logger.debug(f"Executing main with kwargs: {kwargs}")
    main(**kwargs)
