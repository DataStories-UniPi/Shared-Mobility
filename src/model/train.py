from datetime import datetime, timezone
from pathlib import Path
from timeit import default_timer
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn import set_config

from components import ModelFactory
from config import paths
from config.constants import FH, GROUP_COLUMN, TARGET_COLUMN
from utils.helper import create_mask, load_data, load_hparams, split_X_y
from utils.models import MLDataConfig

from .evaluation.evaluator import Evaluator
from .evaluation.models import EvaluationConfig, TaskType

set_config(transform_output="pandas")


def train(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    task: TaskType,
    *,
    eval_set=None,
    early_stopping_rounds: int = 10,
    verbose: Optional[int] = 10,
    **params,
):

    factory = ModelFactory(task, num_output=y.ndim, verbose=True, per_output=False)

    model = factory.build_model(
        "xgb",
        estimator_params={
            "objective": (
                "reg:squarederror" if task == TaskType.REGRESSION else "multi:softmax"
            ),
            "eval_metric": ["rmse"] if task == TaskType.REGRESSION else ["mlogloss"],
            "early_stopping_rounds": early_stopping_rounds if eval_set is not None else None,
            "verbosity": 1,
            **params,
        },
    )
    if eval_set is not None:
        start_time = default_timer()
        model.fit(
            X,
            y,
            eval_set=eval_set,
            verbose=verbose,
        )
    else:
        start_time = default_timer()
        model.fit(X, y)

    end_time = default_timer()

    mins, secs = divmod(end_time - start_time, 60)
    print(f"Training completion time: {mins:.0f}min {secs:.0f}sec")

    return model


def predict(
    model,
    X: pd.DataFrame,
    enforce_positive: bool = True,
    round_predictions: bool = True,
    **kwargs,
) -> np.ndarray:
    """Make predictions with post-processing constraints"""

    start = default_timer()
    predictions = model.predict(X, **kwargs)
    end = default_timer() - start
    logger.info(
        f"Prediction complete in {end:.1f}s - ({end / len(predictions):.1e} s/sample)"
    )

    # Post-processing for constraints
    if enforce_positive:
        predictions = np.maximum(predictions, 0)

    if round_predictions:
        predictions = np.round(predictions).astype(int)

    return predictions


def save_predictions(
    preds: pd.DataFrame | pd.Series,
    dataset: str,
    *,
    window: int,
    suffix: str,
) -> None:
    """
    Save predictions to a CSV file.

    Args:
        preds: Predictions to save.
        dataset: Name of the dataset.
        window: Window size.
        suffix: Suffix for the file name.

    Returns:
        None
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filename = f"predictions_{dataset}_h{FH}_w{window}_{suffix}.csv"

    Path.mkdir(paths.BENCHMARKS_DIR / now, parents=True, exist_ok=True)
    preds.to_csv(paths.BENCHMARKS_DIR / now / filename, index=True)


def main(**kwargs):

    task = kwargs.get("task", TaskType.REGRESSION)

    return_X_y: bool = kwargs.get("return_X_y", False)
    dropna: bool = kwargs.get("dropna", False)
    group_col: str = kwargs.get("group_col", GROUP_COLUMN)
    target: str | List[str] = kwargs.get("target", TARGET_COLUMN)

    if isinstance(target, str):
        target = [target]

    config = MLDataConfig(
        fh=kwargs.get("fh", FH),
        window=kwargs.get("window", 1),
        dataset=kwargs["dataset"],
        suffix=kwargs.get("suffix", "v1"),
        target=target,
    )
    X, y = load_data(
        config=config,
        return_X_y=return_X_y,
        dropna=dropna,
    )

    X_train, y_train, X_test, y_test = split_X_y(X, y, test_size=0.1)

    hparams = load_hparams(config.dataset, task, config.fh)

    if task == TaskType.CLASSIFICATION:
        import json

        from utils.helper import create_crowd_levels

        model_id = datetime.now(timezone.utc).date().strftime("%Y%m%d")
        base_dir = paths.MODEL_DIR / model_id / str(config.dataset)

        logger.debug("Transforming target values to demand levels")
        for col in target:

            # Create class labels
            out, bins = create_crowd_levels(y_train, 3, col, group_col)

            # Store class labels
            y_train[col] = pd.concat(out.values()).to_frame()

            # Mask test data with the same bins
            y_test[col] = create_mask(y_test, bins, col)

            logger.debug(f"Class labels built for {col}")

            with Path.open(base_dir / f"{col}_bins.json", "w") as fp:
                json.dump(bins, fp, indent=4)
                logger.debug("Bins saved successfully")

    models = {}  # list storing initialized ML models
    test_metrics = {}  # dict storing test metrics
    predictions = {}  # dict storing predictions

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filename = f"{config.dataset}_h{config.fh}_w{config.window}_{config.suffix}.csv"
    eval_config = EvaluationConfig(
        eval_type=task,
        save_path=paths.BENCHMARKS_DIR / today / filename,
        verbose=False,
        groupby_level=group_col,
    )

    evaluator = Evaluator(eval_config)
    for output in target:

        _train = y_train[output]
        _test = y_test[output]

        logger.info(f"Training {output} model")
        model = train(
            X_train,
            _train,
            **hparams,
            task=task,
            eval_set=[(X_train, _train)],
            verbose=10,
        )

        pred_kwargs = {}

        if hasattr(model, "best_iteration"):
            best_iteration: int = model.best_iteration
            logger.debug(f"Best iteration: {best_iteration}")
            pred_kwargs = {"iteration_range": (0, best_iteration)}

        _pred = predict(model, X_test, **pred_kwargs)
        _pred = pd.DataFrame(_pred, index=_test.index, columns=[output])

        _metrics = evaluator.evaluate(_test, _pred, y_train=_train)

        test_metrics[output] = _metrics
        predictions[output] = _pred
        models[output] = model

    for output, preds in predictions.items():
        save_predictions(
            preds,
            config.dataset,
            window=config.window,
            suffix=f"{config.suffix}_{output}",
        )
        logger.info(
            f"{' '.join(output.split('_').capitalize())} predictions saved successfully"
        )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a Global-GBDP pipeline for a given city"
    )

    parser.add_argument("-d", "--dataset", type=str, nargs="+", help="Select dataset")
    parser.add_argument("-h", "--fh", type=int, nargs="+", help="Select horizon")
    parser.add_argument("-w", "--window", help="Select window size", type=int)
    parser.add_argument("-s", "--suffix", type=str, default="v4", help="Select data suffix")
    parser.add_argument("--return_X_y", action="store_true", help="Split data into X and y")
    parser.add_argument("--dropna", action="store_true", help="Drop NaN values during load")
    parser.add_argument("-g", "--group_col", type=str, help="Select group column")
    parser.add_argument("-T", "--target", type=str, nargs="+", help="Select target column(s)")
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        help="Select estimator type",
        choices=[TaskType.REGRESSION, TaskType.CLASSIFICATION],
    )

    args = parser.parse_args()

    return {k: v for k, v in vars(args).items() if k is not None}


if __name__ == "__main__":

    kwargs = parse_args()
    main(**kwargs)
