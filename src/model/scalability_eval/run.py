from pprint import pprint
from venv import logger

from components import ModelFactory
from config.constants import CITY, FH, TARGET_COLUMN
from model.scalability_eval.evaluate import GroupedTimeSeriesEvaluator
from utils.helper import load_data, load_hparams
from utils.models import MLDataConfig, TaskType


def main(**kwargs):

    task = kwargs.get("task", TaskType.REGRESSION)
    dataset = kwargs.get("dataset", CITY)

    config = MLDataConfig(
        fh=kwargs.get("fh", FH),
        window=kwargs.get("window", 1),
        dataset=dataset,
        suffix=kwargs.get("suffix", "v1"),
        extension=kwargs.get("extension", None),
    )

    df = load_data(config=config, return_X_y=False).droplevel("split").sort_index()

    X, y = df.drop(TARGET_COLUMN, axis=1), df[TARGET_COLUMN]

    task = TaskType.REGRESSION
    hparams = load_hparams(dataset=dataset.lower(), task=task, fh=FH)

    # Initialize and run evaluator
    factory = ModelFactory(task, num_output=1, verbose=True, per_output=False)

    model = factory.build_model(
        "xgb",
        estimator_params={
            "objective": (
                "reg:squarederror" if task == TaskType.REGRESSION else "multi:softmax"
            ),
            "eval_metric": ["rmse"] if task == TaskType.REGRESSION else ["mlogloss"],
            "verbosity": 1,
            **hparams,
        },
    )

    evaluator = GroupedTimeSeriesEvaluator(
        model=model,
        group_col=kwargs["group_col"],
        n_increments=kwargs.get("increments", 5),
        n_cv_folds=kwargs.get("cv", 3),
    )

    for col in TARGET_COLUMN:
        results = evaluator.evaluate(X, y[col])

        pprint(results, indent=4)


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-c", "--cv", type=int, help="Number of cross-validation folds")
    parser.add_argument("-d", "--dataset", type=str, help="Dataset name")
    parser.add_argument("-f", "--fh", type=int, nargs="+", help="Forecasting horizon")
    parser.add_argument("-g", "--group_col", type=str, help="Group column name")
    parser.add_argument("-i", "--increments", type=int, help="Number of increments")
    parser.add_argument("-s", "--suffix", type=str, help="Dataset suffix")
    parser.add_argument("-t", "--task", type=str, help="Task type")
    parser.add_argument("-w", "--window", type=int, help="Window size")

    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}


if __name__ == "__main__":

    kwargs = parse_args()
    logger.debug(f"Calling main with arguments: {kwargs}")
    main(**kwargs)
