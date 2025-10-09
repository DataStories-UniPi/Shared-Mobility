"""
Benchmarking module for model evaluation.

This module provides components for evaluating machine learning models,
including feature importance analysis, performance benchmarking, and comprehensive
model benchmarking across different scenarios.
"""

from collections import defaultdict
from typing import List

import pandas as pd
from loguru import logger

from components import ModelFactory
from config import paths
from config.constants import GROUP_COLUMN, TARGET_COLUMN
from model.evaluation import ModelBenchmarker
from model.evaluation.models import ImportanceType, OutputFormat
from utils.helper import load_data
from utils.models import TaskType


def save_results(results, task_type: TaskType, identifier: str):
    logger.info("Saving feature importance results")
    group_data = defaultdict(list)

    for (id, fh), data in results["feature_importance"].items():
        data = data.assign(horizon=fh)
        group_data[id].append(data)

    for group in group_data.keys():
        (
            pd.concat(group_data[group])
            .groupby("feature", as_index=False)
            .mean()
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
            .to_csv(
                paths.BENCHMARKS_DIR / f"feature_importance-{task_type}-{group}.csv.gzip",
                compression="gzip",
                index=False,
            )
        )

    logger.info("Saving performance metrics")
    results["performance_metrics"].to_csv(
        paths.BENCHMARKS_DIR / f"performance_metrics-{task_type}-{identifier}.csv.gzip",
        compression="gzip",
    )

    logger.info("Saving timing summary")
    results["timing_summary"].to_csv(
        paths.BENCHMARKS_DIR / f"timing_summary-{task_type}-{identifier}.csv.gzip",
        compression="gzip",
    )


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run benchmark")

    parser.add_argument("-r", "--runs", help="Select number of runs", type=int)
    parser.add_argument("-w", "--warmup_runs", help="Select number of warmup runs", type=int)
    parser.add_argument("-t", "--task", help="Select task", type=str)
    parser.add_argument("-f", "--fh", help="Select forecasting horizon", nargs="+", type=int)
    parser.add_argument("-W", "--windows", help="Select window size", nargs="+", type=int)
    parser.add_argument("-p", "--prefix", help="Select prefix", type=str)
    parser.add_argument("-s", "--suffix", help="Select suffix", type=str)
    parser.add_argument("-d", "--dataset", help="Select dataset", nargs="+", type=str)
    parser.add_argument("-g", "--group_col", help="Select group column", type=str)
    parser.add_argument("-T" "--target", help="Select target column", nargs="+", type=str)
    parser.add_argument("--per_output", help="Select per output model", action="store_true")

    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}


def main(store_results: bool = True, **kwargs):
    task: TaskType = kwargs.pop("task", TaskType.REGRESSION)
    runs: int = kwargs.pop("runs", 10)
    warmup_runs: int = kwargs.pop("warmup_runs", 3)
    group_column = kwargs.pop("group_col", GROUP_COLUMN)
    ids: List[str] = kwargs.pop("dataset", ["rotterdam", "amsterdam", "hague"])
    fh: List[int] = kwargs.pop("horizons", [5, 15, 30, 60])
    target = kwargs.pop("target", TARGET_COLUMN)

    # Initialize benchmarker
    benchmarker = ModelBenchmarker(
        model_factory=ModelFactory,
        target_col=target,
        group_col=group_column,
        task_type=task,
        n_runs=runs,
        warmup_runs=warmup_runs,
        data_loader=load_data,
        per_output_model=kwargs.pop("per_output_model", False),
    )

    # Setup optional analysis components
    benchmarker.setup_feature_analysis(
        importance_types=[ImportanceType.GAIN],
        output_format=OutputFormat.NORMALIZED,
        top_k=15,
    )

    benchmarker.setup_performance_analysis(
        quantiles=[0.25, 0.5, 0.75],
        group_col=group_column,
    )

    # Run comprehensive benchmark
    results = benchmarker.benchmark_combinations(
        identifiers=ids,
        horizons=fh,
        include_feature_importance=True,
        include_performance_metrics=True,
        **kwargs,
    )

    print(results["timing_summary"])
    print("================================================================")
    print(results["performance_metrics"])

    if store_results:
        logger.info("Saving results")
        save_results(results, task, "citi")


if __name__ == "__main__":
    from pprint import pprint

    kwargs = parse_args()
    logger.debug(f"Calling main with arguments: {kwargs}")
    pprint(kwargs)

    main(**kwargs, store_results=True)
