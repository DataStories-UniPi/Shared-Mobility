import re
from typing import Dict, List, Tuple, TypeAlias

import numpy as np
import pandas as pd

MetricStats: TypeAlias = Dict[str, float]  # e.g {'mean': 0.5}
BenchmarkMetrics: TypeAlias = Dict[str, MetricStats]  # Dict for a given metric
BenchmarkIndex: TypeAlias = Tuple[str, int]  # District-Horizon key e.g ('Amsterdam', 5)
Benchmarks: TypeAlias = Dict[BenchmarkIndex, BenchmarkMetrics]
ImportanceID: TypeAlias = Tuple[int, str, float]


def group_feature_importances(
    fh: List[int], feature_importances: Dict[int, List[ImportanceID]]
) -> pd.DataFrame:
    return (
        pd.concat(
            [
                pd.DataFrame(
                    feature_importances[horizon],
                    columns=["horizon", "feature", "importance"],
                )
                for horizon in fh
            ],
            ignore_index=True,
        )
        .groupby(["horizon", "feature"], as_index=False)  # Avoid multi-index
        .agg(importance_mean=("importance", "mean"), importance_std=("importance", "std"))
        .sort_values(by="importance_mean", ascending=False)
        .reset_index(drop=True)  # Clean index for readability
    )


def replace_substrings(names: List[str], neighbors: List[str], target: str) -> List[str]:
    """
    Replace substrings in a list of names based on target and neighbors.

    Parameters
    ----------
    names : List[str]
        List of strings to process.
    neighbors : List[str]
        List of substrings considered as neighbors.
    target : str
        Substring to be replaced with "target".

    Returns
    -------
    List[str]
        The modified list with replaced substrings.
    """

    neighbors_set = sorted(neighbors, key=len, reverse=True)
    neighbors_set.remove(target)

    def replace_name(name: str) -> str:

        for neighbor in neighbors_set:
            if neighbor in name:
                return name.replace(neighbor, "neighbor")
        return name

    return [replace_name(name) for name in names]


def parse_name(text: str, replacements: Dict[str, str]) -> str:
    """
    Replace multiple patterns in a text based on a mapping of patterns to replacements.

    Parameters
    ----------
    text : str
        The input text to perform replacements on.
    replacements : dict
        A dictionary where keys are patterns (strings) to search for and
        values are their corresponding replacements.

    Returns
    -------
    str
        The modified text with all replacements applied.
    """

    # Create a compiled regex pattern that matches any of the keys
    pattern = re.compile("|".join(map(re.escape, replacements.keys())))

    # Use the pattern to replace matches with the corresponding replacement
    return pattern.sub(lambda match: replacements[match.group(0)], text)


def create_crowd_levels(
    data: pd.Series,
    n_bins: int,
    target: str,
) -> Tuple[pd.Series, List[int]]:

    out = pd.qcut(data.rank(method="first"), q=n_bins, labels=list(range(3)))

    crowd_bins = (
        pd.DataFrame([out, data], index=["label", target])
        .T.groupby("label")
        .agg(["min", "max"])
        .reset_index(drop=True)
        .values.flatten()
        .tolist()
    )
    crowd_bins[0] = 0
    crowd_bins[-1] = np.inf
    return out.astype(np.uint8), crowd_bins


def create_mask(y_test: pd.Series, pairs: List[int]):
    y_copy = np.array(y_test)

    # Create arrays for lower and upper bounds
    lower_bounds, upper_bounds = [], []
    for low, high in zip(pairs[::2], pairs[1::2]):
        lower_bounds.append(low)
        upper_bounds.append(high)

    # Broadcast y against the intervals and compute the mask
    conditions = (y_copy[:, None] > lower_bounds) & (y_copy[:, None] <= upper_bounds)

    # Find the index of the matching interval for each y value
    mask = np.argmax(conditions, axis=1)

    return pd.Series(mask)
