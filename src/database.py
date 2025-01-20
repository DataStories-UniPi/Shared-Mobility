from typing import List, Tuple

import geopandas as gpd
import pandas as pd
from loguru import logger

import config as config


def _find_neighbors(city: str = config.CITY, target: str = config.TARGET) -> List[str]:

    geojson_path = config.EXTERNAL_DATA_DIR / f"{city.lower()}_.geojson"
    gdf = gpd.read_file(geojson_path)
    source_idx, _ = gdf.loc[gdf["name"] == target].sindex.query(
        gdf.geometry,
        predicate="intersects",
    )

    neighbors_dict = {}
    for source in source_idx:
        neighbors = gdf.sindex.query(gdf.geometry[source], predicate="intersects")
        source_name = gdf.loc[source, "name"]
        neighbors_dict[source_name] = len(neighbors)

    return list(neighbors_dict.keys())


def extract_neighbors(
    df: pd.DataFrame, neighbors: List[str], mandatory_columns: List[str] = ["timestamp"]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract specified columns from the DataFrame while handling missing columns gracefully.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data.
    neighbors : List[str]
        List of neighbor column names to extract.
    mandatory_columns : List[str], optional
        List of mandatory columns to include in the output. Default is ["timestamp"].

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        A tuple containing:
        - The filtered DataFrame with the available columns.
        - The updated list of neighbors (excluding missing columns).

    """

    # Determine available columns
    available_columns = set(df.columns)
    required_columns = set(mandatory_columns + neighbors)
    missing_columns = required_columns - available_columns

    # Log missing columns
    if missing_columns:
        logger.debug(f"Missing columns: {missing_columns}")

    # Filter columns to include only those available in the DataFrame
    valid_columns = list(required_columns & available_columns)
    filtered_neighbors = [col for col in neighbors if col in available_columns]

    sorted_columns = mandatory_columns + sorted(
        col for col in valid_columns if col not in mandatory_columns
    )

    df_min = df[sorted_columns]
    return df_min, filtered_neighbors
