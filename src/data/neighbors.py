import json
from pathlib import Path
from typing import Optional

import geopandas as gpd

from config import paths
from utils.logger import configure_logger


def load_data(file_path: Path) -> gpd.GeoDataFrame:
    """Load data from CSV and convert to a GeoDataFrame.

    Args:
        file_path : Path to the metadata CSV file.

    Returns:
        A GeoDataFrame containing coordinates.

    """
    df = gpd.read_file(file_path)

    if df["geometry"].dtype != "geometry":
        df["geometry"] = gpd.GeoSeries.from_wkt(df["geometry"])

    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs(3857)


def compute_adjacency_matrix(
    data: gpd.GeoDataFrame,
    id_col: str,
    r: Optional[float] = None,
) -> dict[str, list[str]]:
    """
    Compute the adjacency matrix of a GeoDataFrame.

    For polygon geometries, direct intersection is used. For point geometries,
    a buffer of radius `r` meters is created to perform intersection queries.

    Args:
        data: GeoDataFrame containing the geographic data.
        id_col: Name of the column containing the district IDs.
        r (optional, default=None): Buffer radius in meters to use for point geometries.
            Ignored for polygon geometries. Defaults to 1000.

    Returns:
        Adjacency matrix represented as a dictionary where each key is a district and
        the value is a list of neighboring districts.

    Raises:
        ValueError: If input parameters are invalid or if required columns don't exist.
        TypeError: If input data is not a GeoDataFrame or if geometry column is missing.

    Example:
        >>> import geopandas as gpd
        >>> # For polygon data
        >>> polygons = gpd.GeoDataFrame({
        ...     'id': ['A', 'B', 'C'],
        ...     'geometry': [polygon1, polygon2, polygon3]
        ... })
        >>> adjacency = compute_adjacency_matrix(polygons, 'id')
        >>> print(adjacency)
        {'A': ['B', 'C'], 'B': ['A'], 'C': ['A']}
        >>>
        >>> # For point data with buffer
        >>> points = gpd.GeoDataFrame({
        ...     'id': ['P1', 'P2', 'P3'],
        ...     'geometry': [point1, point2, point3]
        ... })
        >>> adjacency = compute_adjacency_matrix(points, 'id', r=500)
        >>> print(adjacency)
        {'P1': ['P2'], 'P2': ['P1', 'P3'], 'P3': ['P2']}

    Note:
        - For polygon geometries, adjacency is determined by direct geometric intersection
        - For point geometries, adjacency is determined by buffering points and checking
          for intersections with other geometries
        - The function assumes the GeoDataFrame has a valid spatial index
    """
    try:
        if id_col not in data.columns or "geometry" not in data.columns:
            raise ValueError(f"GeoDataFrame must contain '{id_col} and geometry columns.")

        # Create a copy of the data with only necessary columns
        df = data[[id_col, "geometry"]].copy()

        # Validate that geometries exist
        if df.geometry.isnull().any():
            raise ValueError("Geometry column contains null values")

        # Check if spatial index exists, if not create it
        if not hasattr(df, "sindex") or df.sindex is None:
            df = df.copy()  # This forces a copy to ensure sindex is created

        matrix = {}

        # Get all unique district IDs
        districts = df[id_col].tolist()

        for i, district in enumerate(districts, start=1):
            if i % 100 == 0:
                logger.debug(f"Processing district [{i}/{len(districts)}]")

            try:

                # Get the current district's geometry
                current_geom = df[df[id_col] == district].geometry.iloc[0]

                # Validate geometry
                if current_geom is None or current_geom.is_empty:
                    logger.debug(f"No geometry found for district {district}")
                    matrix[district] = []
                    continue

                # Determine if we need to buffer (for points)
                if current_geom.geom_type == "Point":
                    if r is None:
                        raise ValueError("Buffer radius is required for point geometries.")

                    # Buffer the point geometry
                    buffered_geom = current_geom.buffer(r)

                    # Find all geometries that intersect with the buffered point
                    source_idx = df.sindex.query(buffered_geom, predicate="intersects")
                else:
                    # For polygons, use direct intersection
                    source_idx = df.sindex.query(current_geom, predicate="intersects")

                # Get neighbors (excluding the district itself)
                neighbors = []
                for source in source_idx:
                    source_name = df.loc[source, id_col]
                    # Skip the district itself
                    if source_name != district:
                        neighbors.append(source_name)

                matrix[district] = neighbors

            except Exception as e:
                # Log the error but continue processing other districts
                logger.warning(f"Error processing district '{district}': {str(e)}")
                matrix[district] = []

        return matrix

    except Exception as e:
        raise ValueError(f"Error in compute_adjacency_matrix: {str(e)}")


def save_to_json(adjacency_matrix: dict, file_path: Path) -> bool:
    """Save nearest neighbors DataFrame as a JSON file.

    Parameters
    ----------
    neighbors_df : pd.DataFrame
        DataFrame containing nearest neighbor mappings.
    file_path : Path
        Path where JSON file will be saved.

    Returns
    -------
    bool
        True if successful, False otherwise.

    """

    try:
        with Path.open(file_path, "w") as fp:
            json.dump(adjacency_matrix, fp, indent=4)

            return True
    except TypeError as e:
        logger.exception(f"Error serializing object: {e}")
        return False
    except FileNotFoundError:
        logger.exception("File path is incorrect.")
        return False


def main_flow(
    dataset: str,
    id: str,
    radius: Optional[float] = None,
    extension: str = "json",
) -> None:
    """Compute nearest neighbors for stations and save results.

    This function acts as the main entry point for the Prefect flow. It loads data,
    computes nearest neighbors for each, and saves the results to a JSON file.

    Args:
        dataset : The name of the dataset.
        id : The name of the ID column.

    Returns:
        DataFrame with IDs and their nearest neighbors.

    """

    neighbors_df = load_data(paths.EXTERNAL_DATA_DIR / f"{dataset.lower()}.{extension}").pipe(
        compute_adjacency_matrix, id, radius
    )

    fp = paths.EXTERNAL_DATA_DIR / f"{dataset.lower()}_neighbors.json"
    if save_to_json(neighbors_df, fp):
        logger.success("Process completed successfully.")


def parse_args():
    parser = argparse.ArgumentParser(description="Neighbors computation")
    parser.add_argument("-d", "--dataset", help="Select city", type=str)
    parser.add_argument("--id", help="Select ID column of the data", type=str)
    parser.add_argument("-r", "--radius", help="Select radius", type=float, required=False)
    parser.add_argument("-e", "--extension", help="Select file extension", type=str)

    args = parser.parse_args()
    return {k: v for k, v in vars(args).items() if v is not None}


if __name__ == "__main__":
    import argparse

    filename = __file__.split("/")[-1].split(".")[0]
    logger = configure_logger(filename)

    kwargs = parse_args()
    logger.debug(f"Calling main with arguments: {kwargs}")
    main_flow(**kwargs)
