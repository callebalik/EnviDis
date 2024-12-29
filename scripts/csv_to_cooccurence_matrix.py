import pandas as pd
from config_loader import cfg
from typing import Optional, List


def read_csv(input_filepath: str) -> pd.DataFrame:
    # Read the CSV file with headers
    return pd.read_csv(input_filepath, header=0)


def create_cooccurrence_matrix(
    df: pd.DataFrame,
    limit_rows: Optional[int] = None,
    limit_columns: Optional[int] = None,
    lowercase: bool = False,
) -> pd.DataFrame:

    if lowercase:
        df["entity_1"] = df["entity_1"].str.lower()
        df["entity_2"] = df["entity_2"].str.lower()

    # Aggregate the frequencies for the same entity pairs
    df_aggregated = df.groupby(["entity_2", "entity_1"])[["fq"]].sum().reset_index()

    # Create a pivot table to form the co-occurrence matrix
    cooccurrence_matrix = df_aggregated.pivot(
        index="entity_2", columns="entity_1", values="fq"
    ).fillna(0)

    # Ensure the position (0,0) is not populated by the header
    cooccurrence_matrix.index.name = None
    cooccurrence_matrix.columns.name = None

    # Convert the frequencies to integers
    cooccurrence_matrix = cooccurrence_matrix.astype(int)

    # Limit the matrix to only i x j by dropping other rows/columns
    if limit_rows:
        cooccurrence_matrix = cooccurrence_matrix.iloc[:limit_rows, :]
    if limit_columns:
        cooccurrence_matrix = cooccurrence_matrix.iloc[:, :limit_columns]

    # Return the co-occurrence matrix
    return cooccurrence_matrix


def save_cooccurrence_matrix(matrix: pd.DataFrame, output_filepath: str) -> None:
    matrix.to_csv(output_filepath)


def filter_frequency(df: pd.DataFrame, percentile: float = 0.0) -> pd.DataFrame:
    # Calculate the frequency threshold based on the provided percentile

    threshold = df["fq"].quantile(percentile)
    print(
        f"Filtering frequencies below the {percentile * 100}th percentile = cooccurence strength of ({threshold})"
    )

    # Filter the frequencies
    df = df[df["fq"] >= threshold]

    # Return the filtered dataframe
    return df


def filter_rows_columns(
    matrix: pd.DataFrame,
    row_filters: Optional[List[str]] = None,
    column_filters: Optional[List[str]] = None,
) -> pd.DataFrame:
    if row_filters:
        for row_filter in row_filters:
            matrix = matrix[~matrix.index.str.contains(row_filter)]
    if column_filters:
        for column_filter in column_filters:
            matrix = matrix.loc[:, ~matrix.columns.str.contains(column_filter)]
    if row_filters and column_filters:
        for row_filter in row_filters:
            for column_filter in column_filters:
                matrix.loc[
                    matrix.index.str.contains(row_filter),
                    matrix.columns.str.contains(column_filter),
                ] = 0
    return matrix


def drop_unconnected_entities(matrix: pd.DataFrame) -> pd.DataFrame:
    # Drop rows and columns with no connections
    matrix = matrix[(matrix.T != 0).any()]
    matrix = matrix.loc[:, (matrix != 0).any(axis=0)]
    return matrix


if __name__ == "__main__":
    input_filepath = cfg.DATA_DIR + "/processed/entities/cooc50.csv"
    output_filepath = cfg.DATA_DIR + "/processed/entities/cooc50_matrix.csv"

    limit_rows = None  # Example row limit
    limit_columns = None  # Example column limit
    percentile_threshold = 0.6  # Example percentile threshold

    df = read_csv(input_filepath)
    df = filter_frequency(df, percentile_threshold)

    phenomena_ambigous = ["current", "wave", "precipitation", "stream"]
    phenomena_ecluded = phenomena_ambigous

    diseases_misslabelled = [
        "fire",
        "fires",
        "forrest fire",
        "earthquake",
        "drought",
        "flood",
        "tsunami",
        "eutrophication",
        "water eutrophication",
        "landslide",
        "soil erosion",
        "air pollution",
    ]

    diseases_amibgous = ["ad"]

    disease_ecluded = diseases_misslabelled + diseases_amibgous

    cooccurrence_matrix = create_cooccurrence_matrix(
        df, limit_rows, limit_columns, lowercase=True
    )
    cooccurrence_matrix = filter_rows_columns(
        cooccurrence_matrix,
        row_filters=phenomena_ecluded,
        column_filters=disease_ecluded,
    )
    cooccurrence_matrix = drop_unconnected_entities(cooccurrence_matrix)
    save_cooccurrence_matrix(cooccurrence_matrix, output_filepath)
