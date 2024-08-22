import os
import re
import sys
from pathlib import Path

import pandas as pd

# Add the project root to the system path
from config_loader import config


def get_all_terms_and_synonyms_df(df):
    """
    Create a new DataFrame where terms and their corresponding synonyms are placed in separate rows.

    The input DataFrame should contain two columns: 'term' and 'synonyms'. For each row, the 'term' is added as a new row in the resulting DataFrame. The 'synonyms' column (if it contains a valid string) is split by either a semicolon (';') or a comma (','), and each synonym is added as a separate row in the resulting DataFrame. Leading and trailing whitespace is removed from both terms and synonyms.

    The resulting DataFrame has two columns:
    - 'entity': Contains either the term or a synonym.
    - 'type': A label indicating whether the 'entity' is a 'term' or a 'synonym'.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing terms and their synonyms. It must have at least two columns:
        - 'term': The primary term.
        - 'synonyms': A string containing one or more synonyms, separated by commas or semicolons.

    Returns:
    --------
    all_entities_df : pandas.DataFrame
        A DataFrame with two columns:
        - 'entity': Contains either a term or a synonym.
        - 'type': Indicates whether the 'entity' is a 'term' or a 'synonym'.

    Example:
    --------
    Input DataFrame:
        | term    | synonyms              |
        |---------|-----------------------|
        | Apple   | Fruit, Produce; Food  |
        | Orange  | Citrus; Juice         |

    Output DataFrame:
        | entity     | type    |
        |------------|---------|
        | Apple      | term    |
        | Fruit      | synonym |
        | Produce    | synonym |
        | Food       | synonym |
        | Orange     | term    |
        | Citrus     | synonym |
        | Juice      | synonym |

    Notes:
    ------
    - Synonyms are split by semicolons (;) or commas (,).
    - Whitespace around terms and synonyms is removed.
    - Empty or invalid 'synonyms' entries are ignored.
    """

    # List to store the rows of the new DataFrame
    rows = []

    for _, row in df.iterrows():
        term = row["term"].strip()  # Remove leading/trailing whitespace from terms
        synonyms = row["synonyms"]

        # Add the term as a new row
        rows.append({"entity": term, "type": "term"})

        # Ensure 'synonyms' is a valid string, and split by ';' or ','
        if isinstance(synonyms, str) and synonyms.strip():
            # Split the synonyms by semicolon or comma, and strip whitespace from each synonym
            synonyms_list = [
                syn.strip() for syn in re.split(r"[;,]", synonyms) if syn.strip()
            ]
            for synonym in synonyms_list:
                rows.append({"entity": synonym, "type": "synonym"})

    # Create a new DataFrame from the list of rows
    new_df = pd.DataFrame(rows)

    return new_df


# Function to export one column of a DataFrame to a .txt file and verify the newline count
def export_column_to_txt(
    df, column_name, output_dir: str, filename: str, dictionary_name: str = ""
):
    """
    Export a single column of a DataFrame to a .txt file.

    Each row of the specified column will be written as a new line in the output file.
    Ensures that the number of newlines matches the number of non-NaN rows in the column.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the column to export.

    column_name : str
        The name of the column to export.

    filename : str
        The path to the .txt file where the column data will be written.

    Raises:
    -------
    ValueError : If the number of newlines in the output file does not match the expected count.
    """
    # Ensure the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Drop NaN values and convert to a list
    lines = df[column_name].dropna().astype(str).tolist()

    # Write the column to the file, ensuring no trailing newline
    file_path = Path(output_dir) / Path(dictionary_name + "_" + filename + ".txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Read the file back and count the number of lines
    with open(file_path, "r", encoding="utf-8") as f:
        file_lines = f.readlines()

    # Verify the number of lines in the file matches the number of non-NaN rows in the column
    expected_line_count = len(lines)
    actual_line_count = len(file_lines)

    if actual_line_count != expected_line_count:
        raise ValueError(
            f"Expected {expected_line_count} lines in the file but found {actual_line_count}."
        )

    print(
        f"Column '{column_name}' successfully exported to {file_path} with {actual_line_count} lines."
    )
