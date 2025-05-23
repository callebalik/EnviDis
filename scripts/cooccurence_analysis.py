import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations

# Sample data: list of sentences
sentences = [
    "this is a sample sentence",
    "this sentence is another example",
    "example sentence with sample words",
    "more words in this example sentence",
]

# Read content of a text file
with open("tests/easyNer_example_abstract.txt", "r") as file:
    sentences = file.readlines()

# Tokenize sentences into words
tokenized_sentences = [sentence.split() for sentence in sentences]

# Create a list of all unique words
all_words = sorted(set(word for sentence in tokenized_sentences for word in sentence))

# Create a dictionary to map words to indices
word_to_index = {word: i for i, word in enumerate(all_words)}

# Initialize the co-occurrence matrix
co_occurrence_matrix = np.zeros((len(all_words), len(all_words)), dtype=int)

# Populate the co-occurrence matrix
for sentence in tokenized_sentences:
    for word1, word2 in combinations(sentence, 2):
        index1, index2 = word_to_index[word1], word_to_index[word2]
        co_occurrence_matrix[index1, index2] += 1
        co_occurrence_matrix[index2, index1] += 1

# Convert the matrix to a DataFrame for better readability
co_occurrence_df = pd.DataFrame(
    co_occurrence_matrix, index=all_words, columns=all_words
)


def drop_below_threshold(df, threshold):
    """
    Drop all co-occurrence frequencies below a given threshold.

    Parameters:
    df (pd.DataFrame): The co-occurrence matrix as a DataFrame.
    threshold (int): The threshold below which frequencies will be set to 0.

    Returns:
    pd.DataFrame: The modified DataFrame with frequencies below the threshold set to 0.
    """
    df[df < threshold] = 0
    return df


def extract_top_co_occurrences(df, word, top_n=5):
    """
    Extract the top co-occurrences of a word from a co-occurrence DataFrame.

    Parameters:
    df (pd.DataFrame): The co-occurrence matrix as a DataFrame.
    word (str): The word for which to extract co-occurrences.
    top_n (int): The number of top co-occurrences to extract.

    Returns:
    pd.Series: The Series of top co-occurrences.
    """
    return df[word].sort_values(ascending=False).head(top_n)


def keep_top_x_cooccurrences(df, top_x):
    """
    Keep only the top X co-occurrences in the DataFrame.

    Parameters:
    df (pd.DataFrame): The co-occurrence matrix as a DataFrame.
    top_x (int): The number of top co-occurrences to keep.

    Returns:
    pd.DataFrame: The modified DataFrame with only the top X co-occurrences.
    """
    # Flatten the DataFrame and sort by values
    flattened = df.stack().sort_values(ascending=False)

    # Keep only the top X co-occurrences
    top_cooccurrences = flattened.head(top_x)

    # Create a new DataFrame with the same shape, filled with zeros
    new_df = pd.DataFrame(np.zeros(df.shape), index=df.index, columns=df.columns)

    # Populate the new DataFrame with the top co-occurrences
    for (word1, word2), value in top_cooccurrences.items():
        new_df.at[word1, word2] = value
        new_df.at[word2, word1] = value  # Ensure symmetry

    return new_df


# Apply the function to the co-occurrence DataFrame


def drop_self_references(df):
    """
    Drop self-referencing co-occurrences from the DataFrame.

    Parameters:
    df (pd.DataFrame): The co-occurrence matrix as a DataFrame.

    Returns:
    pd.DataFrame: The modified DataFrame with self-references removed.
    """
    df = df.copy()
    np.fill_diagonal(df.values, 0)
    return df


co_occurrence_df = drop_below_threshold(co_occurrence_df, threshold=50)

# Choose a word that exists in the sentences
# top_co_occurrences = extract_top_co_occurrences(co_occurrence_df, word='example', top_n=1)
# print(top_co_occurrences)

# co_occurrence_df = keep_top_x_cooccurrences(co_occurrence_df, top_x=1)
co_occurrence_df = drop_self_references(co_occurrence_df)
print(co_occurrence_df)
co_occurrence_df.to_csv("tests/co_occurrence_matrix.csv")
