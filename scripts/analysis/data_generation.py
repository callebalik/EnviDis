"""
Data Generation Script for Time Series Analysis
This script generates simulated time series data for entity counts and document counts.
"""

import pandas as pd
import numpy as np

# Module-level constants for data generation types
DATA_GENERATION_TYPE = "non_constant_upward"  # Options: "constant_upward", "non_constant_upward"

GENERATION_TYPES = {
    "constant_upward": "Constant upward trend",
    "non_constant_upward": "Non-constant upward trend (with variations)"
}


def generate_constant_upward_trend(years, start_year):
    """Generate entities with a constant linear upward trend."""
    trend = 100 + 5 * (years - start_year)  # Linear growth
    return np.maximum(10, trend)


def generate_non_constant_upward_trend(years, start_year):
    """Generate entities with a non-constant upward trend (cubic with variations)."""
    trend = (
        500 +
        2 * (years - start_year) +
        0.1 * (years - start_year)**2 -
        0.005 * (years - start_year)**3
    )
    return np.maximum(10, trend)


def generate_sample_data(start_year=1970, end_year=2024, seed=42, generation_type=None):
    """
    Generate sample time series data for analysis.

    Parameters:
    -----------
    start_year : int
        Starting year for the time series
    end_year : int
        Ending year for the time series (inclusive)
    seed : int
        Random seed for reproducibility
    generation_type : str, optional
        Type of data generation. If None, uses DATA_GENERATION_TYPE constant

    Returns:
    --------
    pd.DataFrame
        DataFrame with Year as index and ObservedEntities, TotalDocuments as columns
    """
    if generation_type is None:
        generation_type = DATA_GENERATION_TYPE

    if generation_type not in GENERATION_TYPES:
        raise ValueError(f"Invalid generation_type. Must be one of: {list(GENERATION_TYPES.keys())}")

    years = np.arange(start_year, end_year + 1)
    np.random.seed(seed)

    # Simulate total document counts (N_t) - let's say it generally increases
    total_documents = (
        np.random.randint(10000, 50000, size=len(years)) +
        np.linspace(0, 100000, len(years)) +
        np.random.normal(0, 5000, size=len(years))
    )
    total_documents = np.maximum(10000, total_documents).astype(int)

    # Generate true underlying entity counts based on selected type
    if generation_type == "constant_upward":
        true_entity_trend = generate_constant_upward_trend(years, start_year)
    elif generation_type == "non_constant_upward":
        true_entity_trend = generate_non_constant_upward_trend(years, start_year)

    # Simulate observed entity counts (Y_t) as a sample of the true entities,
    # influenced by total documents and NER sensitivity (assumed constant here)
    # Let's say NER sensitivity is around 50% of true entities, plus some noise
    observed_entities = (
        (true_entity_trend * 0.5) +
        (total_documents / 10000) * 50 +
        np.random.normal(0, 50, size=len(years))
    )
    observed_entities = np.maximum(0, observed_entities).astype(int)

    data = pd.DataFrame({
        'Year': years,
        'ObservedEntities': observed_entities,
        'TotalDocuments': total_documents
    })

    # Set 'Year' as index for time series analysis
    data = data.set_index('Year')

    # Add preprocessing features
    data['const'] = 1
    data['Year_centered'] = data.index.values - data.index.values.mean()
    data['Year_scaled'] = data['Year_centered'] / data['Year_centered'].std()

    return data


def save_sample_data(data, filepath):
    """Save the generated data to a CSV file."""
    data.to_csv(filepath)
    print(f"Data saved to {filepath}")


if __name__ == "__main__":
    # Generate sample data using the module constant
    print(f"Generating data with type: {DATA_GENERATION_TYPE} ({GENERATION_TYPES[DATA_GENERATION_TYPE]})")
    data = generate_sample_data()

    print("Sample Data Head:")
    print(data.head())
    print("\nData Info:")
    data.info()
    print(f"\nData shape: {data.shape}")
    print(f"Year range: {data.index.min()} - {data.index.max()}")

    # Save to file
    filename = f'/home/callebalik/EnviDis/data/processed/sample_time_series_data_{DATA_GENERATION_TYPE}.csv'
    save_sample_data(data, filename)

    # Demonstrate different generation types
    print("\nAvailable generation types:")
    for key, description in GENERATION_TYPES.items():
        print(f"  {key}: {description}")
