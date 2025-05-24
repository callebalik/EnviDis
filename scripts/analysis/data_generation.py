"""
Data Generation Script for Time Series Analysis
This script generates simulated time series data for entity counts and document counts.
"""

import pandas as pd
import numpy as np


def generate_sample_data(start_year=1970, end_year=2024, seed=42):
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

    Returns:
    --------
    pd.DataFrame
        DataFrame with Year as index and ObservedEntities, TotalDocuments as columns
    """
    years = np.arange(start_year, end_year + 1)
    np.random.seed(seed)

    # Simulate total document counts (N_t) - let's say it generally increases
    total_documents = (
        np.random.randint(10000, 50000, size=len(years)) +
        np.linspace(0, 100000, len(years)) +
        np.random.normal(0, 5000, size=len(years))
    )
    total_documents = np.maximum(10000, total_documents).astype(int)

    # Simulate true underlying entity counts (unobserved) with a non-monotonic trend
    # For demonstration, let's imagine a cubic trend for the true entities
    true_entity_trend = (
        500 +
        2 * (years - start_year) +
        0.1 * (years - start_year)**2 -
        0.005 * (years - start_year)**3
    )
    true_entity_trend = np.maximum(10, true_entity_trend)

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
    # Generate sample data
    data = generate_sample_data()

    print("Sample Data Head:")
    print(data.head())
    print("\nData Info:")
    data.info()
    print(f"\nData shape: {data.shape}")
    print(f"Year range: {data.index.min()} - {data.index.max()}")

    # Save to file
    save_sample_data(data, '/home/callebalik/EnviDis/data/processed/sample_time_series_data.csv')
