#!/usr/bin/env python3
"""Real Data Loader for Date-Level Time Series Analysis
This script loads and preprocesses the real time series data with date-level resolution.
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def load_real_time_series(filepath=None, start_date=None, end_date=None):
    """Load real time series data with date-level resolution.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to the standard location.
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format to filter data
    end_date : str, optional
        End date in 'YYYY-MM-DD' format to filter data

    Returns
    -------
    pd.DataFrame
        DataFrame with Date as index, co_count and document_count columns

    """
    if filepath is None:
        filepath = "/home/callebalik/EnviDis/data/raw/time-series/time_series.csv"

    print(f"Loading real time series data from: {filepath}")

    # Load the data
    df = pd.read_csv(filepath)
    print(f"Initial data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Convert date column to datetime
    df["Date"] = pd.to_datetime(df["date"])

    # Rename columns to match our framework
    df = df.rename(
        columns={
            "co_count": "co_count",
            "document_count": "document_count",
        },
    )

    # Filter by date range if specified
    if start_date:
        df = df[df["Date"] >= start_date]
        print(f"Filtered to start from {start_date}: {len(df)} rows")

    if end_date:
        df = df[df["Date"] <= end_date]
        print(f"Filtered to end at {end_date}: {len(df)} rows")

    # Set Date as index
    df = df.set_index("Date").sort_index()

    # Keep only the columns we need for analysis
    analysis_df = df[["co_count", "document_count"]].copy()

    # Create scaled time variables for modeling
    # Convert dates to numeric (days since first observation)
    first_date = analysis_df.index.min()
    analysis_df["Days_since_start"] = (analysis_df.index - first_date).days

    # Create multiple time scales
    analysis_df["Years_since_start"] = analysis_df["Days_since_start"] / 365.25
    analysis_df["Months_since_start"] = analysis_df["Days_since_start"] / 30.44

    # Create scaled versions (mean-centered and standardized)
    analysis_df["Days_scaled"] = (
        analysis_df["Days_since_start"] - analysis_df["Days_since_start"].mean()
    ) / analysis_df["Days_since_start"].std()
    analysis_df["Years_scaled"] = (
        analysis_df["Years_since_start"] - analysis_df["Years_since_start"].mean()
    ) / analysis_df["Years_since_start"].std()
    analysis_df["Months_scaled"] = (
        analysis_df["Months_since_start"] - analysis_df["Months_since_start"].mean()
    ) / analysis_df["Months_since_start"].std()

    # Add year and month columns for aggregation if needed
    analysis_df["Year"] = analysis_df.index.year
    analysis_df["Month"] = analysis_df.index.month
    analysis_df["YearMonth"] = analysis_df.index.to_period("M")

    print(f"Final processed data shape: {analysis_df.shape}")
    print(f"Date range: {analysis_df.index.min()} to {analysis_df.index.max()}")
    print(f"Time span: {analysis_df['Years_since_start'].max():.1f} years")
    print(
        f"co_count range: {analysis_df['co_count'].min()} to {analysis_df['co_count'].max()}",
    )
    print(
        f"document_count range: {analysis_df['document_count'].min()} to {analysis_df['document_count'].max()}",
    )

    return analysis_df


def aggregate_to_yearly(data):
    """Aggregate date-level data to yearly for comparison with previous analyses.

    Parameters
    ----------
    data : pd.DataFrame
        Date-indexed DataFrame

    Returns
    -------
    pd.DataFrame
        Year-indexed aggregated DataFrame

    """
    yearly = (
        data.groupby("Year")
        .agg(
            {
                "co_count": "sum",
                "document_count": "sum",
            },
        )
        .reset_index()
    )

    # Recreate scaled variables for yearly data
    yearly["Year_scaled"] = (yearly["Year"] - yearly["Year"].mean()) / yearly[
        "Year"
    ].std()

    yearly = yearly.set_index("Year")

    print(f"Aggregated to yearly data: {yearly.shape}")
    print(f"Year range: {yearly.index.min()} to {yearly.index.max()}")

    return yearly


def aggregate_to_monthly(data):
    """Aggregate date-level data to monthly.

    Parameters
    ----------
    data : pd.DataFrame
        Date-indexed DataFrame

    Returns
    -------
    pd.DataFrame
        Month-indexed aggregated DataFrame

    """
    monthly = (
        data.groupby("YearMonth")
        .agg(
            {
                "co_count": "sum",
                "document_count": "sum",
            },
        )
        .reset_index()
    )

    # Convert period back to datetime for easier handling
    monthly["Date"] = monthly["YearMonth"].dt.to_timestamp()
    monthly = monthly.set_index("Date").drop("YearMonth", axis=1)

    # Create scaled time variables
    first_date = monthly.index.min()
    monthly["Months_since_start"] = (monthly.index.year - first_date.year) * 12 + (
        monthly.index.month - first_date.month
    )
    monthly["Years_since_start"] = monthly["Months_since_start"] / 12
    monthly["Months_scaled"] = (
        monthly["Months_since_start"] - monthly["Months_since_start"].mean()
    ) / monthly["Months_since_start"].std()
    monthly["Years_scaled"] = (
        monthly["Years_since_start"] - monthly["Years_since_start"].mean()
    ) / monthly["Years_since_start"].std()

    print(f"Aggregated to monthly data: {monthly.shape}")
    print(f"Date range: {monthly.index.min()} to {monthly.index.max()}")

    return monthly


def get_data_summary(data):
    """Generate a comprehensive summary of the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data

    Returns
    -------
    dict
        Summary statistics and information

    """
    summary = {
        "total_observations": len(data),
        "date_range": {
            "start": data.index.min(),
            "end": data.index.max(),
            "span_years": (
                data["Years_since_start"].max()
                if "Years_since_start" in data.columns
                else None
            ),
        },
        "observed_entities": {
            "total": data["co_count"].sum(),
            "mean": data["co_count"].mean(),
            "median": data["co_count"].median(),
            "max": data["co_count"].max(),
            "zero_observations": (data["co_count"] == 0).sum(),
            "non_zero_percentage": ((data["co_count"] > 0).sum() / len(data)) * 100,
        },
        "total_documents": {
            "total": data["document_count"].sum(),
            "mean": data["document_count"].mean(),
            "median": data["document_count"].median(),
            "max": data["document_count"].max(),
        },
        "co_occurrence_rate": {
            "overall": (
                data["co_count"].sum() / data["document_count"].sum()
                if data["document_count"].sum() > 0
                else 0
            ),
            "per_1000_docs": (
                (data["co_count"].sum() / data["document_count"].sum()) * 1000
                if data["document_count"].sum() > 0
                else 0
            ),
        },
    }

    return summary


def print_data_summary(data) -> None:
    """Print a formatted data summary."""
    summary = get_data_summary(data)

    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    print(f"Total observations: {summary['total_observations']:,}")
    print(
        f"Date range: {summary['date_range']['start'].strftime('%Y-%m-%d')} to {summary['date_range']['end'].strftime('%Y-%m-%d')}",
    )
    if summary["date_range"]["span_years"]:
        print(f"Time span: {summary['date_range']['span_years']:.1f} years")

    print("\nIdentified DIS-PNM co-occurrences:")
    print(f"  Total: {summary['observed_entities']['total']:,}")
    print(f"  Mean per observation: {summary['observed_entities']['mean']:.2f}")
    print(f"  Median: {summary['observed_entities']['median']:.0f}")
    print(f"  Maximum: {summary['observed_entities']['max']:,}")
    print(
        f"  Zero observations: {summary['observed_entities']['zero_observations']:,} ({(summary['observed_entities']['zero_observations']/summary['total_observations']*100):.1f}%)",
    )
    print(
        f"  Non-zero observations: {summary['observed_entities']['non_zero_percentage']:.1f}%",
    )

    print("\nTotal Documents:")
    print(f"  Total: {summary['total_documents']['total']:,}")
    print(f"  Mean per observation: {summary['total_documents']['mean']:.2f}")
    print(f"  Median: {summary['total_documents']['median']:.0f}")
    print(f"  Maximum: {summary['total_documents']['max']:,}")

    print("\nCo-occurrence Rate:")
    print(f"  Overall: {summary['co_occurrence_rate']['overall']:.6f}")
    print(
        f"  Per 1,000 documents: {summary['co_occurrence_rate']['per_1000_docs']:.3f}",
    )


if __name__ == "__main__":
    # Load and explore the real data
    print("Loading real time series data...")

    # Load full dataset
    data = load_real_time_series()
    print_data_summary(data)

    # Show first few rows
    print("\nFirst 10 observations:")
    print(data[["co_count", "document_count", "Years_since_start"]].head(10))

    # Show recent observations
    print("\nLast 10 observations:")
    print(data[["co_count", "document_count", "Years_since_start"]].tail(10))

    # Test aggregations
    print("\n" + "=" * 60)
    print("TESTING AGGREGATIONS")
    print("=" * 60)

    yearly = aggregate_to_yearly(data)
    print_data_summary(yearly)

    monthly = aggregate_to_monthly(data)
    print_data_summary(monthly)
