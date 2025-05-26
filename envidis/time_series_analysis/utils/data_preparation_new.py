#!/usr/bin/env python3
"""Data preparation utilities for time series analysis."""

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class DataPreparator:
    """Class for preparing data for time series analysis."""

    def __init__(
        self,
        data_path=None,
        date_col="pubdate_resolved",
        response_col="co_count",
        document_col="document_count",
    ):
        """Initialize data preparator.

        Parameters
        ----------
        data_path : str, optional
            Path to CSV file
        date_col : str
            Name of date column
        response_col : str
            Name of response variable column
        document_col : str
            Name of document count column

        """
        self.data_path = data_path
        self.date_col = date_col
        self.response_col = response_col
        self.document_col = document_col
        self.data = None

    def load_real_data(self, data_path=None):
        """Load real time series data from CSV."""
        path = data_path or self.data_path
        if path is None:
            msg = "No data path provided"
            raise ValueError(msg)

        print(f"Loading data from: {path}")
        df = pd.read_csv(path)

        # Convert date column
        df["Date"] = pd.to_datetime(df[self.date_col])

        # Standardize column names
        df["ObservedEntities"] = df[self.response_col]
        df["TotalDocuments"] = df[self.document_col]

        # Extract year and create time variables
        df["Year"] = df["Date"].dt.year
        df = df.set_index("Year")

        # Create time variables
        df = self._create_time_variables(df)

        self.data = df
        print(
            f"Loaded {len(df)} observations from {df.index.min()} to {df.index.max()}",
        )
        return df

    def load_synthetic_data(self, start_year=1990, end_year=2024, seed=42):
        """Load synthetic time series data."""
        from ..demo.demo_data_generation import generate_sample_data

        print(f"Generating synthetic data from {start_year} to {end_year}")
        self.data = generate_sample_data(start_year, end_year, seed)
        return self.data

    def _create_time_variables(self, df):
        """Create time-based variables for modeling."""
        df_copy = df.copy()

        # Create centered and scaled year variables
        years = df_copy.index.values
        df_copy["Year_centered"] = years - years.mean()
        df_copy["Year_scaled"] = (
            df_copy["Year_centered"] / df_copy["Year_centered"].std()
        )

        # Create polynomial terms
        df_copy["Year_squared"] = df_copy["Year_scaled"] ** 2
        df_copy["Year_cubed"] = df_copy["Year_scaled"] ** 3

        # Create rate variable
        df_copy["Rate"] = df_copy["ObservedEntities"] / df_copy["TotalDocuments"]

        return df_copy

    def get_modeling_subset(self, sample_size=None, filter_percentile=0.95):
        """Get a subset of data suitable for modeling.

        Parameters
        ----------
        sample_size : int, optional
            Number of observations to sample
        filter_percentile : float
            Filter out extreme values above this percentile

        Returns
        -------
        pd.DataFrame
            Filtered subset of data

        """
        if self.data is None:
            msg = "No data loaded. Call load_real_data() or load_synthetic_data() first"
            raise ValueError(
                msg,
            )

        df = self.data.copy()

        # Filter extreme values
        if filter_percentile < 1.0:
            threshold = df["ObservedEntities"].quantile(filter_percentile)
            df = df[df["ObservedEntities"] <= threshold]
            print(f"Filtered out {len(self.data) - len(df)} extreme observations")

        # Sample if requested
        if sample_size is not None and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).sort_index()
            print(f"Sampled {sample_size} observations")

        return df

    def get_temporal_aggregations(self):
        """Get temporal aggregations of the data."""
        if self.data is None:
            raise ValueError("No data loaded")

        aggregations = {}

        # Yearly aggregations (if data has sub-yearly frequency)
        if "Date" in self.data.columns:
            yearly = (
                self.data.groupby(self.data["Date"].dt.year)
                .agg(
                    {
                        "ObservedEntities": ["sum", "mean", "count"],
                        "TotalDocuments": ["sum", "mean"],
                        "Rate": "mean",
                    },
                )
                .round(3)
            )
            aggregations["yearly"] = yearly

        # Decadal aggregations
        decades = self.data.index // 10 * 10
        decadal = (
            self.data.groupby(decades)
            .agg(
                {
                    "ObservedEntities": ["sum", "mean"],
                    "TotalDocuments": ["sum", "mean"],
                    "Rate": "mean",
                },
            )
            .round(3)
        )
        aggregations["decadal"] = decadal

        return aggregations

    def get_data_summary(self):
        """Get comprehensive summary of the loaded data."""
        if self.data is None:
            raise ValueError("No data loaded")

        summary = {
            "basic_info": {
                "n_observations": len(self.data),
                "date_range": (self.data.index.min(), self.data.index.max()),
                "years_span": self.data.index.max() - self.data.index.min(),
            },
            "observed_entities": {
                "mean": self.data["ObservedEntities"].mean(),
                "median": self.data["ObservedEntities"].median(),
                "std": self.data["ObservedEntities"].std(),
                "min": self.data["ObservedEntities"].min(),
                "max": self.data["ObservedEntities"].max(),
                "zero_count": (self.data["ObservedEntities"] == 0).sum(),
                "zero_proportion": (self.data["ObservedEntities"] == 0).mean(),
            },
            "total_documents": {
                "mean": self.data["TotalDocuments"].mean(),
                "median": self.data["TotalDocuments"].median(),
                "std": self.data["TotalDocuments"].std(),
                "min": self.data["TotalDocuments"].min(),
                "max": self.data["TotalDocuments"].max(),
            },
            "rate_statistics": {
                "mean_rate": self.data["Rate"].mean(),
                "median_rate": self.data["Rate"].median(),
                "std_rate": self.data["Rate"].std(),
            },
        }

        # Add missing data info
        summary["missing_data"] = {
            "total_missing": self.data.isnull().sum().sum(),
            "missing_by_column": self.data.isnull().sum().to_dict(),
        }

        return summary

    def validate_data(self) -> bool:
        """Validate the loaded data for modeling."""
        if self.data is None:
            raise ValueError("No data loaded")

        issues = []

        # Check for required columns
        required_cols = ["ObservedEntities", "TotalDocuments"]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")

        # Check for negative values
        if (self.data["ObservedEntities"] < 0).any():
            issues.append("Negative values found in ObservedEntities")

        if (self.data["TotalDocuments"] <= 0).any():
            issues.append("Zero or negative values found in TotalDocuments")

        # Check for missing values
        if self.data["ObservedEntities"].isnull().any():
            issues.append("Missing values found in ObservedEntities")

        if self.data["TotalDocuments"].isnull().any():
            issues.append("Missing values found in TotalDocuments")

        # Check for extremely skewed data
        if (
            self.data["ObservedEntities"].max() / self.data["ObservedEntities"].mean()
            > 100
        ):
            issues.append("Extremely skewed ObservedEntities (consider transformation)")

        if len(issues) == 0:
            print("✓ Data validation passed")
            return True
        else:
            print("⚠ Data validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False

    def prepare_for_modeling(self, test_size=0.2, random_state=42):
        """Prepare data for modeling with train/test split.

        Parameters
        ----------
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility

        Returns
        -------
        Dict containing train and test sets

        """
        if self.data is None:
            raise ValueError("No data loaded")

        # Validate data first
        self.validate_data()

        # Sort by year to maintain temporal order
        df = self.data.sort_index()

        # Split data temporally (not randomly for time series)
        n_total = len(df)
        n_train = int(n_total * (1 - test_size))

        train_data = df.iloc[:n_train]
        test_data = df.iloc[n_train:]

        return {
            "train": train_data,
            "test": test_data,
            "train_years": (train_data.index.min(), train_data.index.max()),
            "test_years": (test_data.index.min(), test_data.index.max()),
        }
