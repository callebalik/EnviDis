"""Data Visualization Script for Time Series Analysis
This script provides visualization functions for time series data.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# X-axis time scale resolution constants
DAILY_DATA_THRESHOLD = (
    1000  # Threshold for considering data as "daily" (high frequency)
)
DAILY_SAMPLING_INTERVAL = 30  # Sample every N points for daily data readability
MONTHLY_AGGREGATION_THRESHOLD = 1000  # Threshold for aggregating daily to monthly
MAX_XTICKS_DISPLAY = 20  # Maximum number of x-axis ticks to display
XTICK_STEP_DIVISOR = 10  # Divisor to calculate x-tick step size

# Datetime x-axis formatting constants
YEAR_LABEL_INTERVAL = 5  # Interval in years between x-axis labels for datetime data
DATE_FORMAT = "%Y"  # Date format for x-axis labels
ENABLE_YEAR_LOCATOR = True  # Whether to use YearLocator for datetime x-axis

# Plot display control
SHOW_PLOTS = False  # Whether to display plots with plt.show()


def plot_entities_over_time(data: pd.DataFrame, save_path: str | None = None) -> None:
    """Plot entities over time with standardized x-axis formatting."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(data.index, data["co_count"], linewidth=2, alpha=0.8)
    ax.set_title("Entities Over Time", fontsize=14, fontweight="bold")
    ax.set_ylabel("Entity Count", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)

    # Standardized x-axis formatting - every 5 years with yearly ticks
    ax.xaxis.set_major_locator(mdates.YearLocator(5))  # Major ticks every 5 years
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))  # Minor ticks every year
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Rotate labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, which="major")
    ax.grid(True, alpha=0.1, which="minor")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close()


def plot_entities_with_documents_histogram(
    data: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Plot entities with documents histogram with standardized x-axis formatting."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Entities over time
    ax1.plot(data.index, data["co_count"], linewidth=2, alpha=0.8, label="Entities")
    ax1.set_title("Entities Over Time", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Entity Count", fontsize=12)
    ax1.legend()

    # Standardized x-axis formatting
    ax1.xaxis.set_major_locator(mdates.YearLocator(5))
    ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    ax1.grid(True, alpha=0.3, which="major")
    ax1.grid(True, alpha=0.1, which="minor")

    # Documents over time
    ax2.bar(data.index, data["document_count"], alpha=0.6, label="Documents")
    ax2.set_title("Documents Over Time", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Document Count", fontsize=12)
    ax2.set_xlabel("Year", fontsize=12)
    ax2.legend()

    # Standardized x-axis formatting
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax2.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    ax2.grid(True, alpha=0.3, which="major")
    ax2.grid(True, alpha=0.1, which="minor")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close()


def plot_entities_normalized_and_absolute(data, save_path=None):
    """Plot normalized and absolute entities with standardized x-axis formatting."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Absolute counts
    ax.plot(
        data.index,
        data["co_count"],
        linewidth=2,
        alpha=0.8,
        label="Absolute Counts",
        color="blue",
    )

    # Normalized counts (per 1000 documents)
    ax2 = ax.twinx()
    normalized = (data["co_count"] / data["document_count"]) * 1000
    ax2.plot(
        data.index,
        normalized,
        linewidth=2,
        alpha=0.8,
        label="Per 1K Documents",
        color="red",
        linestyle="--",
    )

    ax.set_title("Absolute vs Normalized Entity Counts", fontsize=14, fontweight="bold")
    ax.set_ylabel("Absolute Counts", color="blue", fontsize=12)
    ax2.set_ylabel("Entities per 1K Documents", color="red", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)

    # Standardized x-axis formatting
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.grid(True, alpha=0.3, which="major")
    ax.grid(True, alpha=0.1, which="minor")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close()


def plot_documents_over_time(data: pd.DataFrame, save_path: str | None = None) -> None:
    """Plot documents over time with standardized x-axis formatting."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(data.index, data["document_count"], alpha=0.7)
    ax.set_title("Documents Over Time", fontsize=14, fontweight="bold")
    ax.set_ylabel("Document Count", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)

    # Standardized x-axis formatting - every 5 years with yearly ticks
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Rotate labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, which="major")
    ax.grid(True, alpha=0.1, which="minor")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close()


def plot_entities_vs_documents(
    data: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Plot observed entities vs total documents with year as color.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with co_count and document_count columns
    save_path : str, optional
        Path to save the plot

    """
    plt.figure(figsize=(10, 7))

    # Convert datetime index to numeric for coloring
    if hasattr(data.index, "year"):
        # For datetime index, use years as numeric values
        hue_values = data.index.year
        hue_label = "Year"
    else:
        # For numeric index, use as is
        hue_values = data.index
        hue_label = "Index"

    sns.scatterplot(
        x="document_count",
        y="co_count",
        data=data,
        hue=hue_values,
        palette="viridis",
        s=100,
        legend=False,  # Disable automatic legend to avoid "best" location
    )
    plt.title("Observed Entities vs. Total Documents (Color by Year)")
    plt.xlabel("Total Documents")
    plt.ylabel("Observed Entities")
    plt.grid(True)

    # Add a custom colorbar instead of legend for large datasets
    if len(data) > DAILY_DATA_THRESHOLD // 10:  # For large datasets, use colorbar
        # Create colorbar with numeric values
        norm = plt.Normalize(vmin=hue_values.min(), vmax=hue_values.max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label(hue_label, rotation=270, labelpad=15)
    else:
        # For smaller datasets, create a simple legend
        plt.legend(title=hue_label, loc="upper left", bbox_to_anchor=(1, 1))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()


def create_all_plots(data: pd.DataFrame, output_dir: str | None = None) -> pd.DataFrame:
    """Create all visualization plots for the data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with the time series data
    output_dir : str, optional
        Directory to save plots

    """
    if output_dir:
        import os

        os.makedirs(output_dir, exist_ok=True)
        plot_entities_over_time(data, f"{output_dir}/entities_over_time.png")
        plot_documents_over_time(data, f"{output_dir}/documents_over_time.png")
        plot_entities_vs_documents(data, f"{output_dir}/entities_vs_documents.png")
        plot_entities_with_documents_histogram(
            data,
            f"{output_dir}/entities_with_documents_histogram.png",
        )
        normalized_data = plot_entities_normalized_and_absolute(
            data,
            f"{output_dir}/entities_normalized_and_absolute.png",
        )
    else:
        plot_entities_over_time(data)
        plot_documents_over_time(data)
        plot_entities_vs_documents(data)
        plot_entities_with_documents_histogram(data)
        normalized_data = plot_entities_normalized_and_absolute(data)

    return normalized_data


if __name__ == "__main__":
    # Load data and create plots - updated for daily data
    try:
        # Try daily data first
        data = pd.read_csv(
            "/home/callebalik/EnviDis/results/analysis/daily_time_series_data.csv",
            index_col="Date",
            parse_dates=True,
        )
        create_all_plots(data, "/home/callebalik/EnviDis/results/analysis/plots")
    except FileNotFoundError:
        try:
            # Fall back to yearly data
            data = pd.read_csv(
                "/home/callebalik/EnviDis/data/processed/sample_time_series_data.csv",
                index_col="Year",
            )
            create_all_plots(data)
        except FileNotFoundError:
            print("Sample data not found. Please run main_analysis.py first.")
