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


def plot_entities_over_time(data, save_path=None) -> None:
    """Plot observed entities over time.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with Date/Year as index and co_count column
    save_path : str, optional
        Path to save the plot

    """
    plt.figure(figsize=(12, 6))

    # For daily data, sample for readability
    if len(data) > DAILY_DATA_THRESHOLD:
        sample_data = data.iloc[::DAILY_SAMPLING_INTERVAL]  # Sample every N points
        sns.lineplot(
            x=sample_data.index,
            y="co_count",
            data=sample_data,
            marker="o",
        )
        plt.title("Observed Entities Over Time (Sampled)")
    else:
        sns.lineplot(x=data.index, y="co_count", data=data, marker="o")
        plt.title("Observed Entities Over Time")

    # Handle datetime vs numeric index
    if hasattr(data.index, "year"):
        plt.xlabel("Date")
        if ENABLE_YEAR_LOCATOR:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(DATE_FORMAT))
            plt.gca().xaxis.set_major_locator(mdates.YearLocator(YEAR_LABEL_INTERVAL))
        plt.xticks(rotation=45)
    else:
        plt.xlabel("Time Period")

    plt.ylabel("Number of Observed Entities")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_entities_with_documents_histogram(data, save_path=None) -> None:
    """Plot observed entities over time with total documents histogram.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with Date/Year as index, co_count and document_count columns
    save_path : str, optional
        Path to save the plot

    """
    # For daily data, aggregate to monthly for better visualization
    if len(data) > MONTHLY_AGGREGATION_THRESHOLD and hasattr(data.index, "year"):
        plot_data = data.resample("M").sum()
        title_suffix = " (Monthly Aggregated)"
    else:
        plot_data = data
        title_suffix = ""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

    # Main plot: Entities over time
    ax1.plot(
        plot_data.index,
        plot_data["co_count"],
        marker="o",
        linewidth=2,
        markersize=6,
        label="Observed Entities",
    )
    ax1.set_ylabel("Number of Observed Entities", fontsize=12)
    ax1.set_title(
        f"Observed Entities Over Time with Document Volume Context{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Histogram: Total documents distribution over time
    width = 0.8 if len(plot_data) < 100 else 20  # Adjust bar width for daily data
    ax2.bar(
        plot_data.index,
        plot_data["document_count"],
        alpha=0.7,
        color="orange",
        width=width,
    )
    ax2.set_ylabel("Total Documents", fontsize=12)
    ax2.set_title(f"Total Documents Volume{title_suffix}", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    # Handle datetime formatting
    if hasattr(plot_data.index, "year"):
        if ENABLE_YEAR_LOCATOR:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FORMAT))
            ax1.xaxis.set_major_locator(mdates.YearLocator(YEAR_LABEL_INTERVAL))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FORMAT))
            ax2.xaxis.set_major_locator(mdates.YearLocator(YEAR_LABEL_INTERVAL))
        ax1.tick_params(axis="x", rotation=45)
        ax2.tick_params(axis="x", rotation=45)
        ax2.set_xlabel("Date", fontsize=12)
    else:
        # Format x-axis to show fewer ticks if too many points
        if len(plot_data.index) > MAX_XTICKS_DISPLAY:
            step = max(1, len(plot_data.index) // XTICK_STEP_DIVISOR)
            ax1.set_xticks(plot_data.index[::step])
            ax2.set_xticks(plot_data.index[::step])
            ax1.tick_params(axis="x", rotation=45)
            ax2.tick_params(axis="x", rotation=45)
        ax2.set_xlabel("Time Period", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_entities_normalized_and_absolute(data, save_path=None):
    """Plot both absolute entities and entities normalized by total documents.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with Date/Year as index, co_count and document_count columns
    save_path : str, optional
        Path to save the plot

    """
    # For daily data, aggregate to monthly for better visualization
    if len(data) > MONTHLY_AGGREGATION_THRESHOLD and hasattr(data.index, "year"):
        plot_data = data.resample("M").sum()
        title_suffix = " (Monthly Aggregated)"
    else:
        plot_data = data.copy()
        title_suffix = ""

    # Calculate normalized entities (entities per document * 1000 for readability)
    data_with_normalized = plot_data.copy()
    data_with_normalized["NormalizedEntities"] = (
        plot_data["co_count"] / plot_data["document_count"]
    ) * 1000

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Top plot: Absolute entities with document histogram
    ax1_hist = ax1.twinx()

    # Line plot for entities
    ax1.plot(
        plot_data.index,
        plot_data["co_count"],
        marker="o",
        linewidth=2,
        markersize=6,
        color="blue",
        label="Observed Entities",
    )
    ax1.set_ylabel("Number of Observed Entities", color="blue", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_title(
        f"Absolute Entities vs Document Volume{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    # Bar plot for documents
    width = 0.8 if len(plot_data) < 100 else 20
    ax1_hist.bar(
        plot_data.index,
        plot_data["document_count"],
        alpha=0.3,
        color="orange",
        width=width,
        label="Total Documents",
    )
    ax1_hist.set_ylabel("Total Documents", color="orange", fontsize=12)
    ax1_hist.tick_params(axis="y", labelcolor="orange")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_hist.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Bottom plot: Normalized entities
    ax2.plot(
        plot_data.index,
        data_with_normalized["NormalizedEntities"],
        marker="s",
        linewidth=2,
        markersize=6,
        color="green",
        label="Entities per 1000 Documents",
    )
    ax2.set_ylabel("Entities per 1000 Documents", color="green", fontsize=12)
    ax2.set_title(
        f"Document-Normalized Entity Trends{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.tick_params(axis="y", labelcolor="green")

    # Handle datetime formatting
    if hasattr(plot_data.index, "year"):
        if ENABLE_YEAR_LOCATOR:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FORMAT))
            ax1.xaxis.set_major_locator(mdates.YearLocator(YEAR_LABEL_INTERVAL))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FORMAT))
            ax2.xaxis.set_major_locator(mdates.YearLocator(YEAR_LABEL_INTERVAL))
        ax1.tick_params(axis="x", rotation=45)
        ax2.tick_params(axis="x", rotation=45)
        ax2.set_xlabel("Date", fontsize=12)
    else:
        # Format x-axis
        if len(plot_data.index) > MAX_XTICKS_DISPLAY:
            step = max(1, len(plot_data.index) // XTICK_STEP_DIVISOR)
            ax1.set_xticks(plot_data.index[::step])
            ax2.set_xticks(plot_data.index[::step])
            ax1.tick_params(axis="x", rotation=45)
            ax2.tick_params(axis="x", rotation=45)
        ax2.set_xlabel("Time Period", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return data_with_normalized


def plot_documents_over_time(data: pd.DataFrame, save_path: str | None = None) -> None:
    """Plot total documents over time.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with Date/Year as index and document_count column
    save_path : str, optional
        Path to save the plot

    """
    plt.figure(figsize=(12, 6))

    # For daily data, sample for readability
    if len(data) > DAILY_DATA_THRESHOLD:
        sample_data = data.iloc[::DAILY_SAMPLING_INTERVAL]
        sns.lineplot(
            x=sample_data.index,
            y="document_count",
            data=sample_data,
            marker="o",
            color="orange",
        )
        plt.title("Total Documents Over Time (Sampled)")
    else:
        sns.lineplot(
            x=data.index,
            y="document_count",
            data=data,
            marker="o",
            color="orange",
        )
        plt.title("Total Documents Over Time")

    # Handle datetime vs numeric index
    if hasattr(data.index, "year"):
        plt.xlabel("Date")
        if ENABLE_YEAR_LOCATOR:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(DATE_FORMAT))
            plt.gca().xaxis.set_major_locator(mdates.YearLocator(YEAR_LABEL_INTERVAL))
        plt.xticks(rotation=45)
    else:
        plt.xlabel("Time Period")

    plt.ylabel("Total Number of Documents")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


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
