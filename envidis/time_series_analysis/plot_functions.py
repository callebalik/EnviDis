#!/usr/bin/env python3
"""Individual Plotting Functions for Time Series Analysis
This module contains individual plotting functions that can be used independently
or combined in comprehensive visualizations.
"""

import warnings
from typing import Any, Dict, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def plot_raw_timeseries(
    data: pd.DataFrame,
    ax: plt.Axes | None = None,
    sample_rate: int = 50,
    **kwargs,
) -> plt.Axes:
    """Plot raw time series data as scatter points.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with datetime index and 'ObservedEntities' column
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, current axes used
    sample_rate : int
        Sample every nth point for visibility (default: 50, reduced from 100)
    **kwargs : dict
        Additional plotting parameters

    Returns
    -------
    plt.Axes
        The axes object with the plot

    """
    if ax is None:
        ax = plt.gca()

    # Default styling for scatter plot
    default_kwargs = {
        "color": "blue",
        "alpha": 0.6,
        "s": 2,  # marker size for scatter
        "marker": ".",
        "title": "Co-occurrence Counts Over Time",
        "ylabel": "Co-occurrence Count",
        "remove_outliers": True,  # New parameter to control outlier removal
        "outlier_percentile": 99.7,  # New configurable parameter
        "filter_sparse_years": True,  # New parameter to filter sparse early years
        "min_year_density": 0.01,  # Minimum fraction of years that must have data
    }
    default_kwargs.update(kwargs)

    # Filter out sparse early years that create misleading x-axis scaling
    if default_kwargs.get("filter_sparse_years", True):
        # Calculate data density by year
        yearly_counts = data.groupby(data.index.year).size()
        total_years = data.index.year.max() - data.index.year.min() + 1
        data_years = len(yearly_counts)
        density = data_years / total_years

        # If overall density is low, find the main continuous period
        if (
            density < default_kwargs.get("min_year_density", 0.01)
            or yearly_counts.min() == 1
        ):
            # Find the year where continuous data starts (remove isolated early points)
            sorted_years = sorted(yearly_counts.index)

            # Look for the first year where we have more consistent coverage
            # (more than 1 observation and not isolated)
            start_year = sorted_years[0]
            for i, year in enumerate(sorted_years[:-1]):
                next_year = sorted_years[i + 1]
                # If there's a gap > 50 years and current year has very few points, skip it
                if next_year - year > 50 and yearly_counts[year] < 10:
                    start_year = next_year
                else:
                    break

            # Filter data to start from the main period
            if start_year > data.index.year.min():
                filtered_data = data[data.index.year >= start_year]
                years_removed = start_year - data.index.year.min()
                default_kwargs[
                    "title"
                ] += f" (from {start_year}, excl. {years_removed} sparse early years)"
            else:
                filtered_data = data
        else:
            filtered_data = data
    else:
        filtered_data = data

    # Filter out extreme outliers using configurable percentile
    if default_kwargs.get("remove_outliers", True):
        percentile_threshold = default_kwargs.get("outlier_percentile", 99.7)
        percentile_value = filtered_data["ObservedEntities"].quantile(
            percentile_threshold / 100
        )
        outlier_data = filtered_data[
            filtered_data["ObservedEntities"] > percentile_value
        ]
        filtered_data = filtered_data[
            filtered_data["ObservedEntities"] <= percentile_value
        ]

        # Add detailed note about filtering with specific information
        if len(outlier_data) > 0:
            outlier_years = sorted(outlier_data.index.year.unique())
            max_outlier = outlier_data["ObservedEntities"].max()

            if len(outlier_years) <= 5:
                year_list = ", ".join(map(str, outlier_years))
                outlier_detail = f"years {year_list}"
            else:
                outlier_detail = f"{len(outlier_years)} years ({outlier_years[0]}-{outlier_years[-1]})"

            if "sparse early years" not in default_kwargs["title"]:
                default_kwargs[
                    "title"
                ] += f" (excl. >{percentile_threshold:.1f}%ile: {outlier_detail}, max={max_outlier:,})"

    # Sample data for visibility
    sample_data = filtered_data.iloc[::sample_rate]

    ax.scatter(
        sample_data.index,
        sample_data["ObservedEntities"],
        color=default_kwargs["color"],
        alpha=default_kwargs["alpha"],
        s=default_kwargs["s"],
    )

    # Set better y-axis limits to focus on the main data range
    if len(sample_data) > 0:
        y_max = sample_data["ObservedEntities"].quantile(
            0.95,
        )  # Use 95th percentile for upper limit
        y_min = max(0, sample_data["ObservedEntities"].min())
        ax.set_ylim(y_min, y_max * 1.05)  # Add 5% padding

    ax.set_title(default_kwargs["title"], fontsize=12, fontweight="bold")
    ax.set_ylabel(default_kwargs["ylabel"])

    # Improve x-axis formatting for better time resolution
    # Adjust based on the actual date range after filtering
    date_range_years = sample_data.index.year.max() - sample_data.index.year.min()

    if date_range_years > 100:
        ax.xaxis.set_major_locator(mdates.YearLocator(10))  # Major ticks every 10 years
        ax.xaxis.set_minor_locator(mdates.YearLocator(5))  # Minor ticks every 5 years
    elif date_range_years > 50:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))  # Major ticks every 5 years
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))  # Minor ticks every year
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))  # Major ticks every 2 years
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))  # Minor ticks every year

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Rotate labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, which="major")
    ax.grid(True, alpha=0.1, which="minor")

    return ax


def plot_yearly_aggregated(
    yearly_data: pd.DataFrame,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot yearly aggregated data as scatter points with optional trend line and document context.

    Parameters
    ----------
    yearly_data : pd.DataFrame
        DataFrame with 'Year' and 'ObservedEntities' columns
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    **kwargs : dict
        Additional plotting parameters

    Returns
    -------
    plt.Axes
        The axes object with the plot

    """
    if ax is None:
        ax = plt.gca()

    default_kwargs = {
        "color": "red",
        "marker": "o",
        "markersize": 4,
        "alpha": 0.7,
        "show_trend": True,
        "show_rate": True,  # New option to show co-occurrence rate
        "trend_color": "darkred",
        "trend_alpha": 0.5,
        "rate_color": "blue",
        "rate_alpha": 0.3,
        "title": "Annual Co-occurrence Totals",
        "ylabel": "Annual Total",
        "remove_outliers": False,  # Changed to False - don't remove outliers for yearly data
        "filter_sparse_years": True,  # Keep sparse year filtering
    }
    default_kwargs.update(kwargs)

    # Filter out sparse early years that create misleading x-axis scaling
    if default_kwargs.get("filter_sparse_years", True):
        # Check for isolated early years (similar logic as raw timeseries)
        sorted_years = sorted(yearly_data["Year"])

        start_year = sorted_years[0]
        for i, year in enumerate(sorted_years[:-1]):
            next_year = sorted_years[i + 1]
            # If there's a gap > 50 years and very few observations, skip early years
            year_data = yearly_data[yearly_data["Year"] == year]
            if (
                next_year - year > 50
                and len(year_data) == 1
                and year_data["ObservedEntities"].iloc[0] < 10
            ):
                start_year = next_year
            else:
                break

        # Filter data to start from the main period
        if start_year > yearly_data["Year"].min():
            filtered_data = yearly_data[yearly_data["Year"] >= start_year]
            years_removed = start_year - yearly_data["Year"].min()
            default_kwargs["title"] += f" (from {start_year})"
        else:
            filtered_data = yearly_data
    else:
        filtered_data = yearly_data

    # Do NOT filter out extreme outlier years - show all data points
    # This allows users to see the full range of annual variations
    final_filtered_data = filtered_data

    # Plot as scatter points
    ax.scatter(
        final_filtered_data["Year"],
        final_filtered_data["ObservedEntities"],
        color=default_kwargs["color"],
        marker=default_kwargs["marker"],
        s=default_kwargs["markersize"] ** 2,
        alpha=default_kwargs["alpha"],
        label="Annual Co-occurrence Totals",
        zorder=3,
    )

    # Set y-axis limits to show all data (no filtering)
    if len(final_filtered_data) > 0:
        y_max = final_filtered_data["ObservedEntities"].max()
        y_min = max(0, final_filtered_data["ObservedEntities"].min())
        ax.set_ylim(y_min, y_max * 1.05)  # Add 5% padding above max

    # Add trend line if requested (use all data)
    if default_kwargs.get("show_trend", True) and len(final_filtered_data) > 3:
        try:
            z = np.polyfit(
                final_filtered_data["Year"],
                final_filtered_data["ObservedEntities"],
                1,
            )
            p = np.poly1d(z)
            ax.plot(
                final_filtered_data["Year"],
                p(final_filtered_data["Year"]),
                color=default_kwargs["trend_color"],
                alpha=default_kwargs["trend_alpha"],
                linewidth=1.5,
                label="Linear Trend",
                zorder=2,
            )
        except:
            pass  # Skip trend if fitting fails

    # Add co-occurrence rate (co-occurrences per document) if document data available
    if (
        default_kwargs.get("show_rate", True)
        and "TotalDocuments" in final_filtered_data.columns
    ):
        try:
            # Calculate rate using all data
            rate = (
                final_filtered_data["ObservedEntities"]
                / final_filtered_data["TotalDocuments"]
            )

            ax2 = ax.twinx()
            ax2.plot(
                final_filtered_data["Year"],
                rate,
                color=default_kwargs["rate_color"],
                alpha=default_kwargs["rate_alpha"],
                linewidth=2,
                label="Co-occurrences per Document",
                linestyle="--",
            )
            ax2.set_ylabel(
                "Co-occurrences per Document",
                color=default_kwargs["rate_color"],
            )
            ax2.tick_params(axis="y", labelcolor=default_kwargs["rate_color"])

            # Add text annotation explaining the discrepancy
            ax.text(
                0.02,
                0.98,
                "Note: Rising totals may reflect\nincreasing publication volume\nrather than higher co-occurrence rates",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
            )
        except:
            pass  # Skip rate calculation if it fails

    ax.set_title(default_kwargs["title"], fontsize=12, fontweight="bold")
    ax.set_ylabel(default_kwargs["ylabel"])
    ax.set_xlabel("Year")

    # Improve x-axis resolution using filtered data
    years = filtered_data["Year"]
    year_range = years.max() - years.min()

    if year_range > 50:
        ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
    elif year_range > 20:
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    else:
        ax.xaxis.set_major_locator(plt.MultipleLocator(2))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

    # Add grid for better readability
    ax.grid(True, alpha=0.3, which="major")
    ax.grid(True, alpha=0.1, which="minor")

    # Rotate labels if needed
    if year_range > 30:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Combine legends if we have both plots
    lines1, labels1 = ax.get_legend_handles_labels()
    if "ax2" in locals():
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    else:
        ax.legend(fontsize=9)

    return ax


def plot_distribution(
    data: pd.DataFrame,
    ax: plt.Axes | None = None,
    log_scale: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plot distribution of co-occurrence counts.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with 'ObservedEntities' column
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    log_scale : bool
        Whether to use log scale for better visualization
    **kwargs : dict
        Additional plotting parameters

    Returns
    -------
    plt.Axes
        The axes object with the plot

    """
    if ax is None:
        ax = plt.gca()

    default_kwargs = {
        "bins": 50,
        "alpha": 0.7,
        "color": "green",
        "title": "Distribution of Co-occurrence Counts",
        "xlabel": "Count",
        "ylabel": "Frequency",
        "remove_outliers": True,  # New parameter to control outlier removal
        "outlier_percentile": 99.7,  # New configurable parameter
    }
    default_kwargs.update(kwargs)

    # Filter out extreme outliers using configurable percentile
    if default_kwargs.get("remove_outliers", True):
        percentile_threshold = default_kwargs.get("outlier_percentile", 99.7)
        percentile_value = data["ObservedEntities"].quantile(percentile_threshold / 100)
        outlier_data = data[data["ObservedEntities"] > percentile_value]
        filtered_data = data[data["ObservedEntities"] <= percentile_value]

        # Add detailed note about filtering
        if len(outlier_data) > 0:
            max_outlier = outlier_data["ObservedEntities"].max()
            outlier_count = len(outlier_data)
            outlier_percentage = (outlier_count / len(data)) * 100

            # Get unique dates for outliers to see if they cluster
            outlier_years = sorted(outlier_data.index.year.unique())
            if len(outlier_years) <= 3:
                year_info = f"from years {', '.join(map(str, outlier_years))}"
            else:
                year_info = f"across {len(outlier_years)} years"

            default_kwargs[
                "title"
            ] += f" (excl. >{percentile_threshold:.1f}%ile: {outlier_count:,} outliers, {outlier_percentage:.1f}%, max={max_outlier:,}, {year_info})"
    else:
        filtered_data = data

    if log_scale:
        non_zero_data = filtered_data[filtered_data["ObservedEntities"] > 0][
            "ObservedEntities"
        ]
        plot_data = np.log10(non_zero_data + 1)
        default_kwargs["title"] += " (Log Scale)"
        default_kwargs["xlabel"] = "Log10(Count + 1)"
    else:
        plot_data = filtered_data["ObservedEntities"]

    ax.hist(
        plot_data,
        bins=default_kwargs["bins"],
        alpha=default_kwargs["alpha"],
        color=default_kwargs["color"],
    )

    ax.set_title(default_kwargs["title"], fontsize=12, fontweight="bold")
    ax.set_xlabel(default_kwargs["xlabel"])
    ax.set_ylabel(default_kwargs["ylabel"])

    return ax


def plot_monthly_patterns(
    data: pd.DataFrame,
    ax: plt.Axes | None = None,
    start_year: int = 2020,
    **kwargs,
) -> plt.Axes:
    """Plot monthly patterns for recent years.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with datetime index and 'ObservedEntities' column
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    start_year : int
        Starting year for analysis (default: 2020)
    **kwargs : dict
        Additional plotting parameters

    Returns
    -------
    plt.Axes
        The axes object with the plot

    """
    if ax is None:
        ax = plt.gca()

    default_kwargs = {
        "color": "purple",
        "alpha": 0.7,
        "title": f"Monthly Pattern ({start_year}-2025)",
        "xlabel": "Month",
        "ylabel": "Average Count",
    }
    default_kwargs.update(kwargs)

    recent_data = data[data.index.year >= start_year]
    if len(recent_data) > 0:
        monthly_means = recent_data.groupby(recent_data.index.month)[
            "ObservedEntities"
        ].mean()
        ax.bar(
            monthly_means.index,
            monthly_means.values,
            color=default_kwargs["color"],
            alpha=default_kwargs["alpha"],
        )

    ax.set_title(default_kwargs["title"], fontsize=12, fontweight="bold")
    ax.set_xlabel(default_kwargs["xlabel"])
    ax.set_ylabel(default_kwargs["ylabel"])

    return ax


def plot_scatter_documents_vs_cooccurrence(
    data: pd.DataFrame,
    ax: plt.Axes | None = None,
    sample_size: int = 1000,
    **kwargs,
) -> plt.Axes:
    """Plot scatter plot of documents vs co-occurrences.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with 'TotalDocuments' and 'ObservedEntities' columns
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    sample_size : int
        Number of points to sample for plotting
    **kwargs : dict
        Additional plotting parameters

    Returns
    -------
    plt.Axes
        The axes object with the plot

    """
    if ax is None:
        ax = plt.gca()

    default_kwargs = {
        "alpha": 0.6,
        "s": 1,
        "color": "orange",
        "title": "Documents vs Co-occurrences",
        "xlabel": "Document Count",
        "ylabel": "Co-occurrence Count",
        "remove_outliers": True,
        "outlier_percentile": 99.7,  # New configurable parameter
    }
    default_kwargs.update(kwargs)

    # Filter out extreme outliers using configurable percentile
    if default_kwargs.get("remove_outliers", True):
        percentile_threshold = default_kwargs.get("outlier_percentile", 99.7)
        doc_percentile = data["TotalDocuments"].quantile(percentile_threshold / 100)
        cooc_percentile = data["ObservedEntities"].quantile(percentile_threshold / 100)

        doc_outliers = data[data["TotalDocuments"] > doc_percentile]
        cooc_outliers = data[data["ObservedEntities"] > cooc_percentile]

        filtered_data = data[
            (data["TotalDocuments"] <= doc_percentile)
            & (data["ObservedEntities"] <= cooc_percentile)
        ]

        # Add detailed note about filtering
        outliers_removed = len(data) - len(filtered_data)
        if outliers_removed > 0:
            outlier_details = []
            if len(doc_outliers) > 0:
                max_docs = doc_outliers["TotalDocuments"].max()
                outlier_details.append(f"docs>{doc_percentile:.0f} (max={max_docs:,})")
            if len(cooc_outliers) > 0:
                max_cooc = cooc_outliers["ObservedEntities"].max()
                outlier_details.append(f"cooc>{cooc_percentile:.0f} (max={max_cooc:,})")

            default_kwargs[
                "title"
            ] += f" (excl. >{percentile_threshold:.1f}%ile: {outliers_removed:,} outliers: {', '.join(outlier_details)})"
    else:
        filtered_data = data

    sample_data = filtered_data.sample(
        n=min(sample_size, len(filtered_data)),
        random_state=42,
    )

    ax.scatter(
        sample_data["TotalDocuments"],
        sample_data["ObservedEntities"],
        alpha=default_kwargs["alpha"],
        s=default_kwargs["s"],
        color=default_kwargs["color"],
    )

    ax.set_title(default_kwargs["title"], fontsize=12, fontweight="bold")
    ax.set_xlabel(default_kwargs["xlabel"])
    ax.set_ylabel(default_kwargs["ylabel"])

    return ax


def plot_model_summary_table(
    model_summary: pd.DataFrame,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot model summary as a table.

    Parameters
    ----------
    model_summary : pd.DataFrame
        DataFrame with model summary statistics
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    **kwargs : dict
        Additional plotting parameters

    Returns
    -------
    plt.Axes
        The axes object with the table

    """
    if ax is None:
        ax = plt.gca()

    default_kwargs = {
        "title": "Model Comparison",
        "fontsize": 7,  # Reduced from 9 for A4 compatibility
        "scale": (1.0, 1.2),  # Reduced scale for A4 width
    }
    default_kwargs.update(kwargs)

    ax.axis("tight")
    ax.axis("off")

    if len(model_summary) > 0:
        table_data = []
        for _, row in model_summary.iterrows():
            table_data.append(
                [row["Model"], row["AIC"], row["Pseudo R²"], row["N Observations"]],
            )

        table = ax.table(
            cellText=table_data,
            colLabels=["Model", "AIC", "Pseudo R²", "N Obs"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(default_kwargs["fontsize"])
        table.scale(*default_kwargs["scale"])

    ax.set_title(default_kwargs["title"], fontsize=12, fontweight="bold")

    return ax


def plot_trend_analysis(
    yearly_data: pd.DataFrame,
    ax: plt.Axes | None = None,
    fit_trend: bool = True,
    trend_degree: int = 2,
    **kwargs,
) -> plt.Axes:
    """Plot long-term trend analysis with scatter points and trend lines.

    Parameters
    ----------
    yearly_data : pd.DataFrame
        DataFrame with 'Year' and 'ObservedEntities' columns
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    fit_trend : bool
        Whether to fit and plot trend line
    trend_degree : int
        Degree of polynomial for trend fitting
    **kwargs : dict
        Additional plotting parameters

    Returns
    -------
    plt.Axes
        The axes object with the plot

    """
    if ax is None:
        ax = plt.gca()

    default_kwargs = {
        "observed_color": "blue",
        "trend_color": "red",
        "markersize": 4,
        "marker_alpha": 0.7,
        "trend_linewidth": 2,
        "trend_alpha": 0.8,
        "title": "Long-term Trend Analysis",
        "xlabel": "Year",
        "ylabel": "Annual Total",
        "remove_outliers": True,
        "outlier_percentile": 99.7,  # New configurable parameter
    }
    default_kwargs.update(kwargs)

    # Filter outliers using configurable percentile
    filtered_data = yearly_data
    if default_kwargs.get("remove_outliers", True) and len(yearly_data) > 0:
        percentile_threshold = default_kwargs.get("outlier_percentile", 99.7)
        percentile_value = yearly_data["ObservedEntities"].quantile(
            percentile_threshold / 100
        )
        outlier_data = yearly_data[yearly_data["ObservedEntities"] > percentile_value]
        filtered_data = yearly_data[yearly_data["ObservedEntities"] <= percentile_value]

        if len(outlier_data) > 0:
            outlier_years = sorted(outlier_data["Year"].tolist())
            outlier_values = outlier_data.set_index("Year")[
                "ObservedEntities"
            ].to_dict()

            if len(outlier_years) <= 3:
                year_details = [
                    f"{year}({outlier_values[year]:,})" for year in outlier_years
                ]
                outlier_detail = f"outlier years {', '.join(year_details)}"
            else:
                max_year = max(outlier_years, key=lambda y: outlier_values[y])
                outlier_detail = f"{len(outlier_years)} outlier years (peak: {max_year}={outlier_values[max_year]:,})"

            default_kwargs[
                "title"
            ] += f" (excl. >{percentile_threshold:.1f}%ile: {outlier_detail})"

    if len(filtered_data) > 10:
        years = filtered_data["Year"]
        counts = filtered_data["ObservedEntities"]

        # Plot observed data as scatter points
        ax.scatter(
            years,
            counts,
            color=default_kwargs["observed_color"],
            s=default_kwargs["markersize"] ** 2,
            alpha=default_kwargs["marker_alpha"],
            label="Observed",
            zorder=3,
        )

        # Set better y-axis limits
        y_max = counts.quantile(0.95)
        y_min = max(0, counts.min())
        ax.set_ylim(y_min, y_max * 1.05)

        if fit_trend:
            # Fit polynomial trend
            try:
                z = np.polyfit(years, counts, trend_degree)
                p = np.poly1d(z)
                trend_label = (
                    f'{"Quadratic" if trend_degree == 2 else "Polynomial"} Trend'
                )
                ax.plot(
                    years,
                    p(years),
                    color=default_kwargs["trend_color"],
                    linewidth=default_kwargs["trend_linewidth"],
                    alpha=default_kwargs["trend_alpha"],
                    label=trend_label,
                    zorder=2,
                )
            except:
                pass  # Skip trend if fitting fails

        ax.legend(fontsize=9)

        # Improve x-axis resolution
        year_range = years.max() - years.min()
        if year_range > 50:
            ax.xaxis.set_major_locator(plt.MultipleLocator(10))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
        elif year_range > 20:
            ax.xaxis.set_major_locator(plt.MultipleLocator(5))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
        else:
            ax.xaxis.set_major_locator(plt.MultipleLocator(2))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

        # Add grid
        ax.grid(True, alpha=0.3, which="major")
        ax.grid(True, alpha=0.1, which="minor")

        # Rotate labels if needed
        if year_range > 30:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    ax.set_title(default_kwargs["title"], fontsize=12, fontweight="bold")
    ax.set_xlabel(default_kwargs["xlabel"])
    ax.set_ylabel(default_kwargs["ylabel"])

    return ax


def plot_zero_inflation_pie(
    data: pd.DataFrame,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot pie chart showing zero vs non-zero co-occurrences.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with 'ObservedEntities' column
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    **kwargs : dict
        Additional plotting parameters

    Returns
    -------
    plt.Axes
        The axes object with the plot

    """
    if ax is None:
        ax = plt.gca()

    default_kwargs = {
        "colors": ["lightcoral", "lightblue"],
        "autopct": "%1.1f%%",
        "title": "Zero-Inflation Analysis",
    }
    default_kwargs.update(kwargs)

    zero_counts = (data["ObservedEntities"] == 0).sum()
    non_zero_counts = (data["ObservedEntities"] > 0).sum()

    ax.pie(
        [zero_counts, non_zero_counts],
        labels=["Zero Co-occurrences", "Non-zero Co-occurrences"],
        autopct=default_kwargs["autopct"],
        colors=default_kwargs["colors"],
    )

    ax.set_title(default_kwargs["title"], fontsize=12, fontweight="bold")

    return ax


def plot_recent_trend_bar(
    data: pd.DataFrame,
    ax: plt.Axes | None = None,
    start_year: int = 2020,
    **kwargs,
) -> plt.Axes:
    """Plot recent trend as bar chart.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with datetime index and 'ObservedEntities' column
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    start_year : int
        Starting year for analysis
    **kwargs : dict
        Additional plotting parameters

    Returns
    -------
    plt.Axes
        The axes object with the plot

    """
    if ax is None:
        ax = plt.gca()

    default_kwargs = {
        "color": "teal",
        "alpha": 0.7,
        "title": f"Recent Trend ({start_year}-2025)",
        "xlabel": "Year",
        "ylabel": "Annual Total",
        "remove_outliers": True,
        "outlier_percentile": 99.7,  # New configurable parameter
    }
    default_kwargs.update(kwargs)

    recent_data_filter = data.index.year >= start_year
    recent_data_subset = data[recent_data_filter]

    if len(recent_data_subset) > 0:
        recent_yearly = recent_data_subset.groupby(recent_data_subset.index.year)[
            "ObservedEntities"
        ].sum()

        # Filter outliers using configurable percentile
        if default_kwargs.get("remove_outliers", True):
            percentile_threshold = default_kwargs.get("outlier_percentile", 99.7)
            percentile_value = recent_yearly.quantile(percentile_threshold / 100)
            outlier_mask = recent_yearly > percentile_value
            outlier_years = recent_yearly[outlier_mask].index.tolist()
            outlier_values = recent_yearly[outlier_mask].to_dict()
            filtered_yearly = recent_yearly[recent_yearly <= percentile_value]

            if len(outlier_years) > 0:
                if len(outlier_years) <= 2:
                    year_details = [
                        f"{year}({outlier_values[year]:,})" for year in outlier_years
                    ]
                    outlier_detail = f"outlier years {', '.join(year_details)}"
                else:
                    max_year = max(outlier_years, key=lambda y: outlier_values[y])
                    outlier_detail = f"{len(outlier_years)} outlier years (peak: {max_year}={outlier_values[max_year]:,})"

                default_kwargs[
                    "title"
                ] += f" (excl. >{percentile_threshold:.1f}%ile: {outlier_detail})"
        else:
            filtered_yearly = recent_yearly

        ax.bar(
            filtered_yearly.index,
            filtered_yearly.values,
            color=default_kwargs["color"],
            alpha=default_kwargs["alpha"],
        )

        # Set better y-axis limits
        if len(filtered_yearly) > 0:
            y_max = filtered_yearly.quantile(0.95)
            y_min = max(0, filtered_yearly.min())
            ax.set_ylim(y_min, y_max * 1.05)

    ax.set_title(default_kwargs["title"], fontsize=12, fontweight="bold")
    ax.set_xlabel(default_kwargs["xlabel"])
    ax.set_ylabel(default_kwargs["ylabel"])

    return ax


def plot_coefficients_table(
    coef_table: pd.DataFrame,
    ax: plt.Axes | None = None,
    model_filter: str = "quadratic",
    **kwargs,
) -> plt.Axes:
    """Plot coefficients table for best model.

    Parameters
    ----------
    coef_table : pd.DataFrame
        DataFrame with coefficient information
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    model_filter : str
        Model name to filter for (default: 'quadratic')
    **kwargs : dict
        Additional plotting parameters

    Returns
    -------
    plt.Axes
        The axes object with the table

    """
    if ax is None:
        ax = plt.gca()

    default_kwargs = {
        "title": "Best Model Coefficients",
        "fontsize": 6,  # Reduced from 8 for A4 compatibility
        "scale": (1.0, 1.2),  # Reduced scale for A4 width
    }
    default_kwargs.update(kwargs)

    ax.axis("tight")
    ax.axis("off")

    if len(coef_table) > 0 and model_filter in [
        m.lower() for m in coef_table["Model"].values
    ]:
        best_model_coefs = coef_table[
            coef_table["Model"].str.lower() == model_filter.lower()
        ]
        if len(best_model_coefs) > 0:
            table_data = []
            for _, row in best_model_coefs.iterrows():
                table_data.append(
                    [
                        row["Parameter"],
                        row["Coefficient"],
                        row["P-value"],
                        row["Significant"],
                    ],
                )

            table = ax.table(
                cellText=table_data,
                colLabels=["Parameter", "Coefficient", "P-value", "Significance"],
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(default_kwargs["fontsize"])
            table.scale(*default_kwargs["scale"])

    ax.set_title(default_kwargs["title"], fontsize=12, fontweight="bold")

    return ax


def plot_data_quality_table(
    data: pd.DataFrame,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot data quality metrics as table.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with datetime index and 'ObservedEntities' column
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    **kwargs : dict
        Additional plotting parameters

    Returns
    -------
    plt.Axes
        The axes object with the table

    """
    if ax is None:
        ax = plt.gca()

    default_kwargs = {
        "title": "Data Quality Summary",
        "fontsize": 7,  # Reduced from 9 for A4 compatibility
        "scale": (1.0, 1.2),  # Reduced scale for A4 width
    }
    default_kwargs.update(kwargs)

    ax.axis("tight")
    ax.axis("off")

    quality_metrics = [
        ["Total Observations", f"{len(data):,}"],
        ["Date Range", f"{data.index.min().date()} to {data.index.max().date()}"],
        [
            "Zero Values",
            f"{(data['ObservedEntities'] == 0).sum():,} ({(data['ObservedEntities'] == 0).mean()*100:.1f}%)",
        ],
        ["Max Co-occurrence", f"{data['ObservedEntities'].max():,}"],
        ["Mean Co-occurrence", f"{data['ObservedEntities'].mean():.2f}"],
        ["Years Covered", f"{data.index.year.nunique()}"],
    ]

    table = ax.table(
        cellText=quality_metrics,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(default_kwargs["fontsize"])
    table.scale(*default_kwargs["scale"])

    ax.set_title(default_kwargs["title"], fontsize=12, fontweight="bold")

    return ax


def plot_time_series_decomposition(
    yearly_data: pd.DataFrame,
    ax: plt.Axes | None = None,
    window: int | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot time series decomposition with moving average, showing points and smooth trend.

    Parameters
    ----------
    yearly_data : pd.DataFrame
        DataFrame with 'Year' and 'ObservedEntities' columns
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    window : int, optional
        Window size for moving average
    **kwargs : dict
        Additional plotting parameters

    Returns
    -------
    plt.Axes
        The axes object with the plot

    """
    if ax is None:
        ax = plt.gca()

    default_kwargs = {
        "original_alpha": 0.4,
        "original_markersize": 3,
        "trend_linewidth": 2.5,
        "trend_alpha": 0.8,
        "original_color": "blue",
        "trend_color": "red",
        "title": "Trend Smoothing",
        "xlabel": "Year",
        "ylabel": "Annual Total",
        "remove_outliers": True,
        "outlier_percentile": 99.7,  # New configurable parameter
    }
    default_kwargs.update(kwargs)

    # Filter outliers using configurable percentile
    filtered_data = yearly_data
    if default_kwargs.get("remove_outliers", True) and len(yearly_data) > 0:
        percentile_threshold = default_kwargs.get("outlier_percentile", 99.7)
        percentile_value = yearly_data["ObservedEntities"].quantile(
            percentile_threshold / 100
        )
        outlier_data = yearly_data[yearly_data["ObservedEntities"] > percentile_value]
        filtered_data = yearly_data[yearly_data["ObservedEntities"] <= percentile_value]

        if len(outlier_data) > 0:
            outlier_years = sorted(outlier_data["Year"].tolist())
            outlier_values = outlier_data.set_index("Year")[
                "ObservedEntities"
            ].to_dict()

            if len(outlier_years) <= 2:
                year_details = [
                    f"{year}({outlier_values[year]:,})" for year in outlier_years
                ]
                outlier_detail = f"outliers {', '.join(year_details)}"
            else:
                outlier_detail = f"{len(outlier_years)} outliers (range: {min(outlier_years)}-{max(outlier_years)})"

            default_kwargs[
                "title"
            ] += f" (excl. >{percentile_threshold:.1f}%ile: {outlier_detail})"

    if len(filtered_data) > 20:
        if window is None:
            window = min(5, len(filtered_data) // 4)

        rolling_mean = (
            pd.Series(filtered_data["ObservedEntities"]).rolling(window=window).mean()
        )

        # Plot original data as scatter points
        ax.scatter(
            filtered_data["Year"],
            filtered_data["ObservedEntities"],
            color=default_kwargs["original_color"],
            alpha=default_kwargs["original_alpha"],
            s=default_kwargs["original_markersize"] ** 2,
            label="Original",
            zorder=2,
        )

        # Plot smoothed trend as line
        ax.plot(
            filtered_data["Year"],
            rolling_mean,
            color=default_kwargs["trend_color"],
            linewidth=default_kwargs["trend_linewidth"],
            alpha=default_kwargs["trend_alpha"],
            label=f"{window}-Year Moving Average",
            zorder=3,
        )

        # Set better y-axis limits
        y_max = filtered_data["ObservedEntities"].quantile(0.95)
        y_min = max(0, filtered_data["ObservedEntities"].min())
        ax.set_ylim(y_min, y_max * 1.05)

        ax.legend(fontsize=9)

        # Improve x-axis resolution
        years = filtered_data["Year"]
        year_range = years.max() - years.min()

        if year_range > 50:
            ax.xaxis.set_major_locator(plt.MultipleLocator(10))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
        elif year_range > 20:
            ax.xaxis.set_major_locator(plt.MultipleLocator(5))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
        else:
            ax.xaxis.set_major_locator(plt.MultipleLocator(2))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

        # Add grid
        ax.grid(True, alpha=0.3, which="major")
        ax.grid(True, alpha=0.1, which="minor")

        # Rotate labels if needed
        if year_range > 30:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    ax.set_title(default_kwargs["title"], fontsize=12, fontweight="bold")
    ax.set_xlabel(default_kwargs["xlabel"])
    ax.set_ylabel(default_kwargs["ylabel"])

    return ax


def plot_publication_volume_context(
    yearly_data: pd.DataFrame,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot publication volume context to explain the trend discrepancy.

    Parameters
    ----------
    yearly_data : pd.DataFrame
        DataFrame with 'Year', 'ObservedEntities', and 'TotalDocuments' columns
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    **kwargs : dict
        Additional plotting parameters

    Returns
    -------
    plt.Axes
        The axes object with the plot

    """
    if ax is None:
        ax = plt.gca()

    default_kwargs = {
        "title": "Publication Volume Over Time",
        "doc_color": "steelblue",
        "rate_color": "orange",
        "alpha": 0.7,
        "remove_outliers": True,
        "outlier_percentile": 99.7,  # New configurable parameter
    }
    default_kwargs.update(kwargs)

    if "TotalDocuments" in yearly_data.columns:
        # Filter out extreme outliers using configurable percentile
        if default_kwargs.get("remove_outliers", True):
            percentile_threshold = default_kwargs.get("outlier_percentile", 99.7)
            doc_percentile = yearly_data["TotalDocuments"].quantile(
                percentile_threshold / 100
            )
            cooc_percentile = yearly_data["ObservedEntities"].quantile(
                percentile_threshold / 100
            )

            doc_outliers = yearly_data[yearly_data["TotalDocuments"] > doc_percentile]
            cooc_outliers = yearly_data[
                yearly_data["ObservedEntities"] > cooc_percentile
            ]

            filtered_data = yearly_data[
                (yearly_data["TotalDocuments"] <= doc_percentile)
                & (yearly_data["ObservedEntities"] <= cooc_percentile)
            ]

            # Add detailed note about filtering
            outliers_removed = len(yearly_data) - len(filtered_data)
            if outliers_removed > 0:
                outlier_details = []
                if len(doc_outliers) > 0:
                    doc_outlier_years = sorted(doc_outliers["Year"].tolist())
                    max_doc_year = doc_outliers.loc[
                        doc_outliers["TotalDocuments"].idxmax(), "Year"
                    ]
                    max_docs = doc_outliers["TotalDocuments"].max()
                    if len(doc_outlier_years) <= 2:
                        outlier_details.append(
                            f"high-doc years {', '.join(map(str, doc_outlier_years))}"
                        )
                    else:
                        outlier_details.append(
                            f"high-doc years (peak: {max_doc_year}={max_docs:,})"
                        )

                if len(cooc_outliers) > 0:
                    cooc_outlier_years = sorted(cooc_outliers["Year"].tolist())
                    if len(cooc_outlier_years) <= 2:
                        outlier_details.append(
                            f"high-cooc years {', '.join(map(str, cooc_outlier_years))}"
                        )
                    else:
                        outlier_details.append(
                            f"{len(cooc_outlier_years)} high-cooc years"
                        )

                default_kwargs[
                    "title"
                ] += f" (excl. >{percentile_threshold:.1f}%ile: {', '.join(outlier_details)})"
        else:
            filtered_data = yearly_data

        # Plot document counts
        ax.bar(
            filtered_data["Year"],
            filtered_data["TotalDocuments"],
            color=default_kwargs["doc_color"],
            alpha=default_kwargs["alpha"],
            label="Total Documents",
        )

        ax.set_ylabel("Total Documents", color=default_kwargs["doc_color"])
        ax.set_xlabel("Year")

        # Add co-occurrence rate on secondary axis
        ax2 = ax.twinx()
        rate = filtered_data["ObservedEntities"] / filtered_data["TotalDocuments"]
        ax2.plot(
            filtered_data["Year"],
            rate,
            color=default_kwargs["rate_color"],
            marker="o",
            linewidth=2,
            markersize=4,
            label="Co-occurrences per Document",
        )
        ax2.set_ylabel(
            "Co-occurrences per Document",
            color=default_kwargs["rate_color"],
        )

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    ax.set_title(default_kwargs["title"], fontsize=12, fontweight="bold")
    return ax
