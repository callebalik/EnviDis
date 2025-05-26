#!/usr/bin/env python3
"""Example Usage of Modular Plotting System
This script demonstrates how to use the new modular plotting system
for time series analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_config import PlotConfig, get_config
from plot_functions import (
    plot_distribution,
    plot_raw_timeseries,
    plot_trend_analysis,
    plot_yearly_aggregated,
)
from plot_orchestrator import (
    PlotOrchestrator,
    quick_comprehensive_plot,
    quick_focused_plot,
)


def load_sample_data():
    """Load or create sample data for demonstration."""
    # Create sample data if real data is not available
    np.random.seed(42)
    dates = pd.date_range("1990-01-01", "2025-01-01", freq="D")

    # Create realistic time series with trend and seasonality
    n_days = len(dates)
    trend = np.linspace(10, 100, n_days)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    noise = np.random.normal(0, 5, n_days)

    # Create co-occurrence counts (non-negative)
    co_counts = np.maximum(0, trend + seasonal + noise)
    co_counts = np.random.poisson(co_counts * 0.1)  # Convert to count data

    # Create document counts
    doc_counts = np.random.poisson(
        np.maximum(1, co_counts * 2 + np.random.normal(50, 10, n_days)),
    )

    data = pd.DataFrame(
        {
            "Date": dates,
            "ObservedEntities": co_counts,
            "TotalDocuments": doc_counts,
        },
    )
    data = data.set_index("Date")

    return data


def example_individual_plots() -> None:
    """Demonstrate individual plotting functions."""
    print("=== Individual Plot Examples ===")

    # Load data
    data = load_sample_data()

    # Create yearly aggregated data
    yearly_data = (
        data.groupby(data.index.year)
        .agg(
            {
                "ObservedEntities": "sum",
                "TotalDocuments": "sum",
            },
        )
        .reset_index()
    )
    yearly_data.columns = ["Year", "ObservedEntities", "TotalDocuments"]

    # Example 1: Raw time series plot
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_raw_timeseries(data, ax=ax, title="Custom Raw Time Series")
    plt.savefig("example_raw_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Individual raw time series plot saved")

    # Example 2: Distribution plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_distribution(data, ax=ax, color="purple", title="Custom Distribution")
    plt.savefig("example_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Individual distribution plot saved")

    # Example 3: Trend analysis with custom parameters
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_trend_analysis(
        yearly_data,
        ax=ax,
        observed_color="darkblue",
        trend_color="orange",
        trend_degree=3,
        title="Custom Trend Analysis (Cubic)",
    )
    plt.savefig("example_trend_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Individual trend analysis plot saved")


def example_orchestrator_usage() -> None:
    """Demonstrate PlotOrchestrator usage."""
    print("\n=== Plot Orchestrator Examples ===")

    # Load data
    data = load_sample_data()
    yearly_data = (
        data.groupby(data.index.year)
        .agg(
            {
                "ObservedEntities": "sum",
                "TotalDocuments": "sum",
            },
        )
        .reset_index()
    )
    yearly_data.columns = ["Year", "ObservedEntities", "TotalDocuments"]

    # Create orchestrator
    orchestrator = PlotOrchestrator(output_dir="./example_plots")

    # Example 1: Comprehensive plot with default settings - now with A and B on separate full rows
    fig = orchestrator.create_comprehensive_plot(data, yearly_data=yearly_data)
    orchestrator.save_plot(fig, "comprehensive_default.png")
    plt.close()
    print("✓ Comprehensive plot (A and B on full rows) saved")

    # Example 2: Comprehensive plot with custom styling
    custom_kwargs = {
        "raw_timeseries": {"color": "darkred", "alpha": 0.8},
        "distribution": {"color": "forestgreen", "bins": 30},
        "trend_analysis": {"trend_color": "purple", "trend_degree": 3},
    }
    fig = orchestrator.create_comprehensive_plot(
        data,
        yearly_data=yearly_data,
        **custom_kwargs,
    )
    orchestrator.save_plot(fig, "comprehensive_custom.png")
    plt.close()
    print("✓ Comprehensive plot (custom styling) saved")

    # Example 3: Focused analysis - temporal
    fig = orchestrator.create_focused_analysis(
        data,
        yearly_data=yearly_data,
        focus_type="temporal",
    )
    orchestrator.save_plot(fig, "focused_temporal.png")
    plt.close()
    print("✓ Focused temporal analysis saved")

    # Example 4: Focused analysis - distribution
    fig = orchestrator.create_focused_analysis(data, focus_type="distribution")
    orchestrator.save_plot(fig, "focused_distribution.png")
    plt.close()
    print("✓ Focused distribution analysis saved")

    # Example 5: Custom layout
    plot_configs = [
        {
            "type": "raw_timeseries",
            "data": data,
            "kwargs": {"title": "Custom Layout - Raw Data"},
        },
        {
            "type": "yearly_aggregated",
            "data": yearly_data,
            "kwargs": {"title": "Custom Layout - Yearly"},
        },
        {
            "type": "distribution",
            "data": data,
            "kwargs": {"title": "Custom Layout - Distribution", "color": "red"},
        },
        {
            "type": "trend_analysis",
            "data": yearly_data,
            "kwargs": {"title": "Custom Layout - Trends"},
        },
    ]
    fig = orchestrator.create_custom_layout(
        plot_configs,
        layout=(2, 2),
        figsize=(12, 8),
    )
    orchestrator.save_plot(fig, "custom_layout.png")
    plt.close()
    print("✓ Custom layout plot saved")


def example_configuration_usage() -> None:
    """Demonstrate configuration system usage."""
    print("\n=== Configuration System Examples ===")

    # Load data
    data = load_sample_data()

    # Example 1: Using predefined configurations
    configs = ["default", "publication", "presentation", "quick"]

    for config_name in configs:
        config = get_config(config_name)
        config.set_matplotlib_style()  # Apply to matplotlib

        fig, ax = plt.subplots(figsize=config.params["figure"]["figsize"])
        plot_config = config.get_plot_config("raw_timeseries")
        plot_raw_timeseries(
            data,
            ax=ax,
            title=f"Time Series ({config_name.title()} Config)",
            **plot_config,
        )
        plt.savefig(
            f"config_example_{config_name}.png",
            dpi=config.params["figure"]["dpi"],
            bbox_inches="tight",
        )
        plt.close()
        print(f"✓ {config_name.title()} configuration example saved")

    # Example 2: Custom configuration
    custom_config = PlotConfig(
        color_scheme="vibrant",
        custom_params={
            "figure": {"figsize": (14, 8), "dpi": 200},
            "fonts": {"title_size": 16, "label_size": 12},
            "lines": {"linewidth": 2, "alpha": 0.9},
        },
    )

    custom_config.set_matplotlib_style()
    fig, ax = plt.subplots(figsize=custom_config.params["figure"]["figsize"])
    plot_config = custom_config.get_plot_config("distribution")
    plot_distribution(data, ax=ax, title="Custom Configuration Example", **plot_config)
    plt.savefig(
        "config_example_custom.png",
        dpi=custom_config.params["figure"]["dpi"],
        bbox_inches="tight",
    )
    plt.close()
    print("✓ Custom configuration example saved")


def example_quick_functions() -> None:
    """Demonstrate quick plotting functions."""
    print("\n=== Quick Functions Examples ===")

    # Load data
    data = load_sample_data()

    # Quick comprehensive plot
    path = quick_comprehensive_plot(
        data,
        output_dir="./quick_plots",
        filename="quick_comprehensive.png",
    )
    print(f"✓ Quick comprehensive plot saved to: {path}")

    # Quick focused plots
    focus_types = ["temporal", "distribution", "trends"]
    for focus_type in focus_types:
        path = quick_focused_plot(
            data,
            focus_type=focus_type,
            output_dir="./quick_plots",
            filename=f"quick_{focus_type}.png",
        )
        print(f"✓ Quick {focus_type} plot saved to: {path}")


def example_advanced_customization() -> None:
    """Demonstrate advanced customization options."""
    print("\n=== Advanced Customization Examples ===")

    # Load data
    data = load_sample_data()

    # Create custom color scheme
    custom_colors = {
        "primary": "#2E86AB",  # Blue
        "secondary": "#A23B72",  # Purple
        "tertiary": "#F18F01",  # Orange
        "quaternary": "#C73E1D",  # Red
        "accent": "#592E83",  # Dark purple
    }

    config = PlotConfig()
    config.create_custom_color_scheme(custom_colors)

    # Create orchestrator with custom config
    orchestrator = PlotOrchestrator(output_dir="./advanced_plots")

    # Create plot with completely custom styling
    custom_styling = {
        "raw_timeseries": {
            "color": config.get_color("primary"),
            "alpha": 0.8,
            "linewidth": 1.5,
            "title": "Advanced Customization - Time Series",
        },
        "distribution": {
            "color": config.get_color("secondary"),
            "bins": 40,
            "alpha": 0.75,
            "title": "Advanced Customization - Distribution",
        },
        "monthly_patterns": {
            "color": config.get_color("tertiary"),
            "alpha": 0.9,
            "title": "Advanced Customization - Monthly Patterns",
        },
        "recent_trend": {
            "color": config.get_color("quaternary"),
            "alpha": 0.8,
            "title": "Advanced Customization - Recent Trends",
        },
    }

    # Create focused temporal analysis with custom styling
    fig = orchestrator.create_focused_analysis(
        data,
        focus_type="temporal",
        **custom_styling,
    )
    orchestrator.save_plot(fig, "advanced_custom_styling.png")
    plt.close()
    print("✓ Advanced customization example saved")


def main() -> None:
    """Run all examples."""
    print("Running Modular Plotting System Examples")
    print("=" * 50)

    try:
        example_individual_plots()
        example_orchestrator_usage()
        example_configuration_usage()
        example_quick_functions()
        example_advanced_customization()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("Check the following directories for output files:")
        print("- Current directory: individual plot examples")
        print("- ./example_plots/: orchestrator examples")
        print("- ./quick_plots/: quick function examples")
        print("- ./advanced_plots/: advanced customization examples")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have the required dependencies installed.")


if __name__ == "__main__":
    main()
