#!/usr/bin/env python3
"""Plot Orchestrator for Time Series Analysis
This module orchestrates individual plotting functions to create comprehensive visualizations.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from envidis.time_series_analysis.plot_functions import (
    plot_coefficients_table,
    plot_data_quality_table,
    plot_distribution,
    plot_model_summary_table,
    plot_monthly_patterns,
    plot_raw_timeseries,
    plot_recent_trend_bar,
    plot_scatter_documents_vs_cooccurrence,
    plot_time_series_decomposition,
    plot_trend_analysis,
    plot_yearly_aggregated,
    plot_zero_inflation_pie,
)


class PlotOrchestrator:
    """Orchestrates multiple plotting functions to create comprehensive visualizations."""

    def __init__(self, output_dir: str = None):
        """Initialize the plot orchestrator.

        Parameters
        ----------
        output_dir : str, optional
            Directory to save plots

        """
        self.output_dir = output_dir or "./plots"
        os.makedirs(self.output_dir, exist_ok=True)

    def create_comprehensive_plot(
        self,
        data: pd.DataFrame,
        yearly_data: pd.DataFrame = None,
        monthly_data: pd.DataFrame = None,
        model_summary: pd.DataFrame = None,
        coef_table: pd.DataFrame = None,
        layout: tuple[int, int] = (6, 3),  # Updated to 6 rows for additional context
        figsize: tuple[int, int] = (
            8.0,
            24,
        ),  # Changed to A4 width (8 inches) with proportional height
        title_suffix: str = "",
        **kwargs,
    ) -> plt.Figure:
        """Create comprehensive plot with all analysis visualizations.

        Parameters
        ----------
        data : pd.DataFrame
            Main time series data
        yearly_data : pd.DataFrame, optional
            Yearly aggregated data
        monthly_data : pd.DataFrame, optional
            Monthly aggregated data
        model_summary : pd.DataFrame, optional
            Model summary statistics
        coef_table : pd.DataFrame, optional
            Coefficients table
        layout : tuple
            Grid layout for subplots (rows, cols)
        figsize : tuple
            Figure size (width optimized for A4 paper: 8 inches)
        title_suffix : str
            Suffix to add to plot titles (e.g., date range filter info)
        **kwargs : dict
            Additional parameters for individual plots

        Returns
        -------
        plt.Figure
            The complete figure with all subplots

        """
        fig = plt.figure(figsize=figsize)

        # Determine actual date range from data
        start_year = data.index.min().year
        end_year = data.index.max().year
        date_range = f"{start_year}-{end_year}"

        # Use gridspec for better control over subplot layout
        import matplotlib.gridspec as gridspec

        # Adjust spacing for A4 width - plots A and B get full rows
        gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.4, wspace=0.3)

        # Plot A: Raw time series (FULL ROW 1)
        ax1 = fig.add_subplot(gs[0, :])  # Full width of first row
        plot_raw_timeseries(
            data,
            ax=ax1,
            title=f"A. Co-occurrence Counts Over Time ({date_range}){title_suffix}",
            **kwargs.get("raw_timeseries", {}),
        )

        # Plot B: Yearly aggregated data (FULL ROW 2)
        if yearly_data is not None:
            ax2 = fig.add_subplot(gs[1, :])  # Full width of second row
            plot_yearly_aggregated(
                yearly_data,
                ax=ax2,
                title=f"B. Annual Co-occurrence Totals vs. Publication Rate{title_suffix}",
                **kwargs.get("yearly_aggregated", {}),
            )

        # Plot C: Publication volume context (FULL ROW 3)
        if yearly_data is not None and "TotalDocuments" in yearly_data.columns:
            ax3 = fig.add_subplot(gs[2, :])  # Full width of third row
            from envidis.time_series_analysis.plot_functions import (
                plot_publication_volume_context,
            )

            plot_publication_volume_context(
                yearly_data,
                ax=ax3,
                title=f"C. Publication Volume Context{title_suffix}",
            )

        # Remaining plots in 3-column layout starting from row 4
        recent_start = max(2020, end_year - 5)

        # Plot D: Distribution (row 4, col 1)
        ax4 = fig.add_subplot(gs[3, 0])
        plot_distribution(
            data,
            ax=ax4,
            title=f"D. Distribution of Co-occurrence Counts (Log Scale){title_suffix}",
            **kwargs.get("distribution", {}),
        )

        # Plot E: Monthly patterns (row 4, col 2)
        ax5 = fig.add_subplot(gs[3, 1])
        plot_monthly_patterns(
            data,
            ax=ax5,
            title=f"E. Monthly Pattern ({recent_start}-{end_year}){title_suffix}",
            start_year=recent_start,
            **kwargs.get("monthly_patterns", {}),
        )

        # Plot F: Scatter plot (row 4, col 3)
        ax6 = fig.add_subplot(gs[3, 2])
        plot_scatter_documents_vs_cooccurrence(
            data,
            ax=ax6,
            title=f"F. Documents vs Co-occurrences{title_suffix}",
            **kwargs.get("scatter", {}),
        )

        # Plot G: Model summary table (row 5, col 1)
        ax7 = fig.add_subplot(gs[4, 0])
        if model_summary is not None:
            plot_model_summary_table(
                model_summary,
                ax=ax7,
                title=f"G. Model Comparison{title_suffix}",
                **kwargs.get("model_summary", {}),
            )

        # Plot H: Trend analysis (row 5, col 2)
        ax8 = fig.add_subplot(gs[4, 1])
        if yearly_data is not None:
            plot_trend_analysis(
                yearly_data,
                ax=ax8,
                title=f"H. Long-term Trend Analysis{title_suffix}",
                **kwargs.get("trend_analysis", {}),
            )

        # Plot I: Zero inflation pie (row 5, col 3)
        ax9 = fig.add_subplot(gs[4, 2])
        plot_zero_inflation_pie(
            data,
            ax=ax9,
            title=f"I. Zero-Inflation Analysis{title_suffix}",
            **kwargs.get("zero_inflation", {}),
        )

        # Plot J: Recent trend bar (row 6, col 1)
        ax10 = fig.add_subplot(gs[5, 0])
        plot_recent_trend_bar(
            data,
            ax=ax10,
            title=f"J. Recent Trend ({recent_start}-{end_year}){title_suffix}",
            start_year=recent_start,
            **kwargs.get("recent_trend", {}),
        )

        # Plot K: Coefficients table (row 6, col 2)
        ax11 = fig.add_subplot(gs[5, 1])
        if coef_table is not None:
            plot_coefficients_table(
                coef_table,
                ax=ax11,
                title=f"K. Best Model Coefficients{title_suffix}",
                **kwargs.get("coefficients", {}),
            )

        # Plot L: Data quality table (row 6, col 3)
        ax12 = fig.add_subplot(gs[5, 2])
        plot_data_quality_table(
            data,
            ax=ax12,
            title=f"L. Data Quality Summary{title_suffix}",
            **kwargs.get("data_quality", {}),
        )

        # Adjust font sizes for A4 width
        plt.rcParams.update(
            {
                "font.size": 8,  # Reduced from 10
                "axes.titlesize": 9,  # Reduced from 12
                "axes.labelsize": 8,  # Reduced from 10
                "xtick.labelsize": 7,  # Reduced from 9
                "ytick.labelsize": 7,  # Reduced from 9
                "legend.fontsize": 7,  # Reduced from 9
            }
        )

        plt.tight_layout(pad=1.0)  # Reduced padding for tighter layout
        return fig

    def create_custom_layout(
        self,
        plot_configs: list[dict[str, Any]],
        layout: tuple[int, int] = None,
        figsize: tuple[int, int] = (15, 10),
    ) -> plt.Figure:
        """Create custom layout with specified plot configurations.

        Parameters
        ----------
        plot_configs : list of dict
            List of plot configurations, each containing:
            - 'type': plot function name
            - 'data': data for the plot
            - 'kwargs': additional parameters
        layout : tuple, optional
            Grid layout (rows, cols). If None, auto-calculated
        figsize : tuple
            Figure size

        Returns
        -------
        plt.Figure
            The figure with custom layout

        """
        n_plots = len(plot_configs)

        if layout is None:
            # Auto-calculate layout
            cols = int(np.ceil(np.sqrt(n_plots)))
            rows = int(np.ceil(n_plots / cols))
            layout = (rows, cols)

        fig = plt.figure(figsize=figsize)

        # Mapping of plot types to functions
        plot_functions = {
            "raw_timeseries": plot_raw_timeseries,
            "yearly_aggregated": plot_yearly_aggregated,
            "distribution": plot_distribution,
            "monthly_patterns": plot_monthly_patterns,
            "scatter": plot_scatter_documents_vs_cooccurrence,
            "model_summary": plot_model_summary_table,
            "trend_analysis": plot_trend_analysis,
            "zero_inflation": plot_zero_inflation_pie,
            "recent_trend": plot_recent_trend_bar,
            "coefficients": plot_coefficients_table,
            "data_quality": plot_data_quality_table,
            "decomposition": plot_time_series_decomposition,
        }

        for i, config in enumerate(plot_configs, 1):
            ax = plt.subplot(layout[0], layout[1], i)

            plot_type = config["type"]
            plot_data = config["data"]
            plot_kwargs = config.get("kwargs", {})

            if plot_type in plot_functions:
                plot_functions[plot_type](plot_data, ax=ax, **plot_kwargs)
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"Unknown plot type: {plot_type}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

        plt.tight_layout()
        return fig

    def create_focused_analysis(
        self,
        data: pd.DataFrame,
        yearly_data: pd.DataFrame = None,
        focus_type: str = "temporal",
        title_suffix: str = "",
        **kwargs,
    ) -> plt.Figure:
        """Create focused analysis plots for specific aspects.

        Parameters
        ----------
        data : pd.DataFrame
            Main time series data
        yearly_data : pd.DataFrame, optional
            Yearly aggregated data
        focus_type : str
            Type of focused analysis ('temporal', 'distribution', 'trends', 'quality')
        title_suffix : str
            Suffix to add to plot titles
        **kwargs : dict
            Additional parameters

        Returns
        -------
        plt.Figure
            The focused analysis figure

        """
        if focus_type == "temporal":
            return self._create_temporal_focus(
                data,
                yearly_data,
                title_suffix,
                **kwargs,
            )
        elif focus_type == "distribution":
            return self._create_distribution_focus(data, title_suffix, **kwargs)
        elif focus_type == "trends":
            return self._create_trends_focus(data, yearly_data, title_suffix, **kwargs)
        elif focus_type == "quality":
            return self._create_quality_focus(data, title_suffix, **kwargs)
        else:
            msg = f"Unknown focus_type: {focus_type}"
            raise ValueError(msg)

    def _create_temporal_focus(
        self,
        data: pd.DataFrame,
        yearly_data: pd.DataFrame = None,
        title_suffix: str = "",
        **kwargs,
    ) -> plt.Figure:
        """Create temporal-focused analysis."""
        fig = plt.figure(figsize=(15, 10))

        # Raw time series
        ax1 = plt.subplot(2, 2, 1)
        plot_raw_timeseries(data, ax=ax1, title=f"Raw Time Series{title_suffix}")

        # Yearly aggregated
        ax2 = plt.subplot(2, 2, 2)
        if yearly_data is not None:
            plot_yearly_aggregated(
                yearly_data,
                ax=ax2,
                title=f"Yearly Totals{title_suffix}",
            )

        # Monthly patterns
        ax3 = plt.subplot(2, 2, 3)
        plot_monthly_patterns(data, ax=ax3, title=f"Monthly Patterns{title_suffix}")

        # Recent trend
        ax4 = plt.subplot(2, 2, 4)
        plot_recent_trend_bar(data, ax=ax4, title=f"Recent Trends{title_suffix}")

        plt.suptitle(
            f"Temporal Analysis Focus{title_suffix}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    def _create_distribution_focus(
        self,
        data: pd.DataFrame,
        title_suffix: str = "",
        **kwargs,
    ) -> plt.Figure:
        """Create distribution-focused analysis."""
        fig = plt.figure(figsize=(12, 8))

        # Distribution (normal scale)
        ax1 = plt.subplot(2, 2, 1)
        plot_distribution(
            data,
            ax=ax1,
            log_scale=False,
            title=f"Distribution (Normal Scale){title_suffix}",
        )

        # Distribution (log scale)
        ax2 = plt.subplot(2, 2, 2)
        plot_distribution(
            data,
            ax=ax2,
            log_scale=True,
            title=f"Distribution (Log Scale){title_suffix}",
        )

        # Zero inflation
        ax3 = plt.subplot(2, 2, 3)
        plot_zero_inflation_pie(data, ax=ax3, title=f"Zero Inflation{title_suffix}")

        # Scatter plot
        ax4 = plt.subplot(2, 2, 4)
        plot_scatter_documents_vs_cooccurrence(
            data,
            ax=ax4,
            title=f"Documents vs Co-occurrences{title_suffix}",
        )

        plt.suptitle(
            f"Distribution Analysis Focus{title_suffix}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    def _create_trends_focus(
        self,
        data: pd.DataFrame,
        yearly_data: pd.DataFrame = None,
        title_suffix: str = "",
        **kwargs,
    ) -> plt.Figure:
        """Create trends-focused analysis."""
        fig = plt.figure(figsize=(12, 8))

        # Trend analysis
        ax1 = plt.subplot(2, 2, 1)
        if yearly_data is not None:
            plot_trend_analysis(
                yearly_data,
                ax=ax1,
                title=f"Long-term Trends{title_suffix}",
            )

        # Time series decomposition
        ax2 = plt.subplot(2, 2, 2)
        if yearly_data is not None:
            plot_time_series_decomposition(
                yearly_data,
                ax=ax2,
                title=f"Trend Smoothing{title_suffix}",
            )

        # Recent trend
        ax3 = plt.subplot(2, 2, 3)
        plot_recent_trend_bar(data, ax=ax3, title=f"Recent Trends{title_suffix}")

        # Monthly patterns
        ax4 = plt.subplot(2, 2, 4)
        plot_monthly_patterns(data, ax=ax4, title=f"Seasonal Patterns{title_suffix}")

        plt.suptitle(
            f"Trend Analysis Focus{title_suffix}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    def _create_quality_focus(
        self,
        data: pd.DataFrame,
        title_suffix: str = "",
        **kwargs,
    ) -> plt.Figure:
        """Create data quality-focused analysis."""
        fig = plt.figure(figsize=(10, 6))

        # Data quality table
        ax1 = plt.subplot(1, 2, 1)
        plot_data_quality_table(
            data,
            ax=ax1,
            title=f"Data Quality Metrics{title_suffix}",
        )

        # Zero inflation
        ax2 = plt.subplot(1, 2, 2)
        plot_zero_inflation_pie(data, ax=ax2, title=f"Data Completeness{title_suffix}")

        plt.suptitle(
            f"Data Quality Analysis Focus{title_suffix}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    def save_plot(
        self,
        fig: plt.Figure,
        filename: str,
        dpi: int = 300,
        bbox_inches: str = "tight",
        **kwargs,
    ) -> str:
        """Save plot to file.

        Parameters
        ----------
        fig : plt.Figure
            Figure to save
        filename : str
            Filename (without path)
        dpi : int
            Resolution for saving
        bbox_inches : str
            Bounding box setting
        **kwargs : dict
            Additional parameters for savefig

        Returns
        -------
        str
            Full path where the plot was saved

        """
        output_path = os.path.join(self.output_dir, filename)
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        print(f"Plot saved to: {output_path}")
        return output_path

    def create_and_save_comprehensive_plot(
        self,
        data: pd.DataFrame,
        filename: str = None,
        **kwargs,
    ) -> str:
        """Create and save comprehensive plot in one step.

        Parameters
        ----------
        data : pd.DataFrame
            Main time series data
        filename : str, optional
            Custom filename
        **kwargs : dict
            Additional parameters

        Returns
        -------
        str
            Path where plot was saved

        """
        if filename is None:
            filename = "comprehensive_analysis.png"

        # Prepare yearly data if not provided
        yearly_data = kwargs.get("yearly_data")
        if yearly_data is None and "ObservedEntities" in data.columns:
            yearly_data = (
                data.groupby(data.index.year)
                .agg(
                    {
                        "ObservedEntities": "sum",
                        "TotalDocuments": (
                            "sum" if "TotalDocuments" in data.columns else "count"
                        ),
                    },
                )
                .reset_index()
            )
            yearly_data.columns = ["Year"] + list(yearly_data.columns[1:])
            kwargs["yearly_data"] = yearly_data

        fig = self.create_comprehensive_plot(data, **kwargs)
        return self.save_plot(fig, filename)


# Convenience functions for quick plotting
def quick_comprehensive_plot(
    data: pd.DataFrame,
    output_dir: str = "./plots",
    filename: str = "comprehensive_analysis.png",
    **kwargs,
) -> str:
    """Quick function to create and save comprehensive plot.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    output_dir : str
        Output directory
    filename : str
        Output filename
    **kwargs : dict
        Additional parameters

    Returns
    -------
    str
        Path to saved plot

    """
    orchestrator = PlotOrchestrator(output_dir=output_dir)
    return orchestrator.create_and_save_comprehensive_plot(
        data,
        filename=filename,
        **kwargs,
    )


def quick_focused_plot(
    data: pd.DataFrame,
    focus_type: str = "temporal",
    output_dir: str = "./plots",
    filename: str = None,
    **kwargs,
) -> str:
    """Quick function to create and save focused analysis plot.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    focus_type : str
        Type of focus analysis
    output_dir : str
        Output directory
    filename : str, optional
        Output filename
    **kwargs : dict
        Additional parameters

    Returns
    -------
    str
        Path to saved plot

    """
    if filename is None:
        filename = f"{focus_type}_analysis.png"

    orchestrator = PlotOrchestrator(output_dir=output_dir)
    fig = orchestrator.create_focused_analysis(data, focus_type=focus_type, **kwargs)
    return orchestrator.save_plot(fig, filename)
