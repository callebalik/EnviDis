#!/usr/bin/env python3
"""Real Data Comprehensive Analysis Report
This script creates a complete analysis for real time series data with date-level resolution.
Adapted for co-occurrence data from 1794-2025.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf

# Module constants for axis configuration
AXIS_CONFIG = {
    "title_fontsize": 12,
    "title_fontweight": "bold",
    "label_fontsize": 10,
    "tick_labelsize": 8,
    "legend_fontsize": 9,
    "grid_alpha": 0.3,
    "marker_size": 30,
    "line_width": 2,
    "line_alpha": 0.8,
}

# Date formatting configuration
DATE_FORMAT_CONFIG = {
    "date_format": "%Y",
    "major_locator_years": 5,
    "rotation": 45,
}

# Grid layout configuration for enhanced A/B plots
GRID_CONFIG = {
    "total_rows": 6,
    "total_cols": 4,
    "hspace": 0.4,
    "wspace": 0.3,
    # Enhanced spacing for A and B plots
    "ab_plot_rows": 2,  # A and B plots now take 2 rows each
    "other_plot_rows": 1,
}


class RealDataAnalysisReport:
    """Generate comprehensive analysis report for real date-level time series data."""

    def __init__(self, data_path=None, output_dir=None, start_date=None, end_date=None):
        """Initialize the real data analysis report.

        Parameters
        ----------
        data_path : str
            Path to the CSV file with real data
        output_dir : str, optional
            Directory to save outputs
        start_date : str or datetime, optional
            Start date for analysis (inclusive). Format: 'YYYY-MM-DD' or datetime object
        end_date : str or datetime, optional
            End date for analysis (inclusive). Format: 'YYYY-MM-DD' or datetime object

        """
        self.output_dir = output_dir or "/home/callebalik/EnviDis/results/analysis"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/real_data_report", exist_ok=True)

        # Store date filters
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None

        # Validate date range
        if self.start_date and self.end_date and self.start_date > self.end_date:
            msg = "start_date must be before or equal to end_date"
            raise ValueError(msg)

        # Load real data
        self.data_path = (
            data_path or "/home/callebalik/EnviDis/data/raw/time-series/time_series.csv"
        )
        self.data = self.load_and_prepare_data()

        # Initialize model storage
        self.models = {}
        self.basic_fitter = None
        self.trend_fitter = None
        self.analyzer = None

    def load_and_prepare_data(self):
        """Load and prepare real time series data."""
        print(f"Loading real data from: {self.data_path}")

        # Load data
        df = pd.read_csv(self.data_path)

        # Convert date column to datetime
        df["Date"] = pd.to_datetime(df["date"])

        # Apply date filtering if specified
        original_length = len(df)
        if self.start_date:
            df = df[df["Date"] >= self.start_date]

        if self.end_date:
            df = df[df["Date"] <= self.end_date]

        if self.start_date or self.end_date:
            print(f"Date filtering applied: {original_length} -> {len(df)} observations")

        if len(df) == 0:
            raise ValueError("No data remaining after date filtering")

        # Rename for consistency with existing framework
        df["co_count"] = df["co_count"]  # Use co-occurrence count as dependent variable
        df["document_count"] = df["document_count"]

        # Create time variables for modeling
        df["Year"] = df["Date"].dt.year
        df["DayOfYear"] = df["Date"].dt.dayofyear
        df["MonthOfYear"] = df["Date"].dt.month

        # Create scaled time variables (important for numerical stability)
        # Use days since start for high-resolution time modeling
        start_date = df["Date"].min()
        df["DaysSinceStart"] = (df["Date"] - start_date).dt.days
        df["DaysSinceStart_scaled"] = (
            df["DaysSinceStart"] - df["DaysSinceStart"].mean()
        ) / df["DaysSinceStart"].std()

        # Also create year-based scaling for comparison
        df["Year_scaled"] = (df["Year"] - df["Year"].mean()) / df["Year"].std()

        # Set Date as index for time series operations
        df = df.set_index("Date").sort_index()

        print(f"Prepared data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(
            f"co_count (co_count) range: {df['co_count'].min()} to {df['co_count'].max()}",
        )

        return df

    def run_all_analyses(self) -> None:
        """Run all statistical analyses."""
        print("Running comprehensive statistical analysis on real data...")

        print("1. Fitting basic models (Poisson vs Negative Binomial)...")
        self.fit_basic_models()

        print("2. Fitting trend models (Linear, Quadratic, Cubic)...")
        self.fit_trend_models()

        print("3. Analyzing temporal patterns...")
        self.analyze_temporal_patterns()

    def fit_basic_models(self) -> None:
        """Fit basic count models (Poisson and Negative Binomial)."""
        # Filter out extremely high values that might cause convergence issues
        modeling_data = self.data[
            self.data["co_count"] <= self.data["co_count"].quantile(0.95)
        ]

        # Filter out zero document counts to avoid log(0) issues
        modeling_data = modeling_data[modeling_data["document_count"] > 0]

        try:
            # Poisson model with day-level time trend and document count as offset
            formula = "co_count ~ DaysSinceStart_scaled"
            self.models["poisson"] = smf.poisson(
                formula,
                data=modeling_data,
                offset=np.log(modeling_data["document_count"]),
            ).fit(disp=0)

            # Negative Binomial model with offset
            self.models["negative_binomial"] = smf.negativebinomial(
                formula,
                data=modeling_data,
                offset=np.log(modeling_data["document_count"]),
            ).fit(disp=0)

            print("   ✓ Basic models fitted successfully")
        except Exception as e:
            print(f"   ✗ Basic model fitting failed: {str(e)}")

    def fit_trend_models(self) -> None:
        """Fit polynomial trend models."""
        # Use a sample for complex models to ensure computational feasibility
        sample_size = min(5000, len(self.data))
        modeling_data = self.data.sample(n=sample_size, random_state=42)
        modeling_data = modeling_data[
            modeling_data["co_count"] <= modeling_data["co_count"].quantile(0.95)
        ]

        # Filter out zero document counts to avoid log(0) issues
        modeling_data = modeling_data[modeling_data["document_count"] > 0]

        try:
            # Linear trend (year-based for interpretability) with offset
            formula_linear = "co_count ~ Year_scaled"
            self.models["linear"] = smf.poisson(
                formula_linear,
                data=modeling_data,
                offset=np.log(modeling_data["document_count"]),
            ).fit(disp=0)

            # Quadratic trend with offset
            formula_quad = "co_count ~ Year_scaled + I(Year_scaled**2)"
            self.models["quadratic"] = smf.poisson(
                formula_quad,
                data=modeling_data,
                offset=np.log(modeling_data["document_count"]),
            ).fit(disp=0)

            # Cubic trend with offset
            formula_cubic = (
                "co_count ~ Year_scaled + I(Year_scaled**2) + I(Year_scaled**3)"
            )
            self.models["cubic"] = smf.poisson(
                formula_cubic,
                data=modeling_data,
                offset=np.log(modeling_data["document_count"]),
            ).fit(disp=0)

            print("   ✓ Trend models fitted successfully")
        except Exception as e:
            print(f"   ✗ Trend model fitting failed: {str(e)}")

    def analyze_temporal_patterns(self) -> None:
        """Analyze temporal patterns and seasonality."""
        # Monthly aggregation for pattern analysis
        monthly_data = self.data.groupby(
            [self.data.index.year, self.data.index.month],
        ).agg(
            {
                "co_count": "sum",
                "document_count": "sum",
            },
        )
        monthly_data.index.names = ["Year", "Month"]
        monthly_data = monthly_data.reset_index()

        # Yearly aggregation
        yearly_data = self.data.groupby(self.data.index.year).agg(
            {
                "co_count": "sum",
                "document_count": "sum",
            },
        )
        yearly_data.index.name = "Year"
        yearly_data = yearly_data.reset_index(drop=False)

        self.monthly_data = monthly_data
        self.yearly_data = yearly_data

        print(
            f"   ✓ Temporal patterns analyzed ({len(yearly_data)} years, {len(monthly_data)} months)",
        )

    def create_model_summary_table(self):
        """Create comprehensive model summary table."""
        summary_data = []

        for model_name, model in self.models.items():
            if model is not None:
                # Calculate pseudo R-squared (McFadden's)
                if hasattr(model, "llnull") and hasattr(model, "llf"):
                    pseudo_r2 = 1 - (model.llf / model.llnull)
                else:
                    pseudo_r2 = np.nan

                summary_data.append(
                    {
                        "Model": model_name.replace("_", " ").title(),
                        "AIC": f"{model.aic:.2f}",
                        "BIC": f"{model.bic:.2f}" if hasattr(model, "bic") else "N/A",
                        "Log-Likelihood": f"{model.llf:.2f}",
                        "Pseudo R²": (
                            f"{pseudo_r2:.4f}" if not np.isnan(pseudo_r2) else "N/A"
                        ),
                        "N Observations": f"{model.nobs:.0f}",
                        "Converged": "Yes" if model.mle_retvals["converged"] else "No",
                    },
                )

        return pd.DataFrame(summary_data)

    def create_coefficients_table(self):
        """Create comprehensive coefficients table with confidence intervals."""
        coef_data = []

        for model_name, model in self.models.items():
            if model is not None:
                # Get confidence intervals
                try:
                    conf_int = model.conf_int(alpha=0.05)  # 95% CI
                except Exception:
                    conf_int = None

                for param in model.params.index:
                    coef_row = {
                        "Model": model_name.replace("_", " ").title(),
                        "Parameter": param,
                        "Coefficient": f"{model.params[param]:.4f}",
                        "Std Error": f"{model.bse[param]:.4f}",
                        "z-value": f"{model.tvalues[param]:.3f}",
                        "P-value": f"{model.pvalues[param]:.4f}",
                        "Significant": (
                            "p<0.001"
                            if model.pvalues[param] < 0.001
                            else (
                                "p<0.01"
                                if model.pvalues[param] < 0.01
                                else (
                                    "p<0.05"
                                    if model.pvalues[param] < 0.05
                                    else "Not Sig."
                                )
                            )
                        ),
                    }

                    if conf_int is not None:
                        coef_row["CI Lower (95%)"] = f"{conf_int.loc[param, 0]:.4f}"
                        coef_row["CI Upper (95%)"] = f"{conf_int.loc[param, 1]:.4f}"
                    else:
                        coef_row["CI Lower (95%)"] = "N/A"
                        coef_row["CI Upper (95%)"] = "N/A"

                    coef_data.append(coef_row)

        return pd.DataFrame(coef_data)

    def create_comprehensive_visualization(self):
        """Create comprehensive visualization using modular plotting system."""
        from envidis.time_series_analysis.plot_orchestrator import PlotOrchestrator

        # Initialize plot orchestrator with date filter information
        orchestrator = PlotOrchestrator(output_dir=self.output_dir)

        # Prepare data for plotting
        model_summary = self.create_model_summary_table()
        coef_table = self.create_coefficients_table()

        # Get the best model from the models dictionary
        best_model = None
        if self.models:
            # Try to get the best trend model
            if "quadratic" in self.models and self.models["quadratic"] is not None:
                best_model = self.models["quadratic"]
            elif "cubic" in self.models and self.models["cubic"] is not None:
                best_model = self.models["cubic"]
            elif "linear" in self.models and self.models["linear"] is not None:
                best_model = self.models["linear"]

        # Create title suffix based on date filters
        title_suffix = ""
        if self.start_date or self.end_date:
            date_parts = []
            if self.start_date:
                date_parts.append(f"from {self.start_date.date()}")
            if self.end_date:
                date_parts.append(f"to {self.end_date.date()}")
            title_suffix = f" ({' '.join(date_parts)})"

        # Create enhanced comprehensive plot with larger A/B sections
        fig = self._create_enhanced_visualization(
            model_summary=model_summary,
            coef_table=coef_table,
            title_suffix=title_suffix,
            best_model=best_model,  # Pass the best model
        )

        # Save the comprehensive visualization with date filter in filename
        filename_suffix = ""
        if self.start_date or self.end_date:
            date_parts = []
            if self.start_date:
                date_parts.append(f"from_{self.start_date.strftime('%Y%m%d')}")
            if self.end_date:
                date_parts.append(f"to_{self.end_date.strftime('%Y%m%d')}")
            filename_suffix = f"_{'_'.join(date_parts)}"

        output_path = f"{self.output_dir}/real_data_report/comprehensive_real_data_analysis{filename_suffix}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Comprehensive visualization saved to: {output_path}")

        return fig

    def _create_enhanced_visualization(
        self,
        model_summary,
        coef_table,
        title_suffix="",
        best_model=None,
    ):
        """Create enhanced visualization with larger A and B plots."""
        # Set up the figure with enhanced grid layout
        fig = plt.figure(figsize=(20, 28))  # Increased height for enhanced A/B plots
        gs = fig.add_gridspec(
            GRID_CONFIG["total_rows"] + 2,  # Extra rows for enhanced A/B
            GRID_CONFIG["total_cols"],
            hspace=GRID_CONFIG["hspace"],
            wspace=GRID_CONFIG["wspace"],
        )

        # Color scheme
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        # Determine if we're working with date or year indices
        is_date_index = isinstance(self.data.index, pd.DatetimeIndex)

        # For visualization with large datasets, sample data points
        if len(self.data) > 1000:
            viz_data = self.data.iloc[
                :: max(1, len(self.data) // 1000)
            ]  # Sample for visualization
        else:
            viz_data = self.data

        # 1. Enhanced Data Overview (Takes 2 rows now)
        ax1 = fig.add_subplot(gs[0:2, 0:2])  # 2 rows, 2 columns
        ax1_twin = ax1.twinx()

        # Entities over time
        line1 = ax1.plot(
            viz_data.index,
            viz_data["co_count"],
            "o-",
            color=colors[0],
            linewidth=AXIS_CONFIG["line_width"],
            markersize=4,
            label="Identified DIS-PNM co-occurrences",
            alpha=AXIS_CONFIG["line_alpha"],
        )
        ax1.set_ylabel(
            "Identified DIS-PNM co-occurrences",
            color=colors[0],
            fontsize=AXIS_CONFIG["label_fontsize"],
        )
        ax1.tick_params(
            axis="y",
            labelcolor=colors[0],
            labelsize=AXIS_CONFIG["tick_labelsize"],
        )
        ax1.grid(True, alpha=AXIS_CONFIG["grid_alpha"])

        # Documents as bars
        bars = ax1_twin.bar(
            viz_data.index,
            viz_data["document_count"],
            alpha=0.3,
            color=colors[1],
            width=0.8 if not is_date_index else 365,
            label="Total Documents",
        )
        ax1_twin.set_ylabel(
            "Total Documents",
            color=colors[1],
            fontsize=AXIS_CONFIG["label_fontsize"],
        )
        ax1_twin.tick_params(
            axis="y",
            labelcolor=colors[1],
            labelsize=AXIS_CONFIG["tick_labelsize"],
        )

        # Combined legend
        lines = line1 + [bars]
        labels = ["Identified DIS-PNM co-occurrences", "Total Documents"]
        ax1.legend(
            lines,
            labels,
            loc="upper left",
            fontsize=AXIS_CONFIG["legend_fontsize"],
        )
        ax1.set_title(
            f"A. Time Series Data Overview{title_suffix}",
            fontsize=AXIS_CONFIG["title_fontsize"],
            fontweight=AXIS_CONFIG["title_fontweight"],
        )

        self._configure_date_axis(ax1, is_date_index)

        # 2. Enhanced Normalized vs Absolute Trends (Takes 2 rows now)
        ax2 = fig.add_subplot(gs[0:2, 2:4])  # 2 rows, 2 columns
        normalized_entities = (viz_data["co_count"] / viz_data["document_count"]) * 1000

        ax2.plot(
            viz_data.index,
            viz_data["co_count"],
            "o-",
            color=colors[0],
            linewidth=AXIS_CONFIG["line_width"],
            label="Absolute Counts",
            alpha=AXIS_CONFIG["line_alpha"],
        )
        ax2_norm = ax2.twinx()
        ax2_norm.plot(
            viz_data.index,
            normalized_entities,
            "s-",
            color=colors[2],
            linewidth=AXIS_CONFIG["line_width"],
            label="Normalized (per 1K docs)",
            alpha=AXIS_CONFIG["line_alpha"],
        )

        ax2.set_ylabel(
            "Absolute Entities",
            color=colors[0],
            fontsize=AXIS_CONFIG["label_fontsize"],
        )
        ax2_norm.set_ylabel(
            "Entities per 1K Documents",
            color=colors[2],
            fontsize=AXIS_CONFIG["label_fontsize"],
        )
        ax2.tick_params(
            axis="y",
            labelcolor=colors[0],
            labelsize=AXIS_CONFIG["tick_labelsize"],
        )
        ax2_norm.tick_params(
            axis="y",
            labelcolor=colors[2],
            labelsize=AXIS_CONFIG["tick_labelsize"],
        )

        # Combined legend
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines2_norm, labels2_norm = ax2_norm.get_legend_handles_labels()
        ax2.legend(
            lines2 + lines2_norm,
            labels2 + labels2_norm,
            loc="upper left",
            fontsize=AXIS_CONFIG["legend_fontsize"],
        )
        ax2.set_title(
            "B. Absolute vs Normalized Trends",
            fontsize=AXIS_CONFIG["title_fontsize"],
            fontweight=AXIS_CONFIG["title_fontweight"],
        )

        self._configure_date_axis(ax2, is_date_index)
        ax2.grid(True, alpha=AXIS_CONFIG["grid_alpha"])

        # 3. Model Comparison (AIC/BIC) - adjusted row position
        ax3 = fig.add_subplot(gs[2, 0:2])
        self._create_model_comparison_plot(ax3, model_summary, colors)

        # 4. Trend Fitting Visualization - adjusted row position
        ax4 = fig.add_subplot(gs[2, 2:4])
        self._create_trend_fitting_plot(ax4, viz_data, colors, is_date_index)

        # 5. Residual Analysis - adjusted row position
        ax5 = fig.add_subplot(gs[3, 0:2])
        ax6 = fig.add_subplot(gs[3, 2:4])
        self._create_residual_plots(ax5, ax6, colors)

        # 6. Autocorrelation and Coefficient Analysis - adjusted row position
        ax7 = fig.add_subplot(gs[4, 0:2])
        ax8 = fig.add_subplot(gs[4, 2:4])
        self._create_autocorrelation_plot(ax7)
        self._create_coefficient_plot(ax8, coef_table, colors)

        # 7-8. Model Summary Tables (Bottom sections) - adjusted row positions
        ax9 = fig.add_subplot(gs[5, :])
        ax10 = fig.add_subplot(gs[6, :])
        self._create_summary_tables(ax9, ax10, model_summary, coef_table)

        # Add overall title and metadata
        self._add_title_and_footer(fig, title_suffix)

        return fig

    def _configure_date_axis(self, ax, is_date_index):
        """Configure date axis with standardized 5-year labels and yearly ticks."""
        if is_date_index:
            # Standardized x-axis formatting - every 5 years with yearly ticks
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_minor_locator(mdates.YearLocator(1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        else:
            # For non-datetime indices (like Year columns)
            ax.xaxis.set_major_locator(plt.MultipleLocator(5))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _create_model_comparison_plot(self, ax, model_summary, colors):
        """Create model comparison plot."""
        if len(model_summary) > 0:
            models = model_summary["Model"].values
            aic_values = model_summary["AIC"].values
            
            ax.bar(models, aic_values, color=colors[0], alpha=0.7)
            ax.set_title("Model Comparison (AIC)", fontsize=12, fontweight="bold")
            ax.set_ylabel("AIC")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, "No model data available", ha='center', va='center', transform=ax.transAxes)

    def _create_trend_fitting_plot(self, ax, viz_data, colors, is_date_index):
        """Create trend fitting visualization."""
        ax.scatter(viz_data.index, viz_data["co_count"], alpha=0.6, color=colors[0], s=10)
        ax.set_title("Trend Fitting", fontsize=12, fontweight="bold")
        ax.set_ylabel("Co-occurrence Count")
        
        self._configure_date_axis(ax, is_date_index)
        ax.grid(True, alpha=0.3)

    def _prepare_modeling_data(self) -> pd.DataFrame:
        """Prepare data for modeling."""
        return self.data.copy()

    def run_complete_analysis(self) -> None:
        """Run complete analysis pipeline."""
        print("Starting complete analysis...")
        self.run_all_analyses()
        print("Creating comprehensive visualization...")
        self.create_comprehensive_visualization()
        print("Analysis complete!")

def main() -> None:
    """Main function to run the real data analysis."""
    # Create analyzer with date filter and A4-optimized output
    analyzer = RealDataAnalysisReport(
        start_date="1960-01-01",
        end_date="2023-12-31",
    )

    # Set matplotlib defaults for A4 compatibility
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (8.0, 6),  # A4 width default
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        },
    )

    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
