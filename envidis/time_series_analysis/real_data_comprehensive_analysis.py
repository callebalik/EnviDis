#!/usr/bin/env python3
"""Real Data Comprehensive Analysis Report Generator
This script creates a complete analysis for real time series data with date-level resolution.
Adapted for co-occurrence data from 1794-2025.
"""

import os
import sys
import warnings
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf


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
        df["Date"] = pd.to_datetime(df["pubdate_resolved"])

        # Apply date filtering if specified
        original_length = len(df)
        if self.start_date:
            df = df[df["Date"] >= self.start_date]
            print(
                f"Filtered data from {self.start_date.date()}: {len(df)} records remaining",
            )

        if self.end_date:
            df = df[df["Date"] <= self.end_date]
            print(
                f"Filtered data to {self.end_date.date()}: {len(df)} records remaining",
            )

        if self.start_date or self.end_date:
            print(
                f"Date filtering reduced dataset from {original_length:,} to {len(df):,} records",
            )

        if len(df) == 0:
            msg = "No data remains after applying date filters"
            raise ValueError(msg)

        # Rename for consistency with existing framework
        df["ObservedEntities"] = df[
            "co_count"
        ]  # Use co-occurrence count as dependent variable
        df["TotalDocuments"] = df["document_count"]

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
            f"ObservedEntities (co_count) range: {df['ObservedEntities'].min()} to {df['ObservedEntities'].max()}",
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
            self.data["ObservedEntities"]
            <= self.data["ObservedEntities"].quantile(0.95)
        ]

        # Filter out zero document counts to avoid log(0) issues
        modeling_data = modeling_data[modeling_data["TotalDocuments"] > 0]

        try:
            # Poisson model with day-level time trend and document count as offset
            formula = "ObservedEntities ~ DaysSinceStart_scaled"
            self.models["poisson"] = smf.poisson(
                formula,
                data=modeling_data,
                offset=np.log(modeling_data["TotalDocuments"]),
            ).fit(disp=0)

            # Negative Binomial model with offset
            self.models["negative_binomial"] = smf.negativebinomial(
                formula,
                data=modeling_data,
                offset=np.log(modeling_data["TotalDocuments"]),
            ).fit(disp=0)

            print(
                f"   ✓ Basic models fitted on {len(modeling_data)} observations (using document count as offset)",
            )
        except Exception as e:
            print(f"   ⚠ Basic model fitting encountered issues: {e}")

    def fit_trend_models(self) -> None:
        """Fit polynomial trend models."""
        # Use a sample for complex models to ensure computational feasibility
        sample_size = min(5000, len(self.data))
        modeling_data = self.data.sample(n=sample_size, random_state=42)
        modeling_data = modeling_data[
            modeling_data["ObservedEntities"]
            <= modeling_data["ObservedEntities"].quantile(0.95)
        ]

        # Filter out zero document counts to avoid log(0) issues
        modeling_data = modeling_data[modeling_data["TotalDocuments"] > 0]

        try:
            # Linear trend (year-based for interpretability) with offset
            formula_linear = "ObservedEntities ~ Year_scaled"
            self.models["linear"] = smf.poisson(
                formula_linear,
                data=modeling_data,
                offset=np.log(modeling_data["TotalDocuments"]),
            ).fit(disp=0)

            # Quadratic trend with offset
            formula_quad = "ObservedEntities ~ Year_scaled + I(Year_scaled**2)"
            self.models["quadratic"] = smf.poisson(
                formula_quad,
                data=modeling_data,
                offset=np.log(modeling_data["TotalDocuments"]),
            ).fit(disp=0)

            # Cubic trend with offset
            formula_cubic = (
                "ObservedEntities ~ Year_scaled + I(Year_scaled**2) + I(Year_scaled**3)"
            )
            self.models["cubic"] = smf.poisson(
                formula_cubic,
                data=modeling_data,
                offset=np.log(modeling_data["TotalDocuments"]),
            ).fit(disp=0)

            print(
                f"   ✓ Trend models fitted on {len(modeling_data)} observations (using document count as offset)",
            )
        except Exception as e:
            print(f"   ⚠ Trend model fitting encountered issues: {e}")

    def analyze_temporal_patterns(self) -> None:
        """Analyze temporal patterns and seasonality."""
        # Monthly aggregation for pattern analysis
        monthly_data = self.data.groupby(
            [self.data.index.year, self.data.index.month],
        ).agg(
            {
                "ObservedEntities": "sum",
                "TotalDocuments": "sum",
            },
        )
        monthly_data.index.names = ["Year", "Month"]
        monthly_data = monthly_data.reset_index()

        # Yearly aggregation
        yearly_data = self.data.groupby(self.data.index.year).agg(
            {
                "ObservedEntities": "sum",
                "TotalDocuments": "sum",
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

        # Create title suffix based on date filters
        title_suffix = ""
        if self.start_date or self.end_date:
            date_parts = []
            if self.start_date:
                date_parts.append(f"from {self.start_date.date()}")
            if self.end_date:
                date_parts.append(f"to {self.end_date.date()}")
            title_suffix = f" ({' '.join(date_parts)})"

        # Create comprehensive plot using the orchestrator
        fig = orchestrator.create_comprehensive_plot(
            data=self.data,
            yearly_data=getattr(self, "yearly_data", None),
            monthly_data=getattr(self, "monthly_data", None),
            model_summary=model_summary if len(model_summary) > 0 else None,
            coef_table=coef_table if len(coef_table) > 0 else None,
            title_suffix=title_suffix,
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

    def create_focused_plots(self, focus_types=None):
        """Create focused analysis plots using modular plotting system."""
        from .plot_orchestrator import PlotOrchestrator

        if focus_types is None:
            focus_types = ["temporal", "distribution", "trends", "quality"]

        # Initialize plot orchestrator
        orchestrator = PlotOrchestrator(output_dir=self.output_dir)

        # Create title suffix based on date filters
        title_suffix = ""
        filename_suffix = ""
        if self.start_date or self.end_date:
            date_parts = []
            filename_parts = []
            if self.start_date:
                date_parts.append(f"from {self.start_date.date()}")
                filename_parts.append(f"from_{self.start_date.strftime('%Y%m%d')}")
            if self.end_date:
                date_parts.append(f"to {self.end_date.date()}")
                filename_parts.append(f"to_{self.end_date.strftime('%Y%m%d')}")
            title_suffix = f" ({' '.join(date_parts)})"
            filename_suffix = f"_{'_'.join(filename_parts)}"

        created_plots = []
        for focus_type in focus_types:
            try:
                fig = orchestrator.create_focused_analysis(
                    data=self.data,
                    yearly_data=getattr(self, "yearly_data", None),
                    focus_type=focus_type,
                    title_suffix=title_suffix,
                )

                output_path = f"{self.output_dir}/real_data_report/focused_{focus_type}_analysis{filename_suffix}.png"
                fig.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

                created_plots.append(output_path)
                print(f"Focused {focus_type} analysis saved to: {output_path}")

            except Exception as e:
                print(f"Error creating {focus_type} focused plot: {e}")

        return created_plots

    def generate_text_summary(self):
        """Generate comprehensive text summary of the analysis."""
        summary = []
        summary.append("=" * 80)
        summary.append("COMPREHENSIVE REAL DATA ANALYSIS SUMMARY")
        summary.append("=" * 80)
        summary.append("")

        # Data overview
        summary.append("📊 DATA OVERVIEW")
        summary.append("-" * 40)
        summary.append(f"• Total observations: {len(self.data):,}")
        summary.append(
            f"• Date range: {self.data.index.min().date()} to {self.data.index.max().date()}",
        )

        # Add filter information if applicable
        if self.start_date or self.end_date:
            summary.append("• Applied date filters:")
            if self.start_date:
                summary.append(f"  - Start date: {self.start_date.date()}")
            if self.end_date:
                summary.append(f"  - End date: {self.end_date.date()}")

        summary.append(f"• Years covered: {self.data.index.year.nunique()}")
        summary.append("• Co-occurrence statistics:")
        summary.append(f"  - Mean: {self.data['ObservedEntities'].mean():.2f}")
        summary.append(f"  - Median: {self.data['ObservedEntities'].median():.2f}")
        summary.append(f"  - Max: {self.data['ObservedEntities'].max():,}")
        summary.append(
            f"  - Zero values: {(self.data['ObservedEntities'] == 0).sum():,} ({(self.data['ObservedEntities'] == 0).mean()*100:.1f}%)",
        )
        summary.append("")

        # Model results
        if self.models:
            summary.append("🔬 MODEL ANALYSIS RESULTS")
            summary.append("-" * 40)

            model_summary = self.create_model_summary_table()
            if len(model_summary) > 0:
                # Find best model by AIC
                best_model_idx = model_summary["AIC"].astype(float).idxmin()
                best_model = model_summary.iloc[best_model_idx]

                summary.append(
                    f"• Best model: {best_model['Model']} (AIC: {best_model['AIC']})",
                )
                summary.append("• Model performance:")
                for _, row in model_summary.iterrows():
                    summary.append(
                        f"  - {row['Model']}: AIC={row['AIC']}, R²={row['Pseudo R²']}",
                    )
                summary.append("")

        # Coefficient interpretation
        coef_table = self.create_coefficients_table()
        if len(coef_table) > 0:
            summary.append("📈 KEY FINDINGS")
            summary.append("-" * 40)

            # Focus on significant coefficients
            significant_coefs = coef_table[coef_table["Significant"] != "Not Sig."]
            if len(significant_coefs) > 0:
                summary.append("• Significant time trends detected:")
                for _, row in significant_coefs.iterrows():
                    if (
                        "time" in row["Parameter"].lower()
                        or "year" in row["Parameter"].lower()
                    ):
                        direction = (
                            "increasing"
                            if float(row["Coefficient"]) > 0
                            else "decreasing"
                        )
                        summary.append(
                            f"  - {row['Parameter']}: {direction} trend ({row['Significant']})",
                        )
                summary.append("")

        # Data quality assessment
        summary.append("✅ DATA QUALITY ASSESSMENT")
        summary.append("-" * 40)

        # Add detailed outlier analysis with configurable threshold
        if hasattr(self, "yearly_data") and self.yearly_data is not None:
            yearly_data = self.yearly_data
            # Use configurable threshold (default 99.7th percentile)
            percentile_threshold = 99.7  # This could be made configurable too
            percentile_value = yearly_data["ObservedEntities"].quantile(
                percentile_threshold / 100,
            )
            outlier_years = yearly_data[
                yearly_data["ObservedEntities"] > percentile_value
            ]

            if len(outlier_years) > 0:
                summary.append("• Outlier years detected:")
                for _, row in outlier_years.iterrows():
                    year = int(row["Year"])
                    value = int(row["ObservedEntities"])
                    # Calculate how many times above the median
                    median_val = yearly_data["ObservedEntities"].median()
                    multiplier = value / median_val if median_val > 0 else "N/A"
                    if isinstance(multiplier, int | float):
                        summary.append(
                            f"  - {year}: {value:,} co-occurrences ({multiplier:.1f}x median)",
                        )
                    else:
                        summary.append(f"  - {year}: {value:,} co-occurrences")

                # Add statistical context
                normal_years = yearly_data[
                    yearly_data["ObservedEntities"] <= percentile_value
                ]
                summary.append(
                    f"  - Normal range ({percentile_threshold:.1f}% of years): 0 to {int(percentile_value):,} co-occurrences",
                )
                summary.append(
                    "  - These outliers were excluded from most trend visualizations",
                )
            else:
                summary.append(
                    f"• ✓ No extreme outlier years detected (>{percentile_threshold:.1f}th percentile)",
                )

        # Raw data outlier analysis with configurable threshold
        percentile_threshold_raw = 99.7  # This could be made configurable too
        percentile_value_raw = self.data["ObservedEntities"].quantile(
            percentile_threshold_raw / 100,
        )
        outlier_days = self.data[self.data["ObservedEntities"] > percentile_value_raw]

        if len(outlier_days) > 0:
            summary.append("• Daily outlier observations:")
            outlier_years_raw = sorted(outlier_days.index.year.unique())
            outlier_count = len(outlier_days)
            outlier_percentage = (outlier_count / len(self.data)) * 100
            max_daily = outlier_days["ObservedEntities"].max()

            summary.append(
                f"  - {outlier_count:,} outlier days ({outlier_percentage:.2f}% of all observations)",
            )
            summary.append(f"  - Maximum daily value: {max_daily:,} co-occurrences")
            summary.append(
                f"  - Outliers span {len(outlier_years_raw)} years: {outlier_years_raw[0]}-{outlier_years_raw[-1]}",
            )
            summary.append(
                f"  - Threshold ({percentile_threshold_raw:.1f}th percentile): {int(percentile_value_raw):,} co-occurrences",
            )
        else:
            summary.append(
                f"• ✓ No extreme daily outliers detected (>{percentile_threshold_raw:.1f}th percentile)",
            )

        # Check for data quality issues
        missing_years = []
        year_range = range(self.data.index.year.min(), self.data.index.year.max() + 1)
        observed_years = set(self.data.index.year.unique())
        missing_years = [year for year in year_range if year not in observed_years]

        if missing_years:
            # Group consecutive missing years for better reporting
            missing_ranges = []
            current_range = [missing_years[0]]

            for year in missing_years[1:]:
                if year == current_range[-1] + 1:
                    current_range.append(year)
                else:
                    if len(current_range) == 1:
                        missing_ranges.append(str(current_range[0]))
                    else:
                        missing_ranges.append(f"{current_range[0]}-{current_range[-1]}")
                    current_range = [year]

            # Add the last range
            if len(current_range) == 1:
                missing_ranges.append(str(current_range[0]))
            else:
                missing_ranges.append(f"{current_range[0]}-{current_range[-1]}")

            summary.append(
                f"• Missing years: {len(missing_years)} total ({', '.join(missing_ranges)})",
            )

            # Analyze impact of missing years
            if len(missing_years) > 10:
                summary.append(
                    "  - Large gaps may indicate data collection issues or historical events",
                )
        else:
            summary.append("• ✓ Complete year coverage")

        # Recent data assessment
        recent_data = self.data[self.data.index.year >= 2020]
        summary.append(f"• Recent data (2020+): {len(recent_data):,} observations")
        summary.append(
            f"• Zero-inflation level: {(self.data['ObservedEntities'] == 0).mean()*100:.1f}%",
        )

        return "\n".join(summary)

    def save_results(self) -> None:
        """Save all analysis results to files."""
        # Create filename suffix for filtered data
        filename_suffix = ""
        if self.start_date or self.end_date:
            date_parts = []
            if self.start_date:
                date_parts.append(f"from_{self.start_date.strftime('%Y%m%d')}")
            if self.end_date:
                date_parts.append(f"to_{self.end_date.strftime('%Y%m%d')}")
            filename_suffix = f"_{'_'.join(date_parts)}"

        # Save model summary
        model_summary = self.create_model_summary_table()
        model_summary.to_csv(
            f"{self.output_dir}/real_data_report/model_summary{filename_suffix}.csv",
            index=False,
        )

        # Save coefficients table
        coef_table = self.create_coefficients_table()
        coef_table.to_csv(
            f"{self.output_dir}/real_data_report/coefficients_table{filename_suffix}.csv",
            index=False,
        )

        # Save text summary
        text_summary = self.generate_text_summary()
        with open(
            f"{self.output_dir}/real_data_report/analysis_summary{filename_suffix}.txt",
            "w",
        ) as f:
            f.write(text_summary)

        # Save processed data summary with filter information
        data_summary = {
            "total_observations": len(self.data),
            "date_range_start": str(self.data.index.min().date()),
            "date_range_end": str(self.data.index.max().date()),
            "filter_start_date": (
                str(self.start_date.date()) if self.start_date else "None"
            ),
            "filter_end_date": str(self.end_date.date()) if self.end_date else "None",
            "years_covered": self.data.index.year.nunique(),
            "mean_co_occurrence": self.data["ObservedEntities"].mean(),
            "max_co_occurrence": self.data["ObservedEntities"].max(),
            "zero_percentage": (self.data["ObservedEntities"] == 0).mean() * 100,
        }

        pd.DataFrame([data_summary]).to_csv(
            f"{self.output_dir}/real_data_report/data_summary{filename_suffix}.csv",
            index=False,
        )

        print("\nAdditional files saved:")
        print(
            f"- Model summary: {self.output_dir}/real_data_report/model_summary{filename_suffix}.csv",
        )
        print(
            f"- Coefficients: {self.output_dir}/real_data_report/coefficients_table{filename_suffix}.csv",
        )
        print(
            f"- Text summary: {self.output_dir}/real_data_report/analysis_summary{filename_suffix}.txt",
        )
        print(
            f"- Data summary: {self.output_dir}/real_data_report/data_summary{filename_suffix}.csv",
        )

    def run_complete_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        print("=" * 80)
        print("REAL DATA COMPREHENSIVE TIME SERIES ANALYSIS")
        print("=" * 80)

        # Run all analyses
        self.run_all_analyses()

        # Create visualizations
        self.create_comprehensive_visualization()

        # Save results
        self.save_results()

        # Print summary
        print("\n" + self.generate_text_summary())

        print("\n" + "=" * 80)
        print("REAL DATA ANALYSIS COMPLETED")
        print("=" * 80)


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
