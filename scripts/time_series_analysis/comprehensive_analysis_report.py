#!/usr/bin/env python3
"""
Comprehensive Analysis Report Generator
This script creates a complete one-page analysis with all relevant plots,
tables of confidence intervals, coefficients, and model summaries.
Now supports both synthetic and real date-based time series data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from scipy import stats
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson

# Add the analysis directory to Python path
sys.path.append("/home/callebalik/EnviDis/scripts/analysis")

# Import necessary modules
from data_generation import generate_sample_data
from basic_model_fitting import BasicModelFitter
from trend_modeling import TrendModelFitter
from autocorrelation_analysis import AutocorrelationAnalyzer


class ComprehensiveAnalysisReport:
    """Generate a comprehensive analysis report with all relevant statistics and visualizations."""

    def __init__(self, data=None, data_path=None, output_dir=None, use_real_data=False):
        """
        Initialize the comprehensive analysis report.

        Parameters:
        -----------
        data : pd.DataFrame, optional
            Time series data. If None, generates sample data or loads real data.
        data_path : str, optional
            Path to real data CSV file
        output_dir : str, optional
            Directory to save outputs
        use_real_data : bool, optional
            Whether to load real data from CSV
        """
        self.output_dir = output_dir or "/home/callebalik/EnviDis/results/analysis"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/comprehensive_report", exist_ok=True)

        self.use_real_data = use_real_data
        self.data_path = (
            data_path or "/home/callebalik/EnviDis/data/raw/time-series/time_series.csv"
        )

        # Generate or use provided data
        if data is None:
            if use_real_data:
                print("Loading real time series data for comprehensive analysis...")
                self.data = self.load_and_prepare_real_data()
            else:
                print("Generating sample data for comprehensive analysis...")
                self.data = generate_sample_data(
                    start_year=1990,
                    end_year=2024,
                    seed=42,
                    generation_type="non_constant_upward",
                )
        else:
            self.data = data

        # Initialize analysis components
        self.basic_fitter = None
        self.trend_fitter = None
        self.analyzer = None
        self.models = {}
        self.results_summary = {}

    def load_and_prepare_real_data(self):
        """Load and prepare real time series data."""
        print(f"Loading real data from: {self.data_path}")

        # Load data
        df = pd.read_csv(self.data_path)

        # Convert date column to datetime
        df["Date"] = pd.to_datetime(df["pubdate_resolved"])

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
            f"ObservedEntities (co_count) range: {df['ObservedEntities'].min()} to {df['ObservedEntities'].max()}"
        )

        return df

    def run_all_analyses(self):
        """Run all statistical analyses."""
        print("Running comprehensive statistical analysis...")

        # Prepare data for modeling if using real data
        if self.use_real_data:
            # For real data with high resolution, we might want to use a sample for complex models
            # to ensure computational feasibility
            modeling_data = self._prepare_modeling_data()
        else:
            modeling_data = self.data

        # 1. Basic model fitting
        print("1. Fitting basic models (Poisson vs Negative Binomial)...")
        try:
            self.basic_fitter = BasicModelFitter(modeling_data)
            self.basic_fitter.fit_poisson_model()
            self.basic_fitter.fit_negative_binomial_model()
            self.basic_fitter.compare_models()
        except Exception as e:
            print(f"   Warning: Basic model fitting failed: {e}")
            self.basic_fitter = None

        # 2. Trend modeling
        print("2. Fitting trend models (Linear, Quadratic, Cubic, Spline)...")
        try:
            self.trend_fitter = TrendModelFitter(modeling_data)
            self.trend_fitter.fit_all_models()
            self.comparison = self.trend_fitter.compare_all_models()
            best_model = self.comparison["best_model"]
        except Exception as e:
            print(f"   Warning: Trend modeling failed: {e}")
            self.trend_fitter = None
            best_model = None

        # 3. Autocorrelation analysis
        print("3. Analyzing autocorrelation...")
        try:
            if best_model is not None:
                self.analyzer = AutocorrelationAnalyzer(best_model)
        except Exception as e:
            print(f"   Warning: Autocorrelation analysis failed: {e}")
            self.analyzer = None

        # Store models for reporting
        self.models = {}
        if self.basic_fitter:
            self.models.update(
                {
                    "poisson": self.basic_fitter.poisson_results,
                    "negative_binomial": self.basic_fitter.nb_results,
                }
            )

        if self.trend_fitter:
            self.models.update(
                {
                    "linear": self.trend_fitter.linear_results,
                    "quadratic": self.trend_fitter.poly2_results,
                    "cubic": self.trend_fitter.poly3_results,
                }
            )
            if best_model:
                self.models["best_trend"] = best_model
            if self.trend_fitter.spline_results:
                self.models["spline"] = self.trend_fitter.spline_results

    def _prepare_modeling_data(self):
        """Prepare modeling data for real data analysis."""
        # For very large datasets, sample to ensure computational feasibility
        if len(self.data) > 10000:
            print(
                f"   Large dataset detected ({len(self.data)} points). Sampling for modeling..."
            )
            sample_size = min(5000, len(self.data))
            modeling_data = self.data.sample(n=sample_size, random_state=42)
            # Also filter out extreme outliers that might cause convergence issues
            q95 = modeling_data["ObservedEntities"].quantile(0.95)
            modeling_data = modeling_data[modeling_data["ObservedEntities"] <= q95]
        else:
            modeling_data = self.data.copy()

        return modeling_data

    def create_model_summary_table(self):
        """Create comprehensive model summary table."""
        model_summaries = []

        for name, model in self.models.items():
            if model is not None:
                summary = {
                    "Model": name.replace("_", " ").title(),
                    "AIC": f"{model.aic:.2f}",
                    "BIC": f"{model.bic:.2f}",
                    "Log-Likelihood": f"{model.llf:.2f}",
                    "Deviance": f"{model.deviance:.2f}",
                    "Pseudo R²": f"{1 - (model.deviance / model.null_deviance):.4f}",
                    "N": model.nobs,
                }
                model_summaries.append(summary)

        return pd.DataFrame(model_summaries)

    def create_coefficients_table(self):
        """Create comprehensive coefficients table with confidence intervals."""
        coef_data = []

        for model_name, model in self.models.items():
            if model is not None:
                # Get confidence intervals
                conf_int = model.conf_int(alpha=0.05)  # 95% CI

                for param in model.params.index:
                    coef_data.append(
                        {
                            "Model": model_name.replace("_", " ").title(),
                            "Parameter": param,
                            "Coefficient": f"{model.params[param]:.4f}",
                            "Std Error": f"{model.bse[param]:.4f}",
                            "z-value": f"{model.tvalues[param]:.3f}",
                            "P-value": f"{model.pvalues[param]:.4f}",
                            "CI Lower (95%)": f"{conf_int.loc[param, 0]:.4f}",
                            "CI Upper (95%)": f"{conf_int.loc[param, 1]:.4f}",
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
                    )

        return pd.DataFrame(coef_data)

    def create_diagnostic_summary(self):
        """Create diagnostic test summary."""
        diagnostics = {}

        # Overdispersion test (for Poisson)
        if (
            self.basic_fitter
            and hasattr(self.basic_fitter, "poisson_results")
            and self.basic_fitter.poisson_results
        ):
            try:
                overdispersion = self.basic_fitter.check_overdispersion()
                diagnostics["Overdispersion Test"] = {
                    "Test": "Deviance/DF",
                    "Statistic": f"{overdispersion['dispersion_statistic']:.3f}",
                    "Interpretation": overdispersion["interpretation"],
                }
            except:
                diagnostics["Overdispersion Test"] = "Test failed"

        # Durbin-Watson test
        if self.analyzer and hasattr(self.analyzer, "durbin_watson_test"):
            try:
                dw_results = self.analyzer.durbin_watson_test()
                diagnostics["Autocorrelation Test"] = {
                    "Test": "Durbin-Watson",
                    "Statistic": f"{dw_results['dw_statistic']:.3f}",
                    "Interpretation": dw_results["interpretation"],
                }
            except:
                diagnostics["Autocorrelation Test"] = "Test failed"

        return diagnostics

    def create_comprehensive_visualization(self):
        """Create a comprehensive one-page visualization."""
        # Set up the figure with subplots
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(6, 4, hspace=0.4, wspace=0.3)

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

        # 1. Data Overview (Top row)
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1_twin = ax1.twinx()

        # Entities over time
        line1 = ax1.plot(
            viz_data.index,
            viz_data["ObservedEntities"],
            "o-",
            color=colors[0],
            linewidth=2,
            markersize=4,
            label="Observed Entities",
        )
        ax1.set_ylabel("Observed Entities", color=colors[0], fontsize=10)
        ax1.tick_params(axis="y", labelcolor=colors[0])
        ax1.grid(True, alpha=0.3)

        # Documents as bars
        bars = ax1_twin.bar(
            viz_data.index,
            viz_data["TotalDocuments"],
            alpha=0.3,
            color=colors[1],
            width=0.8 if not is_date_index else 365,
            label="Total Documents",
        )
        ax1_twin.set_ylabel("Total Documents", color=colors[1], fontsize=10)
        ax1_twin.tick_params(axis="y", labelcolor=colors[1])

        # Combined legend
        lines = line1 + [bars]
        labels = ["Observed Entities", "Total Documents"]
        ax1.legend(lines, labels, loc="upper left", fontsize=9)
        ax1.set_title("A. Time Series Data Overview", fontsize=12, fontweight="bold")

        if is_date_index:
            ax1.set_xlabel("Date", fontsize=10)
            # Format x-axis for dates
            import matplotlib.dates as mdates

            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax1.xaxis.set_major_locator(mdates.YearLocator(5))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax1.set_xlabel("Year", fontsize=10)

        # 2. Normalized vs Absolute Trends
        ax2 = fig.add_subplot(gs[0, 2:4])
        normalized_entities = (
            viz_data["ObservedEntities"] / viz_data["TotalDocuments"]
        ) * 1000

        ax2.plot(
            viz_data.index,
            viz_data["ObservedEntities"],
            "o-",
            color=colors[0],
            linewidth=2,
            label="Absolute Counts",
            alpha=0.8,
        )
        ax2_norm = ax2.twinx()
        ax2_norm.plot(
            viz_data.index,
            normalized_entities,
            "s-",
            color=colors[2],
            linewidth=2,
            label="Normalized (per 1K docs)",
            alpha=0.8,
        )

        ax2.set_ylabel("Absolute Entities", color=colors[0], fontsize=10)
        ax2_norm.set_ylabel("Entities per 1K Documents", color=colors[2], fontsize=10)
        ax2.tick_params(axis="y", labelcolor=colors[0])
        ax2_norm.tick_params(axis="y", labelcolor=colors[2])

        # Combined legend
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines2_norm, labels2_norm = ax2_norm.get_legend_handles_labels()
        ax2.legend(
            lines2 + lines2_norm, labels2 + labels2_norm, loc="upper left", fontsize=9
        )
        ax2.set_title(
            "B. Absolute vs Normalized Trends", fontsize=12, fontweight="bold"
        )

        if is_date_index:
            ax2.set_xlabel("Date", fontsize=10)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax2.xaxis.set_major_locator(mdates.YearLocator(5))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax2.set_xlabel("Year", fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 3. Model Comparison (AIC/BIC)
        ax3 = fig.add_subplot(gs[1, 0:2])
        model_summary = self.create_model_summary_table()

        if len(model_summary) > 0:
            model_names = model_summary["Model"].values
            aic_values = [float(x) for x in model_summary["AIC"].values]
            bic_values = [float(x) for x in model_summary["BIC"].values]

            x = np.arange(len(model_names))
            width = 0.35

            bars1 = ax3.bar(
                x - width / 2,
                aic_values,
                width,
                label="AIC",
                alpha=0.8,
                color=colors[3],
            )
            bars2 = ax3.bar(
                x + width / 2,
                bic_values,
                width,
                label="BIC",
                alpha=0.8,
                color=colors[4],
            )

            ax3.set_ylabel("Information Criterion Value", fontsize=10)
            ax3.set_title(
                "C. Model Comparison (Lower = Better)", fontsize=12, fontweight="bold"
            )
            ax3.set_xticks(x)
            ax3.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3, axis="y")

            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f"{height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
            for bar in bars2:
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f"{height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        else:
            ax3.text(
                0.5,
                0.5,
                "Model fitting failed\nCheck data quality",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=12,
            )
            ax3.set_title(
                "C. Model Comparison (Failed)", fontsize=12, fontweight="bold"
            )

        # 4. Trend Fitting Visualization
        ax4 = fig.add_subplot(gs[1, 2:4])

        # Plot data points (sample for large datasets)
        ax4.scatter(
            viz_data.index,
            viz_data["ObservedEntities"],
            alpha=0.6,
            color=colors[0],
            s=30,
            label="Observed Data",
        )

        # Plot trend models if available
        if (
            self.trend_fitter
            and hasattr(self.trend_fitter, "linear_results")
            and self.trend_fitter.linear_results
        ):
            # For trend plots, we need to use the same data that was used for fitting
            modeling_data = (
                self._prepare_modeling_data() if self.use_real_data else self.data
            )

            if self.trend_fitter.linear_results:
                ax4.plot(
                    modeling_data.index,
                    self.trend_fitter.linear_results.fittedvalues,
                    "--",
                    color=colors[1],
                    linewidth=2,
                    alpha=0.8,
                    label="Linear",
                )

            if self.trend_fitter.poly2_results:
                ax4.plot(
                    modeling_data.index,
                    self.trend_fitter.poly2_results.fittedvalues,
                    "--",
                    color=colors[2],
                    linewidth=2,
                    alpha=0.8,
                    label="Quadratic",
                )

            if self.trend_fitter.poly3_results:
                ax4.plot(
                    modeling_data.index,
                    self.trend_fitter.poly3_results.fittedvalues,
                    "-",
                    color=colors[3],
                    linewidth=3,
                    alpha=0.9,
                    label="Cubic",
                )

        ax4.set_ylabel("Observed Entities", fontsize=10)
        if is_date_index:
            ax4.set_xlabel("Date", fontsize=10)
            ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax4.xaxis.set_major_locator(mdates.YearLocator(5))
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax4.set_xlabel("Year", fontsize=10)
        ax4.set_title("D. Trend Model Fitting", fontsize=12, fontweight="bold")
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

        # 5. Residual Analysis
        ax5 = fig.add_subplot(gs[2, 0:2])
        ax6 = fig.add_subplot(gs[2, 2:4])

        if "best_trend" in self.models and self.models["best_trend"] is not None:
            best_model = self.models["best_trend"]

            # Residuals vs Fitted
            ax5.scatter(
                best_model.fittedvalues,
                best_model.resid_pearson,
                alpha=0.6,
                color=colors[0],
                s=30,
            )
            ax5.axhline(y=0, color="red", linestyle="--", alpha=0.8)
            ax5.set_xlabel("Fitted Values", fontsize=10)
            ax5.set_ylabel("Standardized Residuals", fontsize=10)
            ax5.set_title(
                "E. Residuals vs Fitted Values", fontsize=12, fontweight="bold"
            )
            ax5.grid(True, alpha=0.3)

            # Q-Q Plot
            stats.probplot(best_model.resid_pearson, dist="norm", plot=ax6)
            ax6.set_title(
                "F. Q-Q Plot (Normality Check)", fontsize=12, fontweight="bold"
            )
            ax6.grid(True, alpha=0.3)
        else:
            ax5.text(
                0.5,
                0.5,
                "No best model available\nfor residual analysis",
                ha="center",
                va="center",
                transform=ax5.transAxes,
                fontsize=12,
            )
            ax5.set_title(
                "E. Residuals vs Fitted Values (N/A)", fontsize=12, fontweight="bold"
            )

            ax6.text(
                0.5,
                0.5,
                "No best model available\nfor Q-Q plot",
                ha="center",
                va="center",
                transform=ax6.transAxes,
                fontsize=12,
            )
            ax6.set_title("F. Q-Q Plot (N/A)", fontsize=12, fontweight="bold")

        # 6. Autocorrelation Analysis
        ax7 = fig.add_subplot(gs[3, 0:2])

        if "best_trend" in self.models and self.models["best_trend"] is not None:
            best_model = self.models["best_trend"]
            try:
                plot_acf(
                    best_model.resid_pearson,
                    lags=min(15, len(best_model.resid_pearson) // 4),
                    ax=ax7,
                    alpha=0.05,
                )
                ax7.set_title(
                    "G. Autocorrelation Function of Residuals",
                    fontsize=12,
                    fontweight="bold",
                )
                ax7.grid(True, alpha=0.3)
            except:
                ax7.text(
                    0.5,
                    0.5,
                    "Autocorrelation\nanalysis failed",
                    ha="center",
                    va="center",
                    transform=ax7.transAxes,
                    fontsize=12,
                )
                ax7.set_title(
                    "G. Autocorrelation Analysis (Failed)",
                    fontsize=12,
                    fontweight="bold",
                )
        else:
            ax7.text(
                0.5,
                0.5,
                "No model available\nfor autocorrelation",
                ha="center",
                va="center",
                transform=ax7.transAxes,
                fontsize=12,
            )
            ax7.set_title(
                "G. Autocorrelation Analysis (N/A)", fontsize=12, fontweight="bold"
            )

        # 7. Model Coefficients Visualization
        ax8 = fig.add_subplot(gs[3, 2:4])
        coef_table = self.create_coefficients_table()

        if len(coef_table) > 0:
            # Filter for best trend model
            best_model_coefs = coef_table[coef_table["Model"] == "Best Trend"].copy()
            if len(best_model_coefs) == 0:
                best_model_coefs = coef_table[coef_table["Model"] == "Cubic"].copy()
            if len(best_model_coefs) == 0:
                best_model_coefs = coef_table[coef_table["Model"] == "Quadratic"].copy()
            if len(best_model_coefs) == 0:
                best_model_coefs = coef_table[coef_table["Model"] == "Linear"].copy()

            if len(best_model_coefs) > 0:
                params = best_model_coefs["Parameter"].values
                coeffs = [float(x) for x in best_model_coefs["Coefficient"].values]
                ci_lower = [float(x) for x in best_model_coefs["CI Lower (95%)"].values]
                ci_upper = [float(x) for x in best_model_coefs["CI Upper (95%)"].values]

                y_pos = np.arange(len(params))

                # Plot coefficients with error bars
                ax8.errorbar(
                    coeffs,
                    y_pos,
                    xerr=[
                        np.array(coeffs) - np.array(ci_lower),
                        np.array(ci_upper) - np.array(coeffs),
                    ],
                    fmt="o",
                    capsize=5,
                    capthick=2,
                    color=colors[0],
                )
                ax8.axvline(x=0, color="red", linestyle="--", alpha=0.8)
                ax8.set_yticks(y_pos)
                ax8.set_yticklabels(params, fontsize=9)
                ax8.set_xlabel("Coefficient Value", fontsize=10)
                ax8.set_title(
                    "H. Coefficient Estimates (95% CI)", fontsize=12, fontweight="bold"
                )
                ax8.grid(True, alpha=0.3)
            else:
                ax8.text(
                    0.5,
                    0.5,
                    "No coefficient data\navailable",
                    ha="center",
                    va="center",
                    transform=ax8.transAxes,
                    fontsize=12,
                )
                ax8.set_title(
                    "H. Coefficient Estimates (N/A)", fontsize=12, fontweight="bold"
                )
        else:
            ax8.text(
                0.5,
                0.5,
                "No models fitted\nsuccessfully",
                ha="center",
                va="center",
                transform=ax8.transAxes,
                fontsize=12,
            )
            ax8.set_title(
                "H. Coefficient Estimates (N/A)", fontsize=12, fontweight="bold"
            )

        # 8-9. Model Summary Tables (Bottom section)
        ax9 = fig.add_subplot(gs[4, :])
        ax9.axis("off")

        if len(model_summary) > 0:
            # Model summary table
            table_data = []
            for _, row in model_summary.iterrows():
                table_data.append(
                    [row["Model"], row["AIC"], row["BIC"], row["Pseudo R²"], row["N"]]
                )

            table = ax9.table(
                cellText=table_data,
                colLabels=["Model", "AIC", "BIC", "Pseudo R²", "N"],
                cellLoc="center",
                loc="upper left",
                colWidths=[0.15, 0.1, 0.1, 0.1, 0.08],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)

            # Style the table
            for i in range(len(table_data) + 1):
                for j in range(5):
                    cell = table[(i, j)]
                    if i == 0:  # Header
                        cell.set_facecolor("#4CAF50")
                        cell.set_text_props(weight="bold", color="white")
                    else:
                        cell.set_facecolor("#f0f0f0" if i % 2 == 0 else "white")

            ax9.text(
                0.6,
                0.8,
                "I. Model Summary Statistics",
                fontsize=12,
                fontweight="bold",
                transform=ax9.transAxes,
            )
        else:
            ax9.text(
                0.5,
                0.5,
                "No model summary available\n(Model fitting failed)",
                ha="center",
                va="center",
                transform=ax9.transAxes,
                fontsize=12,
            )
            ax9.text(
                0.6,
                0.8,
                "I. Model Summary Statistics (N/A)",
                fontsize=12,
                fontweight="bold",
                transform=ax9.transAxes,
            )

        # 10. Coefficients table
        ax10 = fig.add_subplot(gs[5, :])
        ax10.axis("off")

        # Show coefficients for the best model
        coef_table = self.create_coefficients_table()
        if len(coef_table) > 0:
            # Get best model coefficients
            best_model_coefs = coef_table[coef_table["Model"] == "Best Trend"].copy()
            if len(best_model_coefs) == 0:
                best_model_coefs = coef_table[coef_table["Model"] == "Cubic"].copy()
            if len(best_model_coefs) == 0:
                best_model_coefs = coef_table[coef_table["Model"] == "Quadratic"].copy()
            if len(best_model_coefs) == 0:
                best_model_coefs = coef_table[coef_table["Model"] == "Linear"].copy()

            if len(best_model_coefs) > 0:
                coef_table_data = []
                for _, row in best_model_coefs.iterrows():
                    coef_table_data.append(
                        [
                            row["Parameter"],
                            row["Coefficient"],
                            row["Std Error"],
                            row["z-value"],
                            row["P-value"],
                            row["Significant"],
                            f"[{row['CI Lower (95%)']}, {row['CI Upper (95%)']}]",
                        ]
                    )

                coef_table_widget = ax10.table(
                    cellText=coef_table_data,
                    colLabels=[
                        "Parameter",
                        "Coefficient",
                        "Std Error",
                        "z-value",
                        "P-value",
                        "Sig.",
                        "95% CI",
                    ],
                    cellLoc="center",
                    loc="center",
                    colWidths=[0.15, 0.12, 0.12, 0.1, 0.12, 0.06, 0.2],
                )

                coef_table_widget.auto_set_font_size(False)
                coef_table_widget.set_fontsize(8)
                coef_table_widget.scale(1, 1.8)

                # Style the coefficients table
                for i in range(len(coef_table_data) + 1):
                    for j in range(7):
                        cell = coef_table_widget[(i, j)]
                        if i == 0:  # Header
                            cell.set_facecolor("#2196F3")
                            cell.set_text_props(weight="bold", color="white")
                        else:
                            cell.set_facecolor("#f8f9fa" if i % 2 == 0 else "white")
                            # Highlight significant coefficients
                            if (
                                j == 5 and i > 0 and coef_table_data[i - 1][5]
                            ):  # Significance column
                                cell.set_facecolor("#ffeb3b")

                ax10.text(
                    0.5,
                    0.9,
                    "J. Best Model Coefficient Details",
                    fontsize=12,
                    fontweight="bold",
                    transform=ax10.transAxes,
                    ha="center",
                )
            else:
                ax10.text(
                    0.5,
                    0.5,
                    "No coefficient data available\nfor best model",
                    ha="center",
                    va="center",
                    transform=ax10.transAxes,
                    fontsize=12,
                )
                ax10.text(
                    0.5,
                    0.9,
                    "J. Best Model Coefficient Details (N/A)",
                    fontsize=12,
                    fontweight="bold",
                    transform=ax10.transAxes,
                    ha="center",
                )
        else:
            ax10.text(
                0.5,
                0.5,
                "No models fitted successfully\nNo coefficient data available",
                ha="center",
                va="center",
                transform=ax10.transAxes,
                fontsize=12,
            )
            ax10.text(
                0.5,
                0.9,
                "J. Best Model Coefficient Details (N/A)",
                fontsize=12,
                fontweight="bold",
                transform=ax10.transAxes,
                ha="center",
            )

        # Add overall title and metadata
        title_text = "Comprehensive Time Series Analysis Report\n"
        if is_date_index:
            date_range = f"{self.data.index.min().strftime('%Y-%m-%d')} to {self.data.index.max().strftime('%Y-%m-%d')}"
        else:
            date_range = f"{self.data.index.min()}-{self.data.index.max()}"
        title_text += f"Environmental Entities Analysis ({date_range})"

        fig.suptitle(title_text, fontsize=16, fontweight="bold", y=0.98)

        # Add footer with metadata
        footer_parts = [
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data points: {len(self.data):,}",
        ]

        if (
            hasattr(self, "comparison")
            and self.comparison
            and "best_overall" in self.comparison
        ):
            footer_parts.append(
                f"Best model: {self.comparison['best_overall'].title()}"
            )
        else:
            footer_parts.append("Best model: N/A")

        footer_parts.append("Significance levels: p<0.001, p<0.01, p<0.05")

        footer_text = " | ".join(footer_parts)

        fig.text(
            0.5,
            0.01,
            footer_text,
            ha="center",
            fontsize=10,
            style="italic",
            color="gray",
        )

        return fig

    def create_diagnostic_summary_text(self):
        """Create a text summary of key findings."""
        try:
            diagnostics = self.create_diagnostic_summary()
        except:
            diagnostics = {}

        # Format date range appropriately
        is_date_index = isinstance(self.data.index, pd.DatetimeIndex)
        if is_date_index:
            date_range = f"{self.data.index.min().strftime('%Y-%m-%d')} to {self.data.index.max().strftime('%Y-%m-%d')}"
        else:
            date_range = f"{self.data.index.min()}-{self.data.index.max()}"

        summary_text = f"""
COMPREHENSIVE ANALYSIS SUMMARY
================================

Dataset Overview:
- Time period: {date_range} ({len(self.data):,} observations)
- Entities range: {self.data['ObservedEntities'].min():,}-{self.data['ObservedEntities'].max():,}
- Documents range: {self.data['TotalDocuments'].min():,}-{self.data['TotalDocuments'].max():,}

Best Model Selection:
"""

        if (
            hasattr(self, "comparison")
            and self.comparison
            and "best_overall" in self.comparison
        ):
            summary_text += (
                f"- Selected model: {self.comparison['best_overall'].title()}\n"
            )
            if "best_model" in self.comparison and self.comparison["best_model"]:
                summary_text += f"- AIC: {self.comparison['best_model'].aic:.2f}\n"
                summary_text += f"- Pseudo R²: {1 - (self.comparison['best_model'].deviance / self.comparison['best_model'].null_deviance):.4f}\n"
        else:
            summary_text += "- Model fitting failed or incomplete\n"

        summary_text += "\nKey Findings:\n"

        # Add diagnostic findings
        for test_name, test_results in diagnostics.items():
            if isinstance(test_results, dict) and "Interpretation" in test_results:
                summary_text += f"- {test_name}: {test_results['Interpretation']}\n"
            else:
                summary_text += f"- {test_name}: {test_results}\n"

        # Add trend interpretation
        summary_text += "\nTrend Analysis:\n"

        if (
            hasattr(self, "comparison")
            and self.comparison
            and "best_model" in self.comparison
            and self.comparison["best_model"]
        ):
            best_model = self.comparison["best_model"]

            for param in best_model.params.index:
                if any(
                    time_word in param.lower()
                    for time_word in ["year", "time", "day", "date"]
                ):
                    coef = best_model.params[param]
                    pval = best_model.pvalues[param]
                    significance = (
                        "p<0.001"
                        if pval < 0.001
                        else (
                            "p<0.01"
                            if pval < 0.01
                            else "p<0.05" if pval < 0.05 else "Not Sig."
                        )
                    )
                    summary_text += (
                        f"- {param}: {coef:.4f} {significance} (p={pval:.4f})\n"
                    )
        else:
            summary_text += "- No trend analysis available (model fitting failed)\n"

        return summary_text

    def generate_report(self):
        """Generate the complete comprehensive report."""
        print("Generating comprehensive analysis report...")

        # Run all analyses
        self.run_all_analyses()

        # Create the comprehensive visualization
        fig = self.create_comprehensive_visualization()

        # Save the main report
        report_path = (
            f"{self.output_dir}/comprehensive_report/comprehensive_analysis_report.png"
        )
        fig.savefig(report_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Comprehensive visualization saved to: {report_path}")

        # Save individual tables
        model_summary = self.create_model_summary_table()
        model_summary.to_csv(
            f"{self.output_dir}/comprehensive_report/model_summary.csv", index=False
        )

        coef_table = self.create_coefficients_table()
        coef_table.to_csv(
            f"{self.output_dir}/comprehensive_report/coefficients_table.csv",
            index=False,
        )

        # Save text summary
        summary_text = self.create_diagnostic_summary_text()
        with open(
            f"{self.output_dir}/comprehensive_report/analysis_summary.txt", "w"
        ) as f:
            f.write(summary_text)

        print("Additional files saved:")
        print(
            f"- Model summary: {self.output_dir}/comprehensive_report/model_summary.csv"
        )
        print(
            f"- Coefficients: {self.output_dir}/comprehensive_report/coefficients_table.csv"
        )
        print(
            f"- Text summary: {self.output_dir}/comprehensive_report/analysis_summary.txt"
        )

        # Display the plot
        plt.show()

        return {
            "figure": fig,
            "model_summary": model_summary,
            "coefficients": coef_table,
            "diagnostics": self.create_diagnostic_summary(),
            "summary_text": summary_text,
        }


def main(use_real_data=False, data_path=None):
    """Main function to run the comprehensive analysis."""
    print("=" * 60)
    print("COMPREHENSIVE TIME SERIES ANALYSIS REPORT")
    print("=" * 60)

    # Create the report generator
    if use_real_data:
        print("Using real time series data...")
        report_generator = ComprehensiveAnalysisReport(
            use_real_data=True, data_path=data_path
        )
    else:
        print("Using synthetic sample data...")
        report_generator = ComprehensiveAnalysisReport()

    # Generate the complete report
    results = report_generator.generate_report()

    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETED")
    print("=" * 60)

    return results


def main_real_data(data_path=None):
    """Convenience function to run analysis on real data."""
    return main(use_real_data=True, data_path=data_path)


if __name__ == "__main__":
    import sys

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        data_path = sys.argv[2] if len(sys.argv) > 2 else None
        results = main_real_data(data_path)
    else:
        results = main()
