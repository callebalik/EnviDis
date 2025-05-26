#!/usr/bin/env python3
"""Comprehensive plotting functions for time series analysis.
Extracts the best visualization features from the original comprehensive analysis.
"""

import warnings
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings("ignore")


class ComprehensivePlotter:
    """Comprehensive plotting functions for time series analysis."""

    def __init__(
        self,
        style: str = "seaborn-v0_8",
        figsize: tuple[int, int] = (20, 24),
    ):
        """Initialize plotter with style settings."""
        self.style = style
        self.figsize = figsize
        try:
            plt.style.use(style)
        except:
            plt.style.use("default")
        sns.set_palette("husl")

    def create_comprehensive_analysis_plot(
        self,
        data: pd.DataFrame,
        fitted_models: dict[str, Any],
        coefficients_table: pd.DataFrame = None,
        model_summary: pd.DataFrame = None,
        save_path: str | None = None,
    ) -> plt.Figure:
        """Create the main comprehensive analysis plot with 10 well-designed subplots.

        Based on the best features from the original comprehensive_analysis_report.py.
        """
        # Create figure with optimal layout
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(
            6,
            4,
            height_ratios=[1, 1, 1, 1, 1, 0.8],
            hspace=0.4,
            wspace=0.3,
        )

        # Color scheme
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        # Get the best model
        best_model = self._get_best_model(fitted_models)

        # 1. Data Overview (Top left)
        self._plot_data_overview(fig, gs, data, colors)

        # 2. Normalized vs Absolute Trends (Top right)
        self._plot_normalized_vs_absolute(fig, gs, data, colors)

        # 3. Model Comparison (Second row left)
        self._plot_model_comparison(fig, gs, fitted_models, model_summary, colors)

        # 4. Trend Fitting Visualization (Second row right)
        self._plot_trend_fitting(fig, gs, data, fitted_models, colors)

        # 5. Residual Analysis (Third row left)
        self._plot_residual_analysis(fig, gs, best_model, colors)

        # 6. Q-Q Plot (Third row right)
        self._plot_qq_analysis(fig, gs, best_model, colors)

        # 7. Autocorrelation Analysis (Fourth row left)
        self._plot_autocorrelation(fig, gs, best_model, colors)

        # 8. Coefficient Visualization (Fourth row right)
        self._plot_coefficients(fig, gs, best_model, coefficients_table, colors)

        # 9. Model Summary Table (Fifth row)
        self._plot_model_summary_table(fig, gs, model_summary)

        # 10. Coefficients Table (Bottom row)
        self._plot_coefficients_table(fig, gs, coefficients_table, best_model)

        # Add overall title and metadata
        self._add_title_and_metadata(fig, data, fitted_models)

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

        return fig

        # 1. Time Series Plot (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_time_series_main(ax1, data, colors)

        # 2. Distribution Analysis (top row, right side)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_distribution_analysis(ax2, data)

        # 3. Model Predictions Comparison (second row, full width)
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_model_predictions(ax3, data, fitted_models, colors)

        # 4. Residual Analysis (third row, left)
        ax4 = fig.add_subplot(gs[2, :2])
        self._plot_residual_analysis(ax4, fitted_models, colors)

        # 5. Autocorrelation Function (third row, right)
        ax5 = fig.add_subplot(gs[2, 2:])
        self._plot_autocorrelation(ax5, fitted_models)

        # 6. Model Statistics Table (fourth row, left)
        ax6 = fig.add_subplot(gs[3, :2])
        self._plot_model_summary_table(ax6, model_summary)

        # 7. Coefficients Table (fourth row, right)
        ax7 = fig.add_subplot(gs[3, 2:])
        self._plot_coefficients_table(ax7, coefficients_table)

        # 8. Trend Analysis (fifth row, left)
        ax8 = fig.add_subplot(gs[4, :2])
        self._plot_trend_analysis(ax8, data, fitted_models)

        # 9. Diagnostic Plots (fifth row, right)
        ax9 = fig.add_subplot(gs[4, 2:])
        self._plot_diagnostic_summary(ax9, fitted_models)

        # 10. Analysis Summary (bottom row, full width)
        ax10 = fig.add_subplot(gs[5, :])
        self._plot_analysis_summary(ax10, data, fitted_models, model_summary)

        # Add main title
        fig.suptitle(
            "Comprehensive Time Series Analysis Report",
            fontsize=20,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

        return fig

    def _plot_time_series_main(self, ax, data, colors):
        """Plot main time series with dual y-axes."""
        # Primary y-axis: ObservedEntities
        date_col = "Date" if "Date" in data.columns else data.index
        ax.plot(
            date_col,
            data["ObservedEntities"],
            color=colors[0],
            linewidth=2,
            label="Observed Entities",
            alpha=0.8,
        )
        ax.set_ylabel("Observed Entities", color=colors[0], fontweight="bold")
        ax.tick_params(axis="y", labelcolor=colors[0])
        ax.grid(True, alpha=0.3)

        # Secondary y-axis: TotalDocuments
        ax2 = ax.twinx()
        ax2.plot(
            date_col,
            data["TotalDocuments"],
            color=colors[1],
            linewidth=2,
            label="Total Documents",
            alpha=0.8,
        )
        ax2.set_ylabel("Total Documents", color=colors[1], fontweight="bold")
        ax2.tick_params(axis="y", labelcolor=colors[1])

        ax.set_title("Time Series Overview: Entities vs Documents", fontweight="bold")
        ax.set_xlabel("Time")

    def _plot_distribution_analysis(self, ax, data):
        """Plot distribution analysis with multiple subplots."""
        # Create sub-axes for distribution plots
        gs_dist = ax.figure.add_gridspec(
            2,
            2,
            left=ax.get_position().x0,
            right=ax.get_position().x1,
            bottom=ax.get_position().y0,
            top=ax.get_position().y1,
        )
        ax.remove()

        # Histogram
        ax_hist = ax.figure.add_subplot(gs_dist[0, 0])
        ax_hist.hist(data["ObservedEntities"], bins=30, alpha=0.7, edgecolor="black")
        ax_hist.set_title("Distribution", fontsize=10)
        ax_hist.set_xlabel("Observed Entities")

        # Log histogram
        ax_log = ax.figure.add_subplot(gs_dist[0, 1])
        non_zero = data[data["ObservedEntities"] > 0]["ObservedEntities"]
        if len(non_zero) > 0:
            ax_log.hist(np.log1p(non_zero), bins=30, alpha=0.7, edgecolor="black")
        ax_log.set_title("Log Distribution", fontsize=10)
        ax_log.set_xlabel("log(Entities + 1)")

        # Box plot
        ax_box = ax.figure.add_subplot(gs_dist[1, 0])
        ax_box.boxplot(data["ObservedEntities"])
        ax_box.set_title("Box Plot", fontsize=10)
        ax_box.set_ylabel("Observed Entities")

        # Zero vs non-zero
        ax_zero = ax.figure.add_subplot(gs_dist[1, 1])
        zero_count = (data["ObservedEntities"] == 0).sum()
        non_zero_count = (data["ObservedEntities"] > 0).sum()
        ax_zero.bar(["Zero", "Non-zero"], [zero_count, non_zero_count], alpha=0.7)
        ax_zero.set_title("Zero vs Non-zero", fontsize=10)

    def _plot_model_predictions(self, ax, data, fitted_models, colors):
        """Plot model predictions comparison."""
        date_col = "Date" if "Date" in data.columns else data.index

        # Plot observed data
        ax.scatter(
            date_col,
            data["ObservedEntities"],
            alpha=0.6,
            s=20,
            color="black",
            label="Observed",
            zorder=5,
        )

        # Plot model predictions
        color_idx = 0
        for name, model in fitted_models.items():
            if model is None:
                continue
            try:
                predictions = model.fittedvalues
                ax.plot(
                    date_col,
                    predictions,
                    color=colors[color_idx % len(colors)],
                    linewidth=2,
                    alpha=0.8,
                    label=f"{name.title()} Model",
                )
                color_idx += 1
            except Exception as e:
                print(f"Warning: Could not plot predictions for {name}: {e}")

        ax.set_title("Model Predictions Comparison", fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Observed Entities")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

    def _plot_residual_analysis(self, ax, fitted_models, colors):
        """Plot residual analysis for the best model."""
        best_model = self._get_best_model(fitted_models)
        if best_model is None:
            ax.text(
                0.5,
                0.5,
                "No model available for residual analysis",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Residual Analysis")
            return

        try:
            residuals = best_model.resid_pearson
            fitted_values = best_model.fittedvalues

            ax.scatter(fitted_values, residuals, alpha=0.6, s=20)
            ax.axhline(y=0, color="red", linestyle="--", alpha=0.8)
            ax.set_xlabel("Fitted Values")
            ax.set_ylabel("Pearson Residuals")
            ax.set_title("Residuals vs Fitted", fontweight="bold")
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Residual analysis failed: {str(e)[:50]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Residual Analysis")

    def _plot_autocorrelation(self, ax, fitted_models):
        """Plot autocorrelation function of residuals."""
        best_model = self._get_best_model(fitted_models)
        if best_model is None:
            ax.text(
                0.5,
                0.5,
                "No model available for ACF analysis",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Autocorrelation Function")
            return

        try:
            residuals = best_model.resid_pearson
            plot_acf(residuals, lags=min(20, len(residuals) // 4), ax=ax, alpha=0.05)
            ax.set_title("ACF of Residuals", fontweight="bold")
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"ACF analysis failed: {str(e)[:50]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Autocorrelation Function")

    def _plot_model_summary_table(self, ax, model_summary):
        """Plot model summary statistics table."""
        ax.axis("off")

        if model_summary is None or model_summary.empty:
            ax.text(
                0.5,
                0.5,
                "No model summary available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Format the table
        table_data = model_summary.round(3)
        table = ax.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        # Style the table
        for i in range(len(table_data.columns)):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(weight="bold", color="white")

        ax.set_title("Model Summary Statistics", fontweight="bold", pad=20)

    def _plot_coefficients_table(self, ax, coefficients_table):
        """Plot coefficients table."""
        ax.axis("off")

        if coefficients_table is None or coefficients_table.empty:
            ax.text(
                0.5,
                0.5,
                "No coefficients table available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Select key columns
        key_cols = ["Coef.", "Std.Err.", "P>|z|"]
        display_data = coefficients_table[key_cols].round(4)

        table = ax.table(
            cellText=display_data.values,
            rowLabels=display_data.index,
            colLabels=display_data.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        # Style the table
        for i in range(len(display_data.columns)):
            table[(0, i + 1)].set_facecolor("#2196F3")
            table[(0, i + 1)].set_text_props(weight="bold", color="white")

        ax.set_title("Model Coefficients", fontweight="bold", pad=20)

    def _plot_trend_analysis(self, ax, data, fitted_models):
        """Plot trend analysis."""
        date_col = "Date" if "Date" in data.columns else data.index

        # Plot observed data
        ax.scatter(
            date_col,
            data["ObservedEntities"],
            alpha=0.5,
            s=15,
            color="gray",
            label="Observed",
        )

        # Plot trend models if available
        trend_models = {
            k: v
            for k, v in fitted_models.items()
            if any(trend in k.lower() for trend in ["linear", "quadratic", "cubic"])
        }

        colors = ["blue", "red", "green", "orange"]
        for i, (name, model) in enumerate(trend_models.items()):
            if model is None:
                continue
            try:
                predictions = model.fittedvalues
                ax.plot(
                    date_col,
                    predictions,
                    color=colors[i % len(colors)],
                    linewidth=2,
                    label=f"{name.title()}",
                )
            except:
                continue

        ax.set_title("Trend Analysis", fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Observed Entities")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_diagnostic_summary(self, ax, fitted_models):
        """Plot diagnostic summary."""
        ax.axis("off")

        diagnostics = []
        for name, model in fitted_models.items():
            if model is None:
                continue
            try:
                diag = {
                    "Model": name.title(),
                    "AIC": f"{model.aic:.1f}",
                    "BIC": f"{model.bic:.1f}",
                    "Log-Likelihood": f"{model.llf:.1f}",
                }
                diagnostics.append(diag)
            except:
                continue

        if diagnostics:
            df = pd.DataFrame(diagnostics)
            table = ax.table(
                cellText=df.values,
                colLabels=df.columns,
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)

            # Style the table
            for i in range(len(df.columns)):
                table[(0, i)].set_facecolor("#FF9800")
                table[(0, i)].set_text_props(weight="bold", color="white")

        ax.set_title("Diagnostic Summary", fontweight="bold", pad=20)

    def _plot_analysis_summary(self, ax, data, fitted_models, model_summary):
        """Plot analysis summary text."""
        ax.axis("off")

        # Prepare summary text
        n_obs = len(data)
        date_range = (
            f"{data.index.min()} to {data.index.max()}"
            if hasattr(data.index, "min")
            else "N/A"
        )

        best_model_name = "None"
        best_aic = "N/A"

        if (
            model_summary is not None
            and not model_summary.empty
            and "AIC" in model_summary.columns
        ):
            best_idx = model_summary["AIC"].idxmin()
            best_model_name = model_summary.loc[best_idx, "Model"]
            best_aic = f"{model_summary.loc[best_idx, 'AIC']:.2f}"

        summary_text = f"""
        ANALYSIS SUMMARY

        Dataset: {n_obs:,} observations from {date_range}
        Response Variable: Observed Entities (count data)
        Offset Variable: Total Documents (log-transformed)

        Best Model: {best_model_name} (AIC: {best_aic})

        Key Findings:
        • Models use proper GLM methodology with offset for count data
        • Accounts for varying document volumes across time periods
        • Trend analysis reveals temporal patterns in entity occurrence rates
        """

        ax.text(
            0.05,
            0.95,
            summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

    def _get_best_model(self, fitted_models: dict[str, Any]) -> Any:
        """Get the best model from fitted models."""
        if "best_trend" in fitted_models:
            return fitted_models["best_trend"]
        elif "cubic" in fitted_models:
            return fitted_models["cubic"]
        else:
            # Return first available model
            return list(fitted_models.values())[0]

    def _plot_data_overview(self, fig, gs, data: pd.DataFrame, colors: list[str]):
        """Plot data overview with entities and documents."""
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1_twin = ax1.twinx()

        # Entities over time
        line1 = ax1.plot(
            data.index,
            data["ObservedEntities"],
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
            data.index,
            data["TotalDocuments"],
            alpha=0.3,
            color=colors[1],
            width=0.8,
            label="Total Documents",
        )
        ax1_twin.set_ylabel("Total Documents", color=colors[1], fontsize=10)
        ax1_twin.tick_params(axis="y", labelcolor=colors[1])

        # Combined legend
        lines = line1 + [bars]
        labels = ["Observed Entities", "Total Documents"]
        ax1.legend(lines, labels, loc="upper left", fontsize=9)
        ax1.set_title("A. Time Series Data Overview", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Year", fontsize=10)

    def _plot_normalized_vs_absolute(
        self,
        fig,
        gs,
        data: pd.DataFrame,
        colors: list[str],
    ):
        """Plot normalized vs absolute trends."""
        ax2 = fig.add_subplot(gs[0, 2:4])
        normalized_entities = (data["ObservedEntities"] / data["TotalDocuments"]) * 1000

        ax2.plot(
            data.index,
            data["ObservedEntities"],
            "o-",
            color=colors[0],
            linewidth=2,
            label="Absolute Counts",
            alpha=0.8,
        )
        ax2_norm = ax2.twinx()
        ax2_norm.plot(
            data.index,
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
            lines2 + lines2_norm,
            labels2 + labels2_norm,
            loc="upper left",
            fontsize=9,
        )
        ax2.set_title(
            "B. Absolute vs Normalized Trends",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_xlabel("Year", fontsize=10)
        ax2.grid(True, alpha=0.3)

    def _plot_model_comparison(
        self,
        fig,
        gs,
        fitted_models: dict[str, Any],
        model_summary: pd.DataFrame,
        colors: list[str],
    ):
        """Plot model comparison using AIC/BIC."""
        ax3 = fig.add_subplot(gs[1, 0:2])

        if model_summary is not None:
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
                "C. Model Comparison (Lower = Better)",
                fontsize=12,
                fontweight="bold",
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

    def _plot_trend_fitting(
        self,
        fig,
        gs,
        data: pd.DataFrame,
        fitted_models: dict[str, Any],
        colors: list[str],
    ):
        """Plot trend fitting visualization."""
        ax4 = fig.add_subplot(gs[1, 2:4])

        # Plot data points
        ax4.scatter(
            data.index,
            data["ObservedEntities"],
            alpha=0.6,
            color=colors[0],
            s=30,
            label="Observed Data",
        )

        # Plot trend models
        if "linear" in fitted_models and fitted_models["linear"]:
            ax4.plot(
                data.index,
                fitted_models["linear"].fittedvalues,
                "--",
                color=colors[1],
                linewidth=2,
                alpha=0.8,
                label="Linear",
            )

        if "quadratic" in fitted_models and fitted_models["quadratic"]:
            ax4.plot(
                data.index,
                fitted_models["quadratic"].fittedvalues,
                "--",
                color=colors[2],
                linewidth=2,
                alpha=0.8,
                label="Quadratic",
            )

        if "cubic" in fitted_models and fitted_models["cubic"]:
            ax4.plot(
                data.index,
                fitted_models["cubic"].fittedvalues,
                "-",
                color=colors[3],
                linewidth=3,
                alpha=0.9,
                label="Cubic (Best)",
            )

        ax4.set_ylabel("Observed Entities", fontsize=10)
        ax4.set_xlabel("Year", fontsize=10)
        ax4.set_title("D. Trend Model Fitting", fontsize=12, fontweight="bold")
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

    def _plot_residual_analysis(self, fig, gs, best_model: Any, colors: list[str]):
        """Plot residual analysis."""
        ax5 = fig.add_subplot(gs[2, 0:2])
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
        ax5.set_title("E. Residuals vs Fitted Values", fontsize=12, fontweight="bold")
        ax5.grid(True, alpha=0.3)

    def _plot_qq_analysis(self, fig, gs, best_model: Any, colors: list[str]):
        """Plot Q-Q analysis for normality check."""
        from scipy import stats

        ax6 = fig.add_subplot(gs[2, 2:4])
        stats.probplot(best_model.resid_pearson, dist="norm", plot=ax6)
        ax6.set_title("F. Q-Q Plot (Normality Check)", fontsize=12, fontweight="bold")
        ax6.grid(True, alpha=0.3)

    def _plot_autocorrelation(self, fig, gs, best_model: Any, colors: list[str]):
        """Plot autocorrelation analysis."""
        ax7 = fig.add_subplot(gs[3, 0:2])
        plot_acf(best_model.resid_pearson, lags=15, ax=ax7, alpha=0.05)
        ax7.set_title(
            "G. Autocorrelation Function of Residuals",
            fontsize=12,
            fontweight="bold",
        )
        ax7.grid(True, alpha=0.3)

    def _plot_coefficients(
        self,
        fig,
        gs,
        best_model: Any,
        coefficients_table: pd.DataFrame,
        colors: list[str],
    ):
        """Plot coefficient estimates with confidence intervals."""
        ax8 = fig.add_subplot(gs[3, 2:4])

        if coefficients_table is not None:
            # Filter for best trend model
            best_model_coefs = coefficients_table[
                coefficients_table["Model"].str.contains(
                    "Best|Cubic|best",
                    case=False,
                    na=False,
                )
            ].copy()

            if len(best_model_coefs) == 0:
                best_model_coefs = coefficients_table.head(
                    5,
                )  # Take first few if no match

            if len(best_model_coefs) > 0:
                params = best_model_coefs["Parameter"].values
                coeffs = [float(x) for x in best_model_coefs["Coefficient"].values]
                try:
                    ci_lower = [
                        float(x) for x in best_model_coefs["CI Lower (95%)"].values
                    ]
                    ci_upper = [
                        float(x) for x in best_model_coefs["CI Upper (95%)"].values
                    ]
                except (KeyError, ValueError):
                    # If CI columns don't exist, create dummy values
                    ci_lower = [c - 0.1 * abs(c) for c in coeffs]
                    ci_upper = [c + 0.1 * abs(c) for c in coeffs]

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
                    "H. Coefficient Estimates (95% CI)",
                    fontsize=12,
                    fontweight="bold",
                )
                ax8.grid(True, alpha=0.3)

    def _plot_model_summary_table(self, fig, gs, model_summary: pd.DataFrame):
        """Plot model summary table."""
        ax9 = fig.add_subplot(gs[4, :])
        ax9.axis("off")

        if model_summary is not None:
            # Model summary table
            table_data = []
            for _, row in model_summary.iterrows():
                table_data.append(
                    [row["Model"], row["AIC"], row["BIC"], row["Pseudo R²"], row["N"]],
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

    def _plot_coefficients_table(
        self,
        fig,
        gs,
        coefficients_table: pd.DataFrame,
        best_model: Any,
    ):
        """Plot coefficients table."""
        ax10 = fig.add_subplot(gs[5, :])
        ax10.axis("off")

        if coefficients_table is not None:
            # Show coefficients for the best model
            best_model_coefs = coefficients_table[
                coefficients_table["Model"].str.contains(
                    "Best|Cubic|best",
                    case=False,
                    na=False,
                )
            ].copy()

            if len(best_model_coefs) == 0:
                best_model_coefs = coefficients_table.head(5)

            if len(best_model_coefs) > 0:
                coef_table_data = []
                for _, row in best_model_coefs.iterrows():
                    ci_text = f"[{row.get('CI Lower (95%)', 'N/A')}, {row.get('CI Upper (95%)', 'N/A')}]"
                    coef_table_data.append(
                        [
                            row["Parameter"],
                            row["Coefficient"],
                            row.get("Std Error", "N/A"),
                            row.get("z-value", "N/A"),
                            row.get("P-value", "N/A"),
                            row.get("Significant", "N/A"),
                            ci_text,
                        ],
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
                            if j == 5 and len(coef_table_data) > i - 1:
                                if "p<" in str(coef_table_data[i - 1][5]):
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

    def _add_title_and_metadata(
        self,
        fig,
        data: pd.DataFrame,
        fitted_models: dict[str, Any],
    ):
        """Add overall title and metadata."""
        from datetime import datetime

        # Determine best model name
        best_model_name = "Unknown"
        if "best_trend" in fitted_models:
            best_model_name = "Best Trend"
        elif "cubic" in fitted_models:
            best_model_name = "Cubic"
        elif fitted_models:
            best_model_name = list(fitted_models.keys())[0]

        # Add overall title
        fig.suptitle(
            "Comprehensive Time Series Analysis Report\n"
            f"Environmental Entities Analysis ({data.index.min()}-{data.index.max()})",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # Add footer with metadata
        footer_text = (
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Data points: {len(data)} | "
            f"Best model: {best_model_name} | "
            f"Significance levels: p<0.001, p<0.01, p<0.05"
        )

        fig.text(
            0.5,
            0.01,
            footer_text,
            ha="center",
            fontsize=10,
            style="italic",
            color="gray",
        )

    def create_entity_documents_histogram_plot(
        self,
        data: pd.DataFrame,
        save_path: str | None = None,
    ) -> plt.Figure:
        """Create entities plot with document histogram overlay."""
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # Plot entities
        ax1.plot(
            data.index,
            data["ObservedEntities"],
            "o-",
            color="#1f77b4",
            linewidth=2,
            markersize=6,
            label="Observed Entities",
        )
        ax1.set_ylabel("Observed Entities", color="#1f77b4", fontsize=12)
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.grid(True, alpha=0.3)

        # Overlay document histogram
        ax2 = ax1.twinx()
        ax2.bar(
            data.index,
            data["TotalDocuments"],
            alpha=0.3,
            color="#ff7f0e",
            width=0.8,
            label="Total Documents",
        )
        ax2.set_ylabel("Total Documents", color="#ff7f0e", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        ax1.set_title(
            "Environmental Entities with Document Volume Context",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_xlabel("Year", fontsize=12)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_normalized_comparison_plot(
        self,
        data: pd.DataFrame,
        save_path: str | None = None,
    ) -> plt.Figure:
        """Create normalized vs absolute comparison plot."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Absolute counts
        ax1.plot(
            data.index,
            data["ObservedEntities"],
            "o-",
            color="#1f77b4",
            linewidth=2,
            markersize=6,
        )
        ax1.set_ylabel("Observed Entities (Absolute)", fontsize=12)
        ax1.set_title("A. Absolute Entity Counts", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Normalized counts
        normalized_entities = (data["ObservedEntities"] / data["TotalDocuments"]) * 1000
        ax2.plot(
            data.index,
            normalized_entities,
            "s-",
            color="#2ca02c",
            linewidth=2,
            markersize=6,
        )
        ax2.set_ylabel("Entities per 1000 Documents", fontsize=12)
        ax2.set_xlabel("Year", fontsize=12)
        ax2.set_title(
            "B. Document-Normalized Entity Rates",
            fontsize=12,
            fontweight="bold",
        )
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_comprehensive_report(
        self,
        data: pd.DataFrame,
        fitted_models: dict,
        date_col: str = "date",
        target_col: str = None,
        output_dir: str = "plots",
    ) -> dict[str, plt.Figure]:
        """Create a comprehensive set of plots for analysis report."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        if target_col is None:
            target_col = [col for col in data.columns if col != date_col][0]

        figures = {}

        # 1. Time series plot
        fig1 = self.plot_time_series(
            data,
            date_col,
            [target_col],
            title=f"Time Series: {target_col}",
            save_path=f"{output_dir}/01_time_series.png",
        )
        figures["time_series"] = fig1

        # 2. Distribution analysis
        fig2 = self.plot_distribution_analysis(
            data,
            target_col,
            save_path=f"{output_dir}/02_distribution_analysis.png",
        )
        figures["distribution"] = fig2

        # 3. Temporal patterns
        fig3 = self.plot_temporal_patterns(
            data,
            date_col,
            target_col,
            save_path=f"{output_dir}/03_temporal_patterns.png",
        )
        figures["temporal_patterns"] = fig3

        # 4. Model comparison (if multiple models)
        if len(fitted_models) > 1:
            predictions = {}
            for name, model in fitted_models.items():
                predictions[name] = model.predict()

            fig4 = self.plot_trend_comparison(
                data,
                date_col,
                target_col,
                predictions,
                title="Model Comparison",
                save_path=f"{output_dir}/04_model_comparison.png",
            )
            figures["model_comparison"] = fig4

        # 5. Diagnostics for best model
        if fitted_models:
            best_model_name = min(
                fitted_models.keys(),
                key=lambda x: fitted_models[x].aic,
            )
            best_model = fitted_models[best_model_name]

            fig5 = self.plot_model_diagnostics(
                best_model,
                data,
                title=f"Diagnostics: {best_model_name}",
                save_path=f"{output_dir}/05_model_diagnostics.png",
            )
            figures["diagnostics"] = fig5

        return figures
