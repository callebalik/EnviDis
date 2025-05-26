"""Non-Monotonic Trend Modeling Script
This script handles polynomial and spline regression models for non-monotonic trends.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from envidis.time.analysis.basic_model_fitting import BasicModelFitter


class TrendModelFitter:
    """Class for fitting non-monotonic trend models."""

    def __init__(self, data, base_family=None):
        """Initialize with data and base model family.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with required columns
        base_family : statsmodels family, optional
            Base family to use (Poisson or NegativeBinomial)

        """
        self.data = data
        if base_family is None:
            # Determine best base family
            basic_fitter = BasicModelFitter(data)
            basic_fitter.fit_poisson_model()
            basic_fitter.fit_negative_binomial_model()
            comparison = basic_fitter.compare_models()
            self.base_family = basic_fitter.chosen_model.model.family
        else:
            self.base_family = base_family

        self.linear_results = None
        self.poly2_results = None
        self.poly3_results = None
        self.spline_results = None
        self.best_model = None

    def fit_linear_model(self):
        """Fit linear trend model."""
        # Change from Year_scaled to DaysSinceStart_scaled
        formula = "co_count ~ DaysSinceStart_scaled + document_count"
        model = smf.glm(formula=formula, data=self.data, family=self.base_family)
        self.linear_results = model.fit()
        return self.linear_results

    def fit_quadratic_model(self):
        """Fit quadratic trend model."""
        # Change from Year_scaled to DaysSinceStart_scaled
        formula = "co_count ~ DaysSinceStart_scaled + I(DaysSinceStart_scaled**2) + document_count"
        model = smf.glm(formula=formula, data=self.data, family=self.base_family)
        self.poly2_results = model.fit()
        return self.poly2_results

    def fit_cubic_model(self):
        """Fit cubic trend model."""
        # Change from Year_scaled to DaysSinceStart_scaled
        formula = "co_count ~ DaysSinceStart_scaled + I(DaysSinceStart_scaled**2) + I(DaysSinceStart_scaled**3) + document_count"
        model = smf.glm(formula=formula, data=self.data, family=self.base_family)
        self.poly3_results = model.fit()
        return self.poly3_results

    def fit_spline_model(self, df=5):
        """Fit cubic spline model.

        Parameters
        ----------
        df : int
            Degrees of freedom for the spline

        """
        # Change from Year_scaled to DaysSinceStart_scaled
        formula = f"co_count ~ cr(DaysSinceStart_scaled, df={df}) + document_count"
        model = smf.glm(formula=formula, data=self.data, family=self.base_family)
        self.spline_results = model.fit()
        return self.spline_results

    def fit_all_models(self, spline_df: int = 5) -> None:
        """Fit all trend models."""
        self.fit_linear_model()
        self.fit_quadratic_model()
        self.fit_cubic_model()
        self.fit_spline_model(spline_df)

    def compare_polynomial_models(self):
        """Compare polynomial models and choose the best one.

        Returns
        -------
        dict
            Comparison results and best polynomial model

        """
        models = {
            "linear": self.linear_results,
            "quadratic": self.poly2_results,
            "cubic": self.poly3_results,
        }

        # Remove None models
        models = {k: v for k, v in models.items() if v is not None}

        best_model_name = min(models.keys(), key=lambda k: models[k].aic)
        best_model = models[best_model_name]

        comparison = {
            "models": {
                name: {"aic": model.aic, "bic": model.bic}
                for name, model in models.items()
            },
            "best_polynomial": best_model_name,
            "best_polynomial_model": best_model,
        }

        return comparison

    def compare_all_models(self):
        """Compare all models including spline and choose the overall best.

        Returns
        -------
        dict
            Complete comparison results

        """
        poly_comparison = self.compare_polynomial_models()
        best_poly = poly_comparison["best_polynomial_model"]

        if self.spline_results is not None:
            if self.spline_results.aic < best_poly.aic:
                self.best_model = self.spline_results
                best_model_name = "spline"
            else:
                self.best_model = best_poly
                best_model_name = poly_comparison["best_polynomial"]
        else:
            self.best_model = best_poly
            best_model_name = poly_comparison["best_polynomial"]

        comparison = {
            "polynomial_comparison": poly_comparison,
            "spline_aic": self.spline_results.aic if self.spline_results else None,
            "spline_bic": self.spline_results.bic if self.spline_results else None,
            "best_overall": best_model_name,
            "best_model": self.best_model,
        }

        return comparison

    def print_results(self) -> None:
        """Print comprehensive results of trend modeling."""
        if self.linear_results:
            print("\n--- Linear Trend Model Results ---")
            print(
                f"AIC: {self.linear_results.aic:.2f}, BIC: {self.linear_results.bic:.2f}",
            )

        if self.poly2_results:
            print("\n--- Quadratic Trend Model Results ---")
            print(
                f"AIC: {self.poly2_results.aic:.2f}, BIC: {self.poly2_results.bic:.2f}",
            )

        if self.poly3_results:
            print("\n--- Cubic Trend Model Results ---")
            print(
                f"AIC: {self.poly3_results.aic:.2f}, BIC: {self.poly3_results.bic:.2f}",
            )

        if self.spline_results:
            print("\n--- Spline Model Results ---")
            print(
                f"AIC: {self.spline_results.aic:.2f}, BIC: {self.spline_results.bic:.2f}",
            )

        # Print comparison
        comparison = self.compare_all_models()
        print(f"\nBest overall model: {comparison['best_overall']}")
        print(f"Best model formula: {comparison['best_model'].model.formula}")

    def plot_enhanced_trends_with_documents(
        self,
        save_path=None,
        normalize_visualization=True,
    ):
        """Create enhanced trend plots with document histograms and optional post-fitting normalization.

        Parameters
        ----------
        save_path : str, optional
            Path to save the plot
        normalize_visualization : bool, default True
            If True, normalize fitted values and observed data for visualization only
            (model fitting still uses absolute counts)

        """
        if self.best_model is None:
            self.compare_all_models()

        # Get model predictions (always fitted on absolute counts)
        fitted_values = self.best_model.fittedvalues
        observed_values = self.data["co_count"]

        # Optionally normalize for visualization
        if normalize_visualization:
            # Normalize both observed and fitted values by total documents
            normalized_observed = (observed_values / self.data["document_count"]) * 1000
            normalized_fitted = (fitted_values / self.data["document_count"]) * 1000

            # Use normalized values for plotting
            plot_observed = normalized_observed
            plot_fitted = normalized_fitted
            y_label = "Identified DIS-PNM co-occurrences per 1000 Documents"
            plot_title_suffix = "(Visualization Normalized)"
        else:
            # Use absolute values for plotting
            plot_observed = observed_values
            plot_fitted = fitted_values
            y_label = "Observed Entities"
            plot_title_suffix = "(Absolute Counts)"

        best_model_name = self.compare_all_models()["best_overall"]

        # Create the enhanced plot
        fig = plt.figure(figsize=(16, 10))

        # Plot 1: Main trend with document histogram
        ax1 = plt.subplot(2, 2, 1)
        ax1_hist = ax1.twinx()

        # Main trend line (using normalized or absolute values)
        ax1.scatter(
            self.data.index,
            plot_observed,
            alpha=0.7,
            color="blue",
            s=60,
            label="Observed Data",
            zorder=3,
        )
        ax1.plot(
            self.data.index,
            plot_fitted,
            "red",
            linewidth=3,
            label="Fitted Trend",
            zorder=4,
        )
        ax1.set_ylabel(y_label, color="blue", fontsize=12)
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.grid(True, alpha=0.3)

        # Document histogram overlay
        ax1_hist.bar(
            self.data.index,
            self.data["document_count"],
            alpha=0.3,
            color="orange",
            width=0.8,
            label="Total Documents",
            zorder=1,
        )
        ax1_hist.set_ylabel("Total Documents", color="orange", fontsize=12)
        ax1_hist.tick_params(axis="y", labelcolor="orange")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_hist.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax1.set_title(
            f"Trend Analysis {plot_title_suffix}\n{best_model_name.title()} Model",
            fontweight="bold",
        )

        # Plot 2: Trend comparison (always show both for comparison)
        ax2 = plt.subplot(2, 2, 2)

        # Always show both absolute and normalized trends for comparison
        abs_trend_effect = fitted_values - np.mean(fitted_values)
        norm_trend_effect = (
            fitted_values / self.data["document_count"] * 1000
        ) - np.mean(fitted_values / self.data["document_count"] * 1000)

        ax2.plot(
            self.data.index,
            abs_trend_effect,
            "blue",
            linewidth=3,
            label="Absolute Trend Effect",
            alpha=0.8,
        )
        ax2.plot(
            self.data.index,
            norm_trend_effect,
            "green",
            linewidth=3,
            label="Normalized Trend Effect",
            alpha=0.8,
        )
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Year", fontsize=12)
        ax2.set_ylabel("Trend Effect (Relative to Mean)", fontsize=12)
        ax2.set_title("Trend Effect Comparison", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Residuals
        ax3 = plt.subplot(2, 2, 3)
        ax3.scatter(
            self.data.index,
            self.best_model.resid_pearson,
            alpha=0.6,
            color="blue",
            s=40,
        )
        ax3.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax3.set_xlabel("Year", fontsize=12)
        ax3.set_ylabel("Standardized Residuals", fontsize=12)
        ax3.set_title(
            "Model Residuals (Always Based on Absolute Counts)",
            fontweight="bold",
        )
        ax3.grid(True, alpha=0.3)

        # Plot 4: Model diagnostics summary
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis("off")

        # Calculate R-squared
        r_squared = 1 - (self.best_model.deviance / self.best_model.null_deviance)

        summary_text = f"""
MODEL SUMMARY
{'='*25}

Model Type: {best_model_name.title()}
AIC: {self.best_model.aic:.2f}
R²: {r_squared:.4f}

Visualization Mode: {'Normalized' if normalize_visualization else 'Absolute'}

Note: Model is always fitted on
absolute counts for statistical
validity. Normalization is applied
post-fitting for visualization clarity.

Trend Direction:
{self._get_trend_direction()}
        """

        ax4.text(
            0.1,
            0.9,
            summary_text,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        return {
            "normalized_observed": (observed_values / self.data["document_count"])
            * 1000,
            "normalized_fitted": (fitted_values / self.data["document_count"]) * 1000,
            "model_results": self.best_model,
        }

    def _get_trend_direction(self):
        """Get a simple description of the trend direction."""
        fitted_values = self.best_model.fittedvalues

        if fitted_values.iloc[-1] > fitted_values.iloc[0]:
            if fitted_values.max() == fitted_values.iloc[-1]:
                return "Increasing"
            else:
                return "Non-monotonic (overall up)"
        else:
            if fitted_values.min() == fitted_values.iloc[-1]:
                return "Decreasing"
            else:
                return "Non-monotonic (overall down)"


def load_data(filepath):
    """Load preprocessed data from CSV file."""
    return pd.read_csv(filepath, index_col="Year")


if __name__ == "__main__":
    # Load data
    try:
        data = load_data(
            "/home/callebalik/EnviDis/data/processed/sample_time_series_data.csv",
        )

        # Fit trend models
        trend_fitter = TrendModelFitter(data)
        trend_fitter.fit_all_models()

        # Print results
        trend_fitter.print_results()

        # Check coefficient significance
        params = trend_fitter.best_model.params
        pvalues = trend_fitter.best_model.pvalues

        year_coef = params["Year_scaled"]
        year_pval = pvalues["Year_scaled"]

        if year_pval < 0.05:
            direction = "increasing" if year_coef > 0 else "decreasing"
            print(f"✓ Significant LINEAR trend detected: {direction}")
        else:
            print("✗ No significant linear trend detected")

    except FileNotFoundError:
        print("Sample data not found. Please run data_generation.py first.")
