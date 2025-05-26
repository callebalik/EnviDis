#!/usr/bin/env python3
"""
Diagnostic plotting functions for model validation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional
from scipy import stats


class DiagnosticPlotter:
    """Diagnostic plots for model validation and residual analysis."""

    def __init__(self, style: str = "seaborn-v0_8", figsize: tuple[int, int] = (10, 8)):
        """Initialize diagnostic plotter."""
        self.style = style
        self.figsize = figsize
        plt.style.use(style)

    def plot_residual_analysis(
        self, fitted_model, data: pd.DataFrame, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Comprehensive residual analysis plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Get model components
        predictions = fitted_model.predict()
        residuals = fitted_model.resid_pearson
        deviance_residuals = fitted_model.resid_deviance

        # 1. Residuals vs Fitted
        axes[0, 0].scatter(predictions, residuals, alpha=0.6, s=20)
        axes[0, 0].axhline(y=0, color="red", linestyle="--", linewidth=2)
        axes[0, 0].set_xlabel("Fitted Values")
        axes[0, 0].set_ylabel("Pearson Residuals")
        axes[0, 0].set_title("Residuals vs Fitted")
        axes[0, 0].grid(True, alpha=0.3)

        # Add smooth line
        try:
            from scipy.interpolate import UnivariateSpline

            sorted_idx = np.argsort(predictions)
            spline = UnivariateSpline(
                predictions[sorted_idx], residuals[sorted_idx], s=0.5
            )
            axes[0, 0].plot(
                predictions[sorted_idx],
                spline(predictions[sorted_idx]),
                color="blue",
                linewidth=2,
            )
        except:
            pass

        # 2. Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("Q-Q Plot: Pearson Residuals")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Scale-Location Plot
        sqrt_abs_residuals = np.sqrt(np.abs(residuals))
        axes[0, 2].scatter(predictions, sqrt_abs_residuals, alpha=0.6, s=20)
        axes[0, 2].set_xlabel("Fitted Values")
        axes[0, 2].set_ylabel("√|Residuals|")
        axes[0, 2].set_title("Scale-Location Plot")
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Deviance Residuals vs Fitted
        axes[1, 0].scatter(predictions, deviance_residuals, alpha=0.6, s=20)
        axes[1, 0].axhline(y=0, color="red", linestyle="--", linewidth=2)
        axes[1, 0].set_xlabel("Fitted Values")
        axes[1, 0].set_ylabel("Deviance Residuals")
        axes[1, 0].set_title("Deviance Residuals vs Fitted")
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Histogram of Residuals
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor="black")
        axes[1, 1].axvline(x=0, color="red", linestyle="--", linewidth=2)
        axes[1, 1].set_xlabel("Pearson Residuals")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Distribution of Residuals")
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Cook's Distance
        try:
            influence = fitted_model.get_influence()
            cooks_d = influence.cooks_distance[0]
            axes[1, 2].stem(range(len(cooks_d)), cooks_d, markerfmt=",")
            axes[1, 2].set_xlabel("Observation Index")
            axes[1, 2].set_ylabel("Cook's Distance")
            axes[1, 2].set_title("Cook's Distance")
            axes[1, 2].grid(True, alpha=0.3)

            # Add threshold line
            threshold = 4 / len(cooks_d)
            axes[1, 2].axhline(
                y=threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Threshold ({threshold:.4f})",
            )
            axes[1, 2].legend()
        except:
            axes[1, 2].text(
                0.5,
                0.5,
                "Cook's Distance\nNot Available",
                transform=axes[1, 2].transAxes,
                ha="center",
                va="center",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_influence_diagnostics(
        self, fitted_model, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot influence diagnostics."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        try:
            influence = fitted_model.get_influence()

            # 1. Leverage
            leverage = influence.hat_matrix_diag
            axes[0].scatter(range(len(leverage)), leverage, alpha=0.6)
            axes[0].set_xlabel("Observation Index")
            axes[0].set_ylabel("Leverage")
            axes[0].set_title("Leverage Values")
            axes[0].grid(True, alpha=0.3)

            # Add threshold
            threshold = 2 * fitted_model.df_model / len(leverage)
            axes[0].axhline(
                y=threshold,
                color="red",
                linestyle="--",
                label=f"Threshold ({threshold:.4f})",
            )
            axes[0].legend()

            # 2. DFFITS
            dffits = influence.dffits[0]
            axes[1].scatter(range(len(dffits)), dffits, alpha=0.6)
            axes[1].set_xlabel("Observation Index")
            axes[1].set_ylabel("DFFITS")
            axes[1].set_title("DFFITS")
            axes[1].grid(True, alpha=0.3)

            # Add threshold
            threshold = 2 * np.sqrt(fitted_model.df_model / len(dffits))
            axes[1].axhline(y=threshold, color="red", linestyle="--")
            axes[1].axhline(
                y=-threshold,
                color="red",
                linestyle="--",
                label=f"Threshold (±{threshold:.4f})",
            )
            axes[1].legend()

            # 3. Cook's Distance vs Leverage
            cooks_d = influence.cooks_distance[0]
            axes[2].scatter(leverage, cooks_d, alpha=0.6)
            axes[2].set_xlabel("Leverage")
            axes[2].set_ylabel("Cook's Distance")
            axes[2].set_title("Cook's Distance vs Leverage")
            axes[2].grid(True, alpha=0.3)

        except Exception as e:
            for ax in axes:
                ax.text(
                    0.5,
                    0.5,
                    f"Influence diagnostics\nnot available:\n{str(e)}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_goodness_of_fit(
        self,
        fitted_model,
        data: pd.DataFrame,
        target_col: str,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot goodness of fit diagnostics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        actual = data[target_col].values
        predicted = fitted_model.predict()

        # 1. Actual vs Predicted
        axes[0, 0].scatter(actual, predicted, alpha=0.6)
        min_val, max_val = min(actual.min(), predicted.min()), max(
            actual.max(), predicted.max()
        )
        axes[0, 0].plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="Perfect Fit",
        )
        axes[0, 0].set_xlabel("Actual Values")
        axes[0, 0].set_ylabel("Predicted Values")
        axes[0, 0].set_title("Actual vs Predicted")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Add R² annotation
        correlation = np.corrcoef(actual, predicted)[0, 1]
        r_squared = correlation**2
        axes[0, 0].text(
            0.05,
            0.95,
            f"R² = {r_squared:.4f}",
            transform=axes[0, 0].transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # 2. Residuals vs Actual
        residuals = actual - predicted
        axes[0, 1].scatter(actual, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color="red", linestyle="--", linewidth=2)
        axes[0, 1].set_xlabel("Actual Values")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residuals vs Actual")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Prediction Intervals (if available)
        try:
            pred_summary = fitted_model.get_prediction()
            conf_int = pred_summary.conf_int()

            # Sort by actual values for better visualization
            sort_idx = np.argsort(actual)
            axes[1, 0].scatter(
                actual[sort_idx], predicted[sort_idx], alpha=0.6, label="Predicted"
            )
            axes[1, 0].fill_between(
                actual[sort_idx],
                conf_int[sort_idx, 0],
                conf_int[sort_idx, 1],
                alpha=0.3,
                label="95% CI",
            )
            axes[1, 0].plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                linewidth=2,
                label="Perfect Fit",
            )
            axes[1, 0].set_xlabel("Actual Values")
            axes[1, 0].set_ylabel("Predicted Values")
            axes[1, 0].set_title("Predictions with Confidence Intervals")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        except:
            axes[1, 0].text(
                0.5,
                0.5,
                "Confidence intervals\nnot available",
                transform=axes[1, 0].transAxes,
                ha="center",
                va="center",
            )

        # 4. Model Statistics Summary
        axes[1, 1].axis("off")

        # Calculate statistics
        mae = np.mean(np.abs(residuals))
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)

        stats_text = f"""
Model Statistics:

AIC: {fitted_model.aic:.2f}
BIC: {fitted_model.bic:.2f}
Log-Likelihood: {fitted_model.llf:.2f}

R²: {r_squared:.4f}
MAE: {mae:.4f}
RMSE: {rmse:.4f}

Observations: {fitted_model.nobs}
Parameters: {fitted_model.df_model}

Model: {fitted_model.family.__class__.__name__}
"""

        axes[1, 1].text(
            0.1,
            0.9,
            stats_text,
            transform=axes[1, 1].transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig
