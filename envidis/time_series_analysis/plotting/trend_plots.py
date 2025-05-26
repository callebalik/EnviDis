#!/usr/bin/env python3
"""
Trend plotting functions for time series analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Optional

class TrendPlotter:
    """Specialized plotting functions for trend analysis."""

    def __init__(self, style='seaborn-v0_8', figsize=(12, 8)):
        """Initialize trend plotter."""
        self.style = style
        self.figsize = figsize
        plt.style.use(style)

    def plot_polynomial_trends(self, data: pd.DataFrame,
                             date_col: str,
                             target_col: str,
                             trend_predictions: Dict[str, np.ndarray],
                             save_path: Optional[str] = None) -> plt.Figure:
        """Plot polynomial trend comparisons."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot raw data as scatter
        ax.scatter(data[date_col], data[target_col],
                  alpha=0.4, s=10, color='gray', label='Observed Data')

        # Plot trend lines
        colors = plt.cm.viridis(np.linspace(0, 1, len(trend_predictions)))

        for i, (trend_name, predictions) in enumerate(trend_predictions.items()):
            ax.plot(data[date_col], predictions,
                   color=colors[i], linewidth=3,
                   label=f'{trend_name.title()} Trend', alpha=0.8)

        ax.set_xlabel('Date')
        ax.set_ylabel(target_col)
        ax.set_title('Polynomial Trend Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_trend_decomposition(self, data: pd.DataFrame,
                               date_col: str,
                               target_col: str,
                               trend_component: np.ndarray,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot trend decomposition."""
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1]*1.5))

        # Original series
        axes[0].plot(data[date_col], data[target_col],
                    color='black', alpha=0.7)
        axes[0].set_title('Original Time Series')
        axes[0].set_ylabel(target_col)
        axes[0].grid(True, alpha=0.3)

        # Trend component
        axes[1].plot(data[date_col], trend_component,
                    color='red', linewidth=2)
        axes[1].set_title('Trend Component')
        axes[1].set_ylabel('Trend')
        axes[1].grid(True, alpha=0.3)

        # Residuals (detrended)
        residuals = data[target_col] - trend_component
        axes[2].scatter(data[date_col], residuals,
                       alpha=0.5, s=10, color='blue')
        axes[2].axhline(y=0, color='red', linestyle='--')
        axes[2].set_title('Residuals (Detrended)')
        axes[2].set_ylabel('Residuals')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_spline_comparison(self, data: pd.DataFrame,
                             date_col: str,
                             target_col: str,
                             spline_predictions: Dict[str, np.ndarray],
                             save_path: Optional[str] = None) -> plt.Figure:
        """Plot spline trend comparisons."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Convert dates to numeric for spline fitting
        date_numeric = pd.to_datetime(data[date_col]).astype('int64') // 10**9

        # Plot raw data
        ax.scatter(data[date_col], data[target_col],
                  alpha=0.3, s=5, color='lightgray', label='Data')

        # Plot splines
        colors = plt.cm.plasma(np.linspace(0, 1, len(spline_predictions)))

        for i, (spline_name, predictions) in enumerate(spline_predictions.items()):
            ax.plot(data[date_col], predictions,
                   color=colors[i], linewidth=2,
                   label=spline_name, alpha=0.8)

        ax.set_xlabel('Date')
        ax.set_ylabel(target_col)
        ax.set_title('Spline Trend Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_trend_confidence_bands(self, data: pd.DataFrame,
                                  date_col: str,
                                  target_col: str,
                                  trend_fit,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Plot trend with confidence bands."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Get predictions and confidence intervals
        predictions = trend_fit.predict()

        try:
            pred_summary = trend_fit.get_prediction()
            conf_int = pred_summary.conf_int()

            # Plot confidence bands
            ax.fill_between(data[date_col],
                           conf_int[:, 0], conf_int[:, 1],
                           alpha=0.3, color='blue',
                           label='95% Confidence Interval')
        except Exception:
            pass

        # Plot data and trend
        ax.scatter(data[date_col], data[target_col],
                  alpha=0.4, s=10, color='gray', label='Observed')
        ax.plot(data[date_col], predictions,
               color='red', linewidth=2, label='Trend')

        ax.set_xlabel('Date')
        ax.set_ylabel(target_col)
        ax.set_title('Trend with Confidence Bands')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_trend_comparison_metrics(self, model_results: Dict,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot model comparison metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        models = list(model_results.keys())
        aic_values = [model_results[m]['aic'] for m in models]
        bic_values = [model_results[m]['bic'] for m in models]
        r2_values = [model_results[m].get('r2', 0) for m in models]

        # AIC comparison
        bars1 = axes[0].bar(models, aic_values, alpha=0.7, color='skyblue')
        axes[0].set_title('AIC Comparison (Lower is Better)')
        axes[0].set_ylabel('AIC')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars1, aic_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{value:.1f}', ha='center', va='bottom')

        # BIC comparison
        bars2 = axes[1].bar(models, bic_values, alpha=0.7, color='lightcoral')
        axes[1].set_title('BIC Comparison (Lower is Better)')
        axes[1].set_ylabel('BIC')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars2, bic_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{value:.1f}', ha='center', va='bottom')

        # R² comparison
        bars3 = axes[2].bar(models, r2_values, alpha=0.7, color='lightgreen')
        axes[2].set_title('R² Comparison (Higher is Better)')
        axes[2].set_ylabel('R²')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars3, r2_values):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig
