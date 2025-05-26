#!/usr/bin/env python3
"""
Modular Plotting Functions for Time Series Analysis
This module provides reusable plotting functions for different analysis modes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesPlotter:
    """Class for creating modular time series plots."""

    def __init__(self, figsize=(20, 24)):
        """Initialize the plotter with default figure size."""
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def create_figure_layout(self, nrows=5, ncols=4, height_ratios=None):
        """Create a flexible figure layout with gridspec."""
        fig = plt.figure(figsize=self.figsize)

        if height_ratios is None:
            height_ratios = [1] * nrows

        gs = gridspec.GridSpec(nrows, ncols, height_ratios=height_ratios,
                              hspace=0.3, wspace=0.3)
        return fig, gs

    def plot_time_series_overview(self, ax, data, date_col=None, value_col='ObservedEntities',
                                title='Time Series Overview', sample_rate=1):
        """Plot basic time series with optional sampling."""
        if date_col is None:
            x_data = data.index
        else:
            x_data = data[date_col]

        # Sample data if needed for performance
        if sample_rate > 1:
            data_sample = data.iloc[::sample_rate]
            if date_col is None:
                x_data = data_sample.index
            else:
                x_data = data_sample[date_col]
            y_data = data_sample[value_col]
        else:
            y_data = data[value_col]

        ax.plot(x_data, y_data, 'b-', alpha=0.7, linewidth=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(value_col.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)

        # Format x-axis for dates
        if hasattr(x_data.iloc[0] if hasattr(x_data, 'iloc') else x_data[0], 'year'):
            ax.tick_params(axis='x', rotation=45)

        return ax

    def plot_aggregated_data(self, ax, aggregated_data, x_col, y_col,
                           title='Aggregated Data', plot_type='line'):
        """Plot aggregated time series data."""
        if plot_type == 'line':
            ax.plot(aggregated_data[x_col], aggregated_data[y_col],
                   'ro-', markersize=3, linewidth=1.5)
        elif plot_type == 'bar':
            ax.bar(aggregated_data[x_col], aggregated_data[y_col],
                  alpha=0.7, color=self.colors[1])

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(x_col.replace('_', ' ').title())
        ax.set_ylabel(y_col.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)

        return ax

    def plot_distribution(self, ax, data, column='ObservedEntities', log_scale=True,
                         bins=50, title='Distribution'):
        """Plot data distribution with optional log transformation."""
        if log_scale:
            # Only plot non-zero values for log scale
            non_zero_data = data[data[column] > 0][column]
            if len(non_zero_data) > 0:
                ax.hist(np.log10(non_zero_data + 1), bins=bins, alpha=0.7,
                       color=self.colors[2])
                ax.set_xlabel(f'Log10({column.replace("_", " ")} + 1)')
            else:
                ax.text(0.5, 0.5, 'No non-zero values', ha='center', va='center',
                       transform=ax.transAxes)
        else:
            ax.hist(data[column], bins=bins, alpha=0.7, color=self.colors[2])
            ax.set_xlabel(column.replace('_', ' ').title())

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

        return ax

    def plot_scatter_relationship(self, ax, data, x_col, y_col, sample_size=1000,
                                title='Relationship Plot'):
        """Plot scatter relationship between two variables."""
        # Sample data for performance if needed
        if len(data) > sample_size:
            data_sample = data.sample(n=sample_size, random_state=42)
        else:
            data_sample = data

        ax.scatter(data_sample[x_col], data_sample[y_col],
                  alpha=0.6, s=1, color=self.colors[3])
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(x_col.replace('_', ' ').title())
        ax.set_ylabel(y_col.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)

        return ax

    def plot_model_comparison_table(self, ax, model_summary, title='Model Comparison'):
        """Create a table showing model comparison metrics."""
        ax.axis('tight')
        ax.axis('off')

        if len(model_summary) > 0:
            # Prepare table data
            table_data = []
            for _, row in model_summary.iterrows():
                table_data.append([
                    row['Model'],
                    row['AIC'],
                    row.get('BIC', 'N/A'),
                    row.get('Pseudo R²', row.get('R²', 'N/A')),
                    str(row.get('N Observations', row.get('N', 'N/A')))
                ])

            # Create table
            table = ax.table(cellText=table_data,
                           colLabels=['Model', 'AIC', 'BIC', 'R²', 'N Obs'],
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)

            # Style the table
            table[(0, 0)].set_facecolor('#E6E6FA')
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#E6E6FA')

        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

        return ax

    def plot_coefficients_with_ci(self, ax, coef_table, model_filter=None,
                                title='Coefficient Estimates with 95% CI'):
        """Plot coefficient estimates with confidence intervals."""
        if model_filter:
            filtered_coefs = coef_table[coef_table['Model'].str.contains(model_filter, case=False)]
        else:
            filtered_coefs = coef_table

        if len(filtered_coefs) > 0:
            params = filtered_coefs['Parameter'].values
            coeffs = [float(x) for x in filtered_coefs['Coefficient'].values]

            # Extract CI values if available
            if 'CI Lower (95%)' in filtered_coefs.columns and 'CI Upper (95%)' in filtered_coefs.columns:
                ci_lower = [float(x) for x in filtered_coefs['CI Lower (95%)'].values]
                ci_upper = [float(x) for x in filtered_coefs['CI Upper (95%)'].values]

                y_pos = np.arange(len(params))

                # Plot coefficients with error bars
                ax.errorbar(coeffs, y_pos,
                           xerr=[np.array(coeffs) - np.array(ci_lower),
                                np.array(ci_upper) - np.array(coeffs)],
                           fmt='o', capsize=5, capthick=2, color=self.colors[0])
            else:
                # Plot coefficients without CI
                y_pos = np.arange(len(params))
                ax.plot(coeffs, y_pos, 'o', color=self.colors[0], markersize=8)

            # Add reference line at x=0
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(params, fontsize=9)
            ax.set_xlabel('Coefficient Value')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No coefficient data available',
                   ha='center', va='center', transform=ax.transAxes)

        ax.set_title(title, fontsize=12, fontweight='bold')

        return ax

    def plot_residual_analysis(self, ax, model_results, title='Residual Analysis'):
        """Plot residual analysis for model diagnostics."""
        if model_results is not None and hasattr(model_results, 'resid_pearson'):
            residuals = model_results.resid_pearson
            fitted = model_results.fittedvalues

            ax.scatter(fitted, residuals, alpha=0.6, s=2, color=self.colors[4])
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            ax.set_xlabel('Fitted Values')
            ax.set_ylabel('Pearson Residuals')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No residual data available',
                   ha='center', va='center', transform=ax.transAxes)

        ax.set_title(title, fontsize=12, fontweight='bold')

        return ax

    def plot_autocorrelation(self, ax, model_results, lags=15, title='Autocorrelation Function'):
        """Plot autocorrelation function of residuals."""
        if model_results is not None and hasattr(model_results, 'resid_pearson'):
            plot_acf(model_results.resid_pearson, lags=lags, ax=ax, alpha=0.05)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No residual data available',
                   ha='center', va='center', transform=ax.transAxes)

        ax.set_title(title, fontsize=12, fontweight='bold')

        return ax

    def plot_model_fit_comparison(self, ax, data, models, time_col=None,
                                y_col='ObservedEntities', title='Model Fit Comparison'):
        """Plot original data with model fits overlaid."""
        # Plot original data
        if time_col is None:
            x_data = data.index
        else:
            x_data = data[time_col]

        ax.plot(x_data, data[y_col], 'ko', markersize=2, alpha=0.5, label='Observed')

        # Plot model fits
        for i, (model_name, model_results) in enumerate(models.items()):
            if model_results is not None and hasattr(model_results, 'fittedvalues'):
                color = self.colors[i % len(self.colors)]
                ax.plot(x_data, model_results.fittedvalues,
                       color=color, linewidth=2, alpha=0.8,
                       label=f'{model_name.replace("_", " ").title()}')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel(y_col.replace('_', ' ').title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        return ax

    def plot_seasonal_patterns(self, ax, data, time_col=None, y_col='ObservedEntities',
                             aggregation='month', title='Seasonal Patterns'):
        """Plot seasonal patterns in the data."""
        if time_col is None:
            date_index = data.index
        else:
            date_index = pd.to_datetime(data[time_col])

        if aggregation == 'month':
            seasonal_data = data.groupby(date_index.month)[y_col].mean()
            x_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        elif aggregation == 'quarter':
            seasonal_data = data.groupby(date_index.quarter)[y_col].mean()
            x_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        elif aggregation == 'dayofweek':
            seasonal_data = data.groupby(date_index.dayofweek)[y_col].mean()
            x_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        else:
            raise ValueError("Aggregation must be 'month', 'quarter', or 'dayofweek'")

        ax.bar(range(len(seasonal_data)), seasonal_data.values,
              alpha=0.7, color=self.colors[5])
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Mean {y_col.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)

        return ax

    def plot_data_quality_summary(self, ax, data_summary, title='Data Quality Summary'):
        """Create a table showing data quality metrics."""
        ax.axis('tight')
        ax.axis('off')

        # Prepare quality metrics table
        quality_metrics = []

        if isinstance(data_summary, dict):
            # Handle dictionary format
            quality_metrics.append(['Total Observations', f"{data_summary.get('total_observations', 'N/A'):,}"])

            date_range = data_summary.get('date_range', {})
            if isinstance(date_range, dict):
                start_date = date_range.get('start', 'N/A')
                end_date = date_range.get('end', 'N/A')
                quality_metrics.append(['Date Range', f"{start_date} to {end_date}"])
            else:
                quality_metrics.append(['Date Range', str(date_range)])

            obs_entities = data_summary.get('observed_entities', {})
            if isinstance(obs_entities, dict):
                quality_metrics.append(['Mean Value', f"{obs_entities.get('mean', 'N/A'):.2f}"])
                quality_metrics.append(['Max Value', f"{obs_entities.get('max', 'N/A'):,}"])
                zero_pct = obs_entities.get('zero_observations', 0)
                total_obs = data_summary.get('total_observations', 1)
                quality_metrics.append(['Zero Values', f"{zero_pct:,} ({zero_pct/total_obs*100:.1f}%)"])
        else:
            quality_metrics.append(['Data Summary', 'Available'])

        # Create table
        if quality_metrics:
            table = ax.table(cellText=quality_metrics,
                           colLabels=['Metric', 'Value'],
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)

            # Style the table
            for i in range(2):
                table[(0, i)].set_facecolor('#E6E6FA')

        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

        return ax


class ComprehensivePlotGenerator:
    """Generate comprehensive plots using the modular plotter."""

    def __init__(self, data, models=None, figsize=(20, 24)):
        """Initialize with data and optional models."""
        self.data = data
        self.models = models or {}
        self.plotter = TimeSeriesPlotter(figsize=figsize)

    def create_comprehensive_plot(self, model_summary=None, coef_table=None,
                                data_summary=None, output_path=None):
        """Create a comprehensive plot with multiple panels."""
        # Create figure layout
        fig, gs = self.plotter.create_figure_layout(nrows=5, ncols=4,
                                                   height_ratios=[1, 1, 1, 1, 0.8])

        # 1. Time series overview
        ax1 = fig.add_subplot(gs[0, 0:2])
        sample_rate = max(1, len(self.data) // 1000)  # Sample for performance
        self.plotter.plot_time_series_overview(ax1, self.data,
                                             title='A. Time Series Overview',
                                             sample_rate=sample_rate)

        # 2. Distribution
        ax2 = fig.add_subplot(gs[0, 2])
        self.plotter.plot_distribution(ax2, self.data,
                                     title='B. Distribution (Log Scale)')

        # 3. Zero inflation analysis
        ax3 = fig.add_subplot(gs[0, 3])
        zero_counts = (self.data['ObservedEntities'] == 0).sum()
        non_zero_counts = (self.data['ObservedEntities'] > 0).sum()
        ax3.pie([zero_counts, non_zero_counts],
               labels=['Zero Values', 'Non-zero Values'],
               autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
        ax3.set_title('C. Zero-Inflation Analysis', fontsize=12, fontweight='bold')

        # 4. Relationship plot
        ax4 = fig.add_subplot(gs[1, 0])
        self.plotter.plot_scatter_relationship(ax4, self.data,
                                             'TotalDocuments', 'ObservedEntities',
                                             title='D. Documents vs Entities')

        # 5. Seasonal patterns (if applicable)
        ax5 = fig.add_subplot(gs[1, 1])
        try:
            if hasattr(self.data.index, 'month'):
                self.plotter.plot_seasonal_patterns(ax5, self.data,
                                                   title='E. Monthly Pattern')
            else:
                ax5.text(0.5, 0.5, 'No date index for seasonal analysis',
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('E. Seasonal Pattern', fontsize=12, fontweight='bold')
        except:
            ax5.text(0.5, 0.5, 'Seasonal analysis not available',
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('E. Seasonal Pattern', fontsize=12, fontweight='bold')

        # 6. Model fit comparison
        ax6 = fig.add_subplot(gs[1, 2:4])
        if self.models:
            self.plotter.plot_model_fit_comparison(ax6, self.data, self.models,
                                                 title='F. Model Fit Comparison')
        else:
            ax6.text(0.5, 0.5, 'No models available',
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('F. Model Fit Comparison', fontsize=12, fontweight='bold')

        # 7. Residual analysis
        ax7 = fig.add_subplot(gs[2, 0])
        if self.models:
            # Use the first available model for residual analysis
            model_results = next(iter(self.models.values()))
            self.plotter.plot_residual_analysis(ax7, model_results,
                                              title='G. Residual Analysis')
        else:
            ax7.text(0.5, 0.5, 'No models for residual analysis',
                    ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('G. Residual Analysis', fontsize=12, fontweight='bold')

        # 8. Autocorrelation
        ax8 = fig.add_subplot(gs[2, 1])
        if self.models:
            model_results = next(iter(self.models.values()))
            self.plotter.plot_autocorrelation(ax8, model_results,
                                            title='H. Autocorrelation')
        else:
            ax8.text(0.5, 0.5, 'No models for autocorrelation',
                    ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('H. Autocorrelation', fontsize=12, fontweight='bold')

        # 9. Coefficient plot
        ax9 = fig.add_subplot(gs[2, 2:4])
        if coef_table is not None and len(coef_table) > 0:
            # Find the best model for coefficient plot
            best_model = None
            if model_summary is not None and len(model_summary) > 0:
                best_idx = model_summary['AIC'].astype(float).idxmin()
                best_model = model_summary.iloc[best_idx]['Model']

            self.plotter.plot_coefficients_with_ci(ax9, coef_table,
                                                 model_filter=best_model,
                                                 title='I. Coefficient Estimates')
        else:
            ax9.text(0.5, 0.5, 'No coefficient data available',
                    ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('I. Coefficient Estimates', fontsize=12, fontweight='bold')

        # 10. Model comparison table
        ax10 = fig.add_subplot(gs[3, 0:2])
        if model_summary is not None:
            self.plotter.plot_model_comparison_table(ax10, model_summary,
                                                    title='J. Model Comparison')
        else:
            ax10.text(0.5, 0.5, 'No model summary available',
                     ha='center', va='center', transform=ax10.transAxes)
            ax10.set_title('J. Model Comparison', fontsize=12, fontweight='bold')

        # 11. Data quality summary
        ax11 = fig.add_subplot(gs[3, 2:4])
        if data_summary is not None:
            self.plotter.plot_data_quality_summary(ax11, data_summary,
                                                  title='K. Data Quality Summary')
        else:
            ax11.text(0.5, 0.5, 'No data summary available',
                     ha='center', va='center', transform=ax11.transAxes)
            ax11.set_title('K. Data Quality Summary', fontsize=12, fontweight='bold')

        # 12. Footer with analysis info
        ax12 = fig.add_subplot(gs[4, :])
        ax12.axis('off')

        # Create footer text
        footer_text = f"""
        Analysis Summary: {len(self.data):,} observations | Date range: {self.data.index.min()} to {self.data.index.max()}
        Generated by Modular Time Series Analysis Framework | Significance levels: p<0.001, p<0.01, p<0.05
        """

        ax12.text(0.5, 0.5, footer_text.strip(), ha='center', va='center',
                 transform=ax12.transAxes, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

        plt.tight_layout()

        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Comprehensive plot saved to: {output_path}")

        return fig


def create_quick_overview_plot(data, output_path=None):
    """Create a quick 4-panel overview plot."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plotter = TimeSeriesPlotter()

    # Time series plot
    plotter.plot_time_series_overview(axes[0, 0], data, title='Time Series Overview')

    # Distribution
    plotter.plot_distribution(axes[0, 1], data, title='Distribution')

    # Scatter plot
    plotter.plot_scatter_relationship(axes[1, 0], data, 'TotalDocuments', 'ObservedEntities',
                                    title='Documents vs Entities')

    # Zero analysis
    zero_counts = (data['ObservedEntities'] == 0).sum()
    non_zero_counts = (data['ObservedEntities'] > 0).sum()
    axes[1, 1].pie([zero_counts, non_zero_counts],
                   labels=['Zero Values', 'Non-zero Values'],
                   autopct='%1.1f%%')
    axes[1, 1].set_title('Zero-Inflation Analysis')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Quick overview plot saved to: {output_path}")

    return fig
