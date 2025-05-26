#!/usr/bin/env python3
"""
Real Data Comprehensive Analysis Report Generator
This script creates a complete analysis for real time series data with date-level resolution.
Adapted for co-occurrence data from 1794-2025.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

import statsmodels.formula.api as smf


class RealDataAnalysisReport:
    """Generate comprehensive analysis report for real date-level time series data."""

    def __init__(self, data_path=None, output_dir=None):
        """
        Initialize the real data analysis report.

        Parameters:
        -----------
        data_path : str
            Path to the CSV file with real data
        output_dir : str, optional
            Directory to save outputs
        """
        self.output_dir = output_dir or '/home/callebalik/EnviDis/results/analysis'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/real_data_report", exist_ok=True)

        # Load real data
        self.data_path = data_path or '/home/callebalik/EnviDis/data/raw/time-series/time_series.csv'
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
        df['Date'] = pd.to_datetime(df['pubdate_resolved'])

        # Rename for consistency with existing framework
        df['ObservedEntities'] = df['co_count']  # Use co-occurrence count as dependent variable
        df['TotalDocuments'] = df['document_count']

        # Create time variables for modeling
        df['Year'] = df['Date'].dt.year
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['MonthOfYear'] = df['Date'].dt.month

        # Create scaled time variables (important for numerical stability)
        # Use days since start for high-resolution time modeling
        start_date = df['Date'].min()
        df['DaysSinceStart'] = (df['Date'] - start_date).dt.days
        df['DaysSinceStart_scaled'] = (df['DaysSinceStart'] - df['DaysSinceStart'].mean()) / df['DaysSinceStart'].std()

        # Also create year-based scaling for comparison
        df['Year_scaled'] = (df['Year'] - df['Year'].mean()) / df['Year'].std()

        # Set Date as index for time series operations
        df = df.set_index('Date').sort_index()

        print(f"Prepared data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"ObservedEntities (co_count) range: {df['ObservedEntities'].min()} to {df['ObservedEntities'].max()}")

        return df

    def run_all_analyses(self):
        """Run all statistical analyses."""
        print("Running comprehensive statistical analysis on real data...")

        print("1. Fitting basic models (Poisson vs Negative Binomial)...")
        self.fit_basic_models()

        print("2. Fitting trend models (Linear, Quadratic, Cubic)...")
        self.fit_trend_models()

        print("3. Analyzing temporal patterns...")
        self.analyze_temporal_patterns()

    def fit_basic_models(self):
        """Fit basic count models (Poisson and Negative Binomial)."""
        # Filter out extremely high values that might cause convergence issues
        modeling_data = self.data[self.data['ObservedEntities'] <= self.data['ObservedEntities'].quantile(0.95)]

        try:
            # Poisson model with day-level time trend
            formula = 'ObservedEntities ~ DaysSinceStart_scaled + TotalDocuments'
            self.models['poisson'] = smf.poisson(formula, data=modeling_data).fit(disp=0)

            # Negative Binomial model
            self.models['negative_binomial'] = smf.negativebinomial(formula, data=modeling_data).fit(disp=0)

            print(f"   ✓ Basic models fitted on {len(modeling_data)} observations")
        except Exception as e:
            print(f"   ⚠ Basic model fitting encountered issues: {e}")

    def fit_trend_models(self):
        """Fit polynomial trend models."""
        # Use a sample for complex models to ensure computational feasibility
        sample_size = min(5000, len(self.data))
        modeling_data = self.data.sample(n=sample_size, random_state=42)
        modeling_data = modeling_data[modeling_data['ObservedEntities'] <= modeling_data['ObservedEntities'].quantile(0.95)]

        try:
            # Linear trend (year-based for interpretability)
            formula_linear = 'ObservedEntities ~ Year_scaled + TotalDocuments'
            self.models['linear'] = smf.poisson(formula_linear, data=modeling_data).fit(disp=0)

            # Quadratic trend
            formula_quad = 'ObservedEntities ~ Year_scaled + I(Year_scaled**2) + TotalDocuments'
            self.models['quadratic'] = smf.poisson(formula_quad, data=modeling_data).fit(disp=0)

            # Cubic trend
            formula_cubic = 'ObservedEntities ~ Year_scaled + I(Year_scaled**2) + I(Year_scaled**3) + TotalDocuments'
            self.models['cubic'] = smf.poisson(formula_cubic, data=modeling_data).fit(disp=0)

            print(f"   ✓ Trend models fitted on {len(modeling_data)} observations")
        except Exception as e:
            print(f"   ⚠ Trend model fitting encountered issues: {e}")

    def analyze_temporal_patterns(self):
        """Analyze temporal patterns and seasonality."""
        # Monthly aggregation for pattern analysis
        monthly_data = self.data.groupby([self.data.index.year, self.data.index.month]).agg({
            'ObservedEntities': 'sum',
            'TotalDocuments': 'sum'
        })
        monthly_data.index.names = ['Year', 'Month']
        monthly_data = monthly_data.reset_index()

        # Yearly aggregation
        yearly_data = self.data.groupby(self.data.index.year).agg({
            'ObservedEntities': 'sum',
            'TotalDocuments': 'sum'
        })
        yearly_data.index.name = 'Year'
        yearly_data = yearly_data.reset_index(drop=False)

        self.monthly_data = monthly_data
        self.yearly_data = yearly_data

        print(f"   ✓ Temporal patterns analyzed ({len(yearly_data)} years, {len(monthly_data)} months)")

    def create_model_summary_table(self):
        """Create comprehensive model summary table."""
        summary_data = []

        for model_name, model in self.models.items():
            if model is not None:
                # Calculate pseudo R-squared (McFadden's)
                if hasattr(model, 'llnull') and hasattr(model, 'llf'):
                    pseudo_r2 = 1 - (model.llf / model.llnull)
                else:
                    pseudo_r2 = np.nan

                summary_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'AIC': f"{model.aic:.2f}",
                    'BIC': f"{model.bic:.2f}" if hasattr(model, 'bic') else 'N/A',
                    'Log-Likelihood': f"{model.llf:.2f}",
                    'Pseudo R²': f"{pseudo_r2:.4f}" if not np.isnan(pseudo_r2) else 'N/A',
                    'N Observations': f"{model.nobs:.0f}",
                    'Converged': 'Yes' if model.mle_retvals['converged'] else 'No'
                })

        return pd.DataFrame(summary_data)

    def create_coefficients_table(self):
        """Create comprehensive coefficients table with confidence intervals."""
        coef_data = []

        for model_name, model in self.models.items():
            if model is not None:
                # Get confidence intervals
                try:
                    conf_int = model.conf_int(alpha=0.05)  # 95% CI
                except:
                    conf_int = None

                for param in model.params.index:
                    coef_row = {
                        'Model': model_name.replace('_', ' ').title(),
                        'Parameter': param,
                        'Coefficient': f"{model.params[param]:.4f}",
                        'Std Error': f"{model.bse[param]:.4f}",
                        'z-value': f"{model.tvalues[param]:.3f}",
                        'P-value': f"{model.pvalues[param]:.4f}",
                        'Significant': 'p<0.001' if model.pvalues[param] < 0.001
                                     else 'p<0.01' if model.pvalues[param] < 0.01
                                     else 'p<0.05' if model.pvalues[param] < 0.05
                                     else 'Not Sig.'
                    }

                    if conf_int is not None:
                        coef_row['CI Lower (95%)'] = f"{conf_int.loc[param, 0]:.4f}"
                        coef_row['CI Upper (95%)'] = f"{conf_int.loc[param, 1]:.4f}"
                    else:
                        coef_row['CI Lower (95%)'] = 'N/A'
                        coef_row['CI Upper (95%)'] = 'N/A'

                    coef_data.append(coef_row)

        return pd.DataFrame(coef_data)

    def create_comprehensive_visualization(self):
        """Create comprehensive visualization with 12 subplots."""
        fig = plt.figure(figsize=(20, 24))

        # 1. Raw time series (full data)
        ax1 = plt.subplot(4, 3, 1)
        sample_data = self.data.iloc[::100]  # Sample every 100th point for visibility
        ax1.plot(sample_data.index, sample_data['ObservedEntities'], 'b-', alpha=0.7, linewidth=0.5)
        ax1.set_title('A. Co-occurrence Counts Over Time (1794-2025)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Co-occurrence Count')
        ax1.tick_params(axis='x', rotation=45)

        # 2. Yearly aggregated data
        ax2 = plt.subplot(4, 3, 2)
        if hasattr(self, 'yearly_data'):
            ax2.plot(self.yearly_data['Year'], self.yearly_data['ObservedEntities'], 'ro-', markersize=3)
            ax2.set_title('B. Annual Co-occurrence Totals', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Annual Total')

        # 3. Distribution of counts
        ax3 = plt.subplot(4, 3, 3)
        # Log scale for better visualization
        non_zero_data = self.data[self.data['ObservedEntities'] > 0]['ObservedEntities']
        ax3.hist(np.log10(non_zero_data + 1), bins=50, alpha=0.7, color='green')
        ax3.set_title('C. Distribution of Co-occurrence Counts (Log Scale)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Log10(Count + 1)')
        ax3.set_ylabel('Frequency')

        # 4. Monthly patterns (recent years)
        ax4 = plt.subplot(4, 3, 4)
        recent_data = self.data[self.data.index.year >= 2020]
        if len(recent_data) > 0:
            monthly_means = recent_data.groupby(recent_data.index.month)['ObservedEntities'].mean()
            ax4.bar(monthly_means.index, monthly_means.values, color='purple', alpha=0.7)
            ax4.set_title('D. Monthly Pattern (2020-2025)', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Average Count')

        # 5. Documents vs Co-occurrences scatter
        ax5 = plt.subplot(4, 3, 5)
        sample_scatter = self.data.sample(n=min(1000, len(self.data)), random_state=42)
        ax5.scatter(sample_scatter['TotalDocuments'], sample_scatter['ObservedEntities'],
                   alpha=0.6, s=1, color='orange')
        ax5.set_title('E. Documents vs Co-occurrences', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Document Count')
        ax5.set_ylabel('Co-occurrence Count')

        # 6. Model summary table
        ax6 = plt.subplot(4, 3, 6)
        ax6.axis('tight')
        ax6.axis('off')
        model_summary = self.create_model_summary_table()
        if len(model_summary) > 0:
            table_data = []
            for _, row in model_summary.iterrows():
                table_data.append([row['Model'], row['AIC'], row['Pseudo R²'], row['N Observations']])

            table = ax6.table(cellText=table_data,
                            colLabels=['Model', 'AIC', 'Pseudo R²', 'N Obs'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
        ax6.set_title('F. Model Comparison', fontsize=12, fontweight='bold')

        # 7. Trend analysis (yearly data)
        ax7 = plt.subplot(4, 3, 7)
        if hasattr(self, 'yearly_data') and len(self.yearly_data) > 10:
            years = self.yearly_data['Year']
            counts = self.yearly_data['ObservedEntities']

            ax7.plot(years, counts, 'bo-', markersize=3, label='Observed')

            # Fit simple trend line for visualization
            z = np.polyfit(years, counts, 2)  # Quadratic fit
            p = np.poly1d(z)
            ax7.plot(years, p(years), 'r-', linewidth=2, label='Quadratic Trend')

            ax7.set_title('G. Long-term Trend Analysis', fontsize=12, fontweight='bold')
            ax7.set_xlabel('Year')
            ax7.set_ylabel('Annual Total')
            ax7.legend()

        # 8. Zero vs Non-zero analysis
        ax8 = plt.subplot(4, 3, 8)
        zero_counts = (self.data['ObservedEntities'] == 0).sum()
        non_zero_counts = (self.data['ObservedEntities'] > 0).sum()
        ax8.pie([zero_counts, non_zero_counts],
               labels=['Zero Co-occurrences', 'Non-zero Co-occurrences'],
               autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
        ax8.set_title('H. Zero-Inflation Analysis', fontsize=12, fontweight='bold')

        # 9. Recent trend (last 5 years)
        ax9 = plt.subplot(4, 3, 9)
        recent_data_filter = self.data.index.year >= 2020
        recent_data_subset = self.data[recent_data_filter]
        if len(recent_data_subset) > 0:
            recent_yearly = recent_data_subset.groupby(recent_data_subset.index.year)['ObservedEntities'].sum()
            ax9.bar(recent_yearly.index, recent_yearly.values, color='teal', alpha=0.7)
            ax9.set_title('I. Recent Trend (2020-2025)', fontsize=12, fontweight='bold')
            ax9.set_xlabel('Year')
            ax9.set_ylabel('Annual Total')

        # 10. Coefficient estimates (best model)
        ax10 = plt.subplot(4, 3, 10)
        ax10.axis('tight')
        ax10.axis('off')
        coef_table = self.create_coefficients_table()
        if len(coef_table) > 0 and 'quadratic' in [m.lower() for m in coef_table['Model'].values]:
            best_model_coefs = coef_table[coef_table['Model'].str.lower() == 'quadratic']
            if len(best_model_coefs) > 0:
                table_data = []
                for _, row in best_model_coefs.iterrows():
                    table_data.append([
                        row['Parameter'], row['Coefficient'],
                        row['P-value'], row['Significant']
                    ])

                table = ax10.table(cellText=table_data,
                                 colLabels=['Parameter', 'Coefficient', 'P-value', 'Significance'],
                                 cellLoc='center',
                                 loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.2, 1.5)
        ax10.set_title('J. Best Model Coefficients', fontsize=12, fontweight='bold')

        # 11. Data quality metrics
        ax11 = plt.subplot(4, 3, 11)
        ax11.axis('tight')
        ax11.axis('off')

        quality_metrics = [
            ['Total Observations', f"{len(self.data):,}"],
            ['Date Range', f"{self.data.index.min().date()} to {self.data.index.max().date()}"],
            ['Zero Values', f"{(self.data['ObservedEntities'] == 0).sum():,} ({(self.data['ObservedEntities'] == 0).mean()*100:.1f}%)"],
            ['Max Co-occurrence', f"{self.data['ObservedEntities'].max():,}"],
            ['Mean Co-occurrence', f"{self.data['ObservedEntities'].mean():.2f}"],
            ['Years Covered', f"{self.data.index.year.nunique()}"]
        ]

        table = ax11.table(cellText=quality_metrics,
                          colLabels=['Metric', 'Value'],
                          cellLoc='center',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax11.set_title('K. Data Quality Summary', fontsize=12, fontweight='bold')

        # 12. Time series decomposition preview
        ax12 = plt.subplot(4, 3, 12)
        if hasattr(self, 'yearly_data') and len(self.yearly_data) > 20:
            # Simple moving average for trend visualization
            window = min(5, len(self.yearly_data) // 4)
            rolling_mean = pd.Series(self.yearly_data['ObservedEntities']).rolling(window=window).mean()

            ax12.plot(self.yearly_data['Year'], self.yearly_data['ObservedEntities'], 'b-', alpha=0.3, label='Original')
            ax12.plot(self.yearly_data['Year'], rolling_mean, 'r-', linewidth=2, label=f'{window}-Year Moving Average')
            ax12.set_title('L. Trend Smoothing', fontsize=12, fontweight='bold')
            ax12.set_xlabel('Year')
            ax12.set_ylabel('Annual Total')
            ax12.legend()

        plt.tight_layout()

        # Save the comprehensive visualization
        output_path = f"{self.output_dir}/real_data_report/comprehensive_real_data_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive visualization saved to: {output_path}")

        return fig

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
        summary.append(f"• Date range: {self.data.index.min().date()} to {self.data.index.max().date()}")
        summary.append(f"• Years covered: {self.data.index.year.nunique()}")
        summary.append("• Co-occurrence statistics:")
        summary.append(f"  - Mean: {self.data['ObservedEntities'].mean():.2f}")
        summary.append(f"  - Median: {self.data['ObservedEntities'].median():.2f}")
        summary.append(f"  - Max: {self.data['ObservedEntities'].max():,}")
        summary.append(f"  - Zero values: {(self.data['ObservedEntities'] == 0).sum():,} ({(self.data['ObservedEntities'] == 0).mean()*100:.1f}%)")
        summary.append("")

        # Model results
        if self.models:
            summary.append("🔬 MODEL ANALYSIS RESULTS")
            summary.append("-" * 40)

            model_summary = self.create_model_summary_table()
            if len(model_summary) > 0:
                # Find best model by AIC
                best_model_idx = model_summary['AIC'].astype(float).idxmin()
                best_model = model_summary.iloc[best_model_idx]

                summary.append(f"• Best model: {best_model['Model']} (AIC: {best_model['AIC']})")
                summary.append("• Model performance:")
                for _, row in model_summary.iterrows():
                    summary.append(f"  - {row['Model']}: AIC={row['AIC']}, R²={row['Pseudo R²']}")
                summary.append("")

        # Coefficient interpretation
        coef_table = self.create_coefficients_table()
        if len(coef_table) > 0:
            summary.append("📈 KEY FINDINGS")
            summary.append("-" * 40)

            # Focus on significant coefficients
            significant_coefs = coef_table[coef_table['Significant'] != 'Not Sig.']
            if len(significant_coefs) > 0:
                summary.append("• Significant time trends detected:")
                for _, row in significant_coefs.iterrows():
                    if 'time' in row['Parameter'].lower() or 'year' in row['Parameter'].lower():
                        direction = "increasing" if float(row['Coefficient']) > 0 else "decreasing"
                        summary.append(f"  - {row['Parameter']}: {direction} trend ({row['Significant']})")
                summary.append("")

        # Data quality assessment
        summary.append("✅ DATA QUALITY ASSESSMENT")
        summary.append("-" * 40)

        # Check for data quality issues
        missing_years = []
        year_range = range(self.data.index.year.min(), self.data.index.year.max() + 1)
        observed_years = set(self.data.index.year.unique())
        missing_years = [year for year in year_range if year not in observed_years]

        if missing_years:
            summary.append(f"• Missing years: {len(missing_years)} ({missing_years[:5]}{'...' if len(missing_years) > 5 else ''})")
        else:
            summary.append("• ✓ Complete year coverage")

        # Recent data assessment
        recent_data = self.data[self.data.index.year >= 2020]
        summary.append(f"• Recent data (2020+): {len(recent_data):,} observations")
        summary.append(f"• Zero-inflation level: {(self.data['ObservedEntities'] == 0).mean()*100:.1f}%")

        return "\n".join(summary)

    def save_results(self):
        """Save all analysis results to files."""
        # Save model summary
        model_summary = self.create_model_summary_table()
        model_summary.to_csv(f"{self.output_dir}/real_data_report/model_summary.csv", index=False)

        # Save coefficients table
        coef_table = self.create_coefficients_table()
        coef_table.to_csv(f"{self.output_dir}/real_data_report/coefficients_table.csv", index=False)

        # Save text summary
        text_summary = self.generate_text_summary()
        with open(f"{self.output_dir}/real_data_report/analysis_summary.txt", 'w') as f:
            f.write(text_summary)

        # Save processed data summary
        data_summary = {
            'total_observations': len(self.data),
            'date_range_start': str(self.data.index.min().date()),
            'date_range_end': str(self.data.index.max().date()),
            'years_covered': self.data.index.year.nunique(),
            'mean_co_occurrence': self.data['ObservedEntities'].mean(),
            'max_co_occurrence': self.data['ObservedEntities'].max(),
            'zero_percentage': (self.data['ObservedEntities'] == 0).mean() * 100
        }

        pd.DataFrame([data_summary]).to_csv(f"{self.output_dir}/real_data_report/data_summary.csv", index=False)

        print("\nAdditional files saved:")
        print(f"- Model summary: {self.output_dir}/real_data_report/model_summary.csv")
        print(f"- Coefficients: {self.output_dir}/real_data_report/coefficients_table.csv")
        print(f"- Text summary: {self.output_dir}/real_data_report/analysis_summary.txt")
        print(f"- Data summary: {self.output_dir}/real_data_report/data_summary.csv")

    def run_complete_analysis(self):
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


def main():
    """Main function to run the real data analysis."""
    # Create analyzer
    analyzer = RealDataAnalysisReport()

    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
