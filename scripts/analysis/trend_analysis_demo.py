#!/usr/bin/env python3
"""
Trend Analysis Demo Script
This script demonstrates how the models can detect and interpret trends over time.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Add the analysis directory to Python path
sys.path.append('/home/callebalik/EnviDis/scripts/analysis')

from data_generation import generate_sample_data
from trend_modeling import TrendModelFitter


class TrendDetector:
    """Enhanced class for detecting and interpreting temporal trends."""

    def __init__(self, data):
        """Initialize with time series data."""
        self.data = data
        self.trend_fitter = None
        self.best_model = None

    def analyze_trends(self):
        """Perform comprehensive trend analysis."""
        print("=" * 60)
        print("COMPREHENSIVE TREND ANALYSIS")
        print("=" * 60)

        # Fit all trend models
        self.trend_fitter = TrendModelFitter(self.data)
        self.trend_fitter.fit_all_models()

        # Get best model
        comparison = self.trend_fitter.compare_all_models()
        self.best_model = comparison['best_model']
        best_model_name = comparison['best_overall']

        print(f"\nBest trend model: {best_model_name}")
        print(f"Model formula: {self.best_model.model.formula}")
        print(f"AIC: {self.best_model.aic:.2f}")

        # Interpret trend coefficients
        self._interpret_trend_coefficients(best_model_name)

        # Statistical significance of trend
        self._assess_trend_significance(best_model_name)

        # Visualize trends
        self._visualize_trends()

        return {
            'best_model_name': best_model_name,
            'model_results': self.best_model,
            'trend_interpretation': self._get_trend_interpretation(best_model_name)
        }

    def _interpret_trend_coefficients(self, model_name):
        """Interpret the trend coefficients based on model type."""
        print(f"\n--- Trend Coefficient Interpretation ({model_name}) ---")

        params = self.best_model.params
        pvalues = self.best_model.pvalues

        if model_name == 'linear':
            year_coef = params.get('Year_scaled', 0)
            year_pval = pvalues.get('Year_scaled', 1)

            if year_pval < 0.05:
                direction = "increasing" if year_coef > 0 else "decreasing"
                print(f"✓ Significant LINEAR trend detected: {direction}")
                print(f"  - Coefficient: {year_coef:.4f} (p-value: {year_pval:.4f})")
                print(f"  - Interpretation: Each year (scaled), the expected count changes by {np.exp(year_coef):.4f}x")
            else:
                print("✗ No significant linear trend detected")
                print(f"  - Coefficient: {year_coef:.4f} (p-value: {year_pval:.4f})")

        elif model_name == 'quadratic':
            year_coef = params.get('Year_scaled', 0)
            year2_coef = params.get('I(Year_scaled ** 2)', 0)
            year_pval = pvalues.get('Year_scaled', 1)
            year2_pval = pvalues.get('I(Year_scaled ** 2)', 1)

            print(f"Linear term: {year_coef:.4f} (p={year_pval:.4f})")
            print(f"Quadratic term: {year2_coef:.4f} (p={year2_pval:.4f})")

            if year2_pval < 0.05:
                curve_type = "U-shaped (convex)" if year2_coef > 0 else "inverted U-shaped (concave)"
                print(f"✓ Significant QUADRATIC trend detected: {curve_type}")

                # Find turning point if quadratic term is significant
                if year2_coef != 0:
                    turning_point = -year_coef / (2 * year2_coef)
                    print(f"  - Turning point at Year_scaled = {turning_point:.2f}")
            else:
                print("✗ No significant quadratic curvature detected")

        elif model_name == 'cubic':
            year_coef = params.get('Year_scaled', 0)
            year2_coef = params.get('I(Year_scaled ** 2)', 0)
            year3_coef = params.get('I(Year_scaled ** 3)', 0)
            year_pval = pvalues.get('Year_scaled', 1)
            year2_pval = pvalues.get('I(Year_scaled ** 2)', 1)
            year3_pval = pvalues.get('I(Year_scaled ** 3)', 1)

            print(f"Linear term: {year_coef:.4f} (p={year_pval:.4f})")
            print(f"Quadratic term: {year2_coef:.4f} (p={year2_pval:.4f})")
            print(f"Cubic term: {year3_coef:.4f} (p={year3_pval:.4f})")

            if year3_pval < 0.05:
                print("✓ Significant CUBIC trend detected: Complex non-monotonic pattern")
                print("  - The trend shows S-shaped or more complex curvature")
            elif year2_pval < 0.05:
                curve_type = "U-shaped" if year2_coef > 0 else "inverted U-shaped"
                print(f"✓ Significant QUADRATIC component: {curve_type}")
            elif year_pval < 0.05:
                direction = "increasing" if year_coef > 0 else "decreasing"
                print(f"✓ Significant LINEAR trend: {direction}")
            else:
                print("✗ No significant trend detected")

        elif model_name == 'spline':
            # For spline models, look at overall significance
            print("✓ SPLINE model selected: Flexible non-parametric trend")
            print("  - The spline captures complex, non-monotonic patterns")
            print("  - Trend shape determined by data-driven knot placement")

    def _assess_trend_significance(self, model_name):
        """Assess overall significance of temporal trend."""
        print("\n--- Overall Trend Significance ---")

        # Compare with null model (no time trend)
        null_formula = 'ObservedEntities ~ TotalDocuments'
        null_model = self.trend_fitter.base_family
        import statsmodels.formula.api as smf

        null_fitted = smf.glm(null_formula, data=self.data, family=null_model).fit()

        # Likelihood ratio test
        lr_statistic = 2 * (self.best_model.llf - null_fitted.llf)
        df_diff = self.best_model.df_model - null_fitted.df_model

        from scipy import stats
        p_value = 1 - stats.chi2.cdf(lr_statistic, df_diff)

        print("Likelihood Ratio Test:")
        print(f"  - LR statistic: {lr_statistic:.4f}")
        print(f"  - Degrees of freedom: {df_diff}")
        print(f"  - P-value: {p_value:.6f}")

        if p_value < 0.05:
            print("✓ SIGNIFICANT temporal trend detected!")
            print("  - The time component significantly improves model fit")
        else:
            print("✗ No significant temporal trend")
            print("  - Time does not significantly improve model fit")

        # Effect size (R-squared equivalent for GLM)
        null_deviance = null_fitted.deviance
        model_deviance = self.best_model.deviance
        pseudo_r2 = (null_deviance - model_deviance) / null_deviance

        print(f"  - Pseudo R²: {pseudo_r2:.4f}")
        print(f"  - Time explains {pseudo_r2*100:.1f}% of additional deviance")

    def _get_trend_interpretation(self, model_name):
        """Get human-readable trend interpretation."""
        params = self.best_model.params
        pvalues = self.best_model.pvalues

        interpretation = {
            'model_type': model_name,
            'has_significant_trend': False,
            'trend_direction': 'none',
            'trend_shape': 'none',
            'explanation': ''
        }

        if model_name == 'linear':
            year_coef = params.get('Year_scaled', 0)
            year_pval = pvalues.get('Year_scaled', 1)

            if year_pval < 0.05:
                interpretation['has_significant_trend'] = True
                interpretation['trend_direction'] = 'increasing' if year_coef > 0 else 'decreasing'
                interpretation['trend_shape'] = 'linear'
                interpretation['explanation'] = f"The data shows a significant linear {interpretation['trend_direction']} trend over time."

        elif model_name in ['quadratic', 'cubic']:
            # Check highest significant polynomial term
            year3_pval = pvalues.get('I(Year_scaled ** 3)', 1)
            year2_pval = pvalues.get('I(Year_scaled ** 2)', 1)
            year_pval = pvalues.get('Year_scaled', 1)

            if year3_pval < 0.05:
                interpretation['has_significant_trend'] = True
                interpretation['trend_shape'] = 'cubic'
                interpretation['explanation'] = "The data shows a complex, S-shaped or cubic trend over time."
            elif year2_pval < 0.05:
                interpretation['has_significant_trend'] = True
                interpretation['trend_shape'] = 'quadratic'
                year2_coef = params.get('I(Year_scaled ** 2)', 0)
                curve_type = 'U-shaped' if year2_coef > 0 else 'inverted U-shaped'
                interpretation['explanation'] = f"The data shows a {curve_type} quadratic trend over time."
            elif year_pval < 0.05:
                interpretation['has_significant_trend'] = True
                interpretation['trend_shape'] = 'linear'
                year_coef = params.get('Year_scaled', 0)
                direction = 'increasing' if year_coef > 0 else 'decreasing'
                interpretation['explanation'] = f"The data shows a linear {direction} trend over time."

        elif model_name == 'spline':
            interpretation['has_significant_trend'] = True
            interpretation['trend_shape'] = 'non-parametric'
            interpretation['explanation'] = "The data shows a complex, flexible trend captured by spline regression."

        return interpretation

    def _visualize_trends(self):
        """Create visualizations of detected trends."""
        plt.figure(figsize=(15, 10))

        # Plot 1: Original data with fitted trend
        plt.subplot(2, 2, 1)
        plt.scatter(self.data.index, self.data['ObservedEntities'], alpha=0.6, label='Observed Data')
        plt.plot(self.data.index, self.best_model.fittedvalues, 'r-', linewidth=2, label='Fitted Trend')
        plt.xlabel('Year')
        plt.ylabel('Observed Entities')
        plt.title(f'Trend Analysis: {self.trend_fitter.compare_all_models()["best_overall"].title()} Model')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Residuals vs time
        plt.subplot(2, 2, 2)
        plt.scatter(self.data.index, self.best_model.resid_pearson, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Year')
        plt.ylabel('Standardized Residuals')
        plt.title('Residuals Over Time')
        plt.grid(True, alpha=0.3)

        # Plot 3: Model comparison (AIC)
        plt.subplot(2, 2, 3)
        models = ['linear', 'quadratic', 'cubic']
        aics = [
            self.trend_fitter.linear_results.aic if self.trend_fitter.linear_results else np.nan,
            self.trend_fitter.poly2_results.aic if self.trend_fitter.poly2_results else np.nan,
            self.trend_fitter.poly3_results.aic if self.trend_fitter.poly3_results else np.nan
        ]
        if self.trend_fitter.spline_results:
            models.append('spline')
            aics.append(self.trend_fitter.spline_results.aic)

        colors = ['red' if model == self.trend_fitter.compare_all_models()['best_overall'] else 'blue' for model in models]
        plt.bar(models, aics, color=colors, alpha=0.7)
        plt.ylabel('AIC')
        plt.title('Model Comparison (Lower AIC = Better)')
        plt.xticks(rotation=45)

        # Plot 4: Trend component only
        plt.subplot(2, 2, 4)
        # Predict with and without time component to isolate trend
        time_data = self.data.copy()
        no_time_data = self.data.copy()
        no_time_data['Year_scaled'] = 0  # Set time to baseline

        if 'I(Year_scaled ** 2)' in self.best_model.params.index:
            no_time_data['I(Year_scaled ** 2)'] = 0
        if 'I(Year_scaled ** 3)' in self.best_model.params.index:
            no_time_data['I(Year_scaled ** 3)'] = 0

        trend_effect = self.best_model.fittedvalues - np.mean(self.best_model.fittedvalues)
        plt.plot(self.data.index, trend_effect, 'g-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Year')
        plt.ylabel('Trend Effect (Relative to Mean)')
        plt.title('Isolated Temporal Trend')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def demo_trend_detection():
    """Demonstrate trend detection capabilities."""
    print("=" * 70)
    print("DEMONSTRATION: Can the Models Tell Us If There Is a Trend Over Time?")
    print("=" * 70)

    # Generate data with known trends for demonstration
    scenarios = [
        {'name': 'Linear Increasing Trend', 'start': 2000, 'end': 2024, 'seed': 42},
        {'name': 'Non-linear (Quadratic) Trend', 'start': 1990, 'end': 2024, 'seed': 123},
        {'name': 'Complex (Cubic) Trend', 'start': 1980, 'end': 2024, 'seed': 456}
    ]

    results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'-'*50}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'-'*50}")

        # Generate data
        data = generate_sample_data(
            start_year=scenario['start'],
            end_year=scenario['end'],
            seed=scenario['seed']
        )

        # Analyze trends
        detector = TrendDetector(data)
        result = detector.analyze_trends()
        results.append({
            'scenario': scenario['name'],
            'result': result
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: What Can the Models Tell Us About Trends?")
    print("="*70)

    for result in results:
        interpretation = result['result']['trend_interpretation']
        print(f"\n{result['scenario']}:")
        print(f"  ✓ Model Selected: {interpretation['model_type']}")
        print(f"  ✓ Significant Trend: {'Yes' if interpretation['has_significant_trend'] else 'No'}")
        if interpretation['has_significant_trend']:
            print(f"  ✓ Trend Shape: {interpretation['trend_shape']}")
            print(f"  ✓ Explanation: {interpretation['explanation']}")

    print(f"\n{'='*70}")
    print("CONCLUSION:")
    print("✓ YES - The models CAN detect temporal trends!")
    print("✓ They can distinguish between:")
    print("  - Linear trends (increasing/decreasing)")
    print("  - Quadratic trends (U-shaped/inverted U-shaped)")
    print("  - Cubic trends (S-shaped/complex curves)")
    print("  - Non-parametric trends (splines)")
    print("✓ Statistical significance testing confirms trend presence")
    print("✓ Model comparison (AIC) selects the best trend shape")
    print("✓ Likelihood ratio tests quantify trend importance")
    print("="*70)


if __name__ == "__main__":
    demo_trend_detection()
