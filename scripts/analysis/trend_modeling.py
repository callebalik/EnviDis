"""
Non-Monotonic Trend Modeling Script
This script handles polynomial and spline regression models for non-monotonic trends.
"""

import pandas as pd
import statsmodels.formula.api as smf
from basic_model_fitting import BasicModelFitter


class TrendModelFitter:
    """Class for fitting non-monotonic trend models."""

    def __init__(self, data, base_family=None):
        """
        Initialize with data and base model family.

        Parameters:
        -----------
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
        formula = 'ObservedEntities ~ Year_scaled + TotalDocuments'
        model = smf.glm(formula=formula, data=self.data, family=self.base_family)
        self.linear_results = model.fit()
        return self.linear_results

    def fit_quadratic_model(self):
        """Fit quadratic trend model."""
        formula = 'ObservedEntities ~ Year_scaled + I(Year_scaled**2) + TotalDocuments'
        model = smf.glm(formula=formula, data=self.data, family=self.base_family)
        self.poly2_results = model.fit()
        return self.poly2_results

    def fit_cubic_model(self):
        """Fit cubic trend model."""
        formula = 'ObservedEntities ~ Year_scaled + I(Year_scaled**2) + I(Year_scaled**3) + TotalDocuments'
        model = smf.glm(formula=formula, data=self.data, family=self.base_family)
        self.poly3_results = model.fit()
        return self.poly3_results

    def fit_spline_model(self, df=5):
        """
        Fit cubic spline model.

        Parameters:
        -----------
        df : int
            Degrees of freedom for the spline
        """
        formula = f'ObservedEntities ~ cr(Year_scaled, df={df}) + TotalDocuments'
        model = smf.glm(formula=formula, data=self.data, family=self.base_family)
        self.spline_results = model.fit()
        return self.spline_results

    def fit_all_models(self, spline_df=5):
        """Fit all trend models."""
        self.fit_linear_model()
        self.fit_quadratic_model()
        self.fit_cubic_model()
        self.fit_spline_model(spline_df)

    def compare_polynomial_models(self):
        """
        Compare polynomial models and choose the best one.

        Returns:
        --------
        dict
            Comparison results and best polynomial model
        """
        models = {
            'linear': self.linear_results,
            'quadratic': self.poly2_results,
            'cubic': self.poly3_results
        }

        # Remove None models
        models = {k: v for k, v in models.items() if v is not None}

        best_model_name = min(models.keys(), key=lambda k: models[k].aic)
        best_model = models[best_model_name]

        comparison = {
            'models': {name: {'aic': model.aic, 'bic': model.bic}
                      for name, model in models.items()},
            'best_polynomial': best_model_name,
            'best_polynomial_model': best_model
        }

        return comparison

    def compare_all_models(self):
        """
        Compare all models including spline and choose the overall best.

        Returns:
        --------
        dict
            Complete comparison results
        """
        poly_comparison = self.compare_polynomial_models()
        best_poly = poly_comparison['best_polynomial_model']

        if self.spline_results is not None:
            if self.spline_results.aic < best_poly.aic:
                self.best_model = self.spline_results
                best_model_name = 'spline'
            else:
                self.best_model = best_poly
                best_model_name = poly_comparison['best_polynomial']
        else:
            self.best_model = best_poly
            best_model_name = poly_comparison['best_polynomial']

        comparison = {
            'polynomial_comparison': poly_comparison,
            'spline_aic': self.spline_results.aic if self.spline_results else None,
            'spline_bic': self.spline_results.bic if self.spline_results else None,
            'best_overall': best_model_name,
            'best_model': self.best_model
        }

        return comparison

    def print_results(self):
        """Print comprehensive results of trend modeling."""
        if self.linear_results:
            print("\n--- Linear Trend Model Results ---")
            print(f"AIC: {self.linear_results.aic:.2f}, BIC: {self.linear_results.bic:.2f}")

        if self.poly2_results:
            print("\n--- Quadratic Trend Model Results ---")
            print(f"AIC: {self.poly2_results.aic:.2f}, BIC: {self.poly2_results.bic:.2f}")
            print(self.poly2_results.summary())

        if self.poly3_results:
            print("\n--- Cubic Trend Model Results ---")
            print(f"AIC: {self.poly3_results.aic:.2f}, BIC: {self.poly3_results.bic:.2f}")
            print(self.poly3_results.summary())

        if self.spline_results:
            print("\n--- Spline Model Results ---")
            print(f"AIC: {self.spline_results.aic:.2f}, BIC: {self.spline_results.bic:.2f}")
            print(self.spline_results.summary())

        # Print comparison
        comparison = self.compare_all_models()
        print(f"\nBest overall model: {comparison['best_overall']}")
        print(f"Best model formula: {comparison['best_model'].model.formula}")


def load_data(filepath):
    """Load preprocessed data from CSV file."""
    return pd.read_csv(filepath, index_col='Year')


if __name__ == "__main__":
    # Load data
    try:
        data = load_data('/home/callebalik/EnviDis/data/processed/sample_time_series_data.csv')

        # Fit trend models
        trend_fitter = TrendModelFitter(data)
        trend_fitter.fit_all_models()

        # Print results
        trend_fitter.print_results()

    except FileNotFoundError:
        print("Sample data not found. Please run data_generation.py first.")
