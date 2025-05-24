"""
Autocorrelation Analysis Script
This script handles autocorrelation analysis and lagged variable modeling.
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson


class AutocorrelationAnalyzer:
    """Class for analyzing autocorrelation in model residuals."""

    def __init__(self, model_results):
        """
        Initialize with fitted model results.

        Parameters:
        -----------
        model_results : statsmodels results object
            Fitted model results
        """
        self.model_results = model_results
        self.residuals = model_results.resid_pearson
        self.dw_statistic = None
        self.acf_values = None

    def plot_acf_residuals(self, lags=None, save_path=None):
        """
        Plot autocorrelation function of residuals.

        Parameters:
        -----------
        lags : int, optional
            Number of lags to plot
        save_path : str, optional
            Path to save the plot
        """
        if lags is None:
            lags = min(20, len(self.residuals)//2 - 1)

        plot_acf(self.residuals, lags=lags, alpha=0.05)
        plt.title('Autocorrelation Function (ACF) of Model Residuals')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def durbin_watson_test(self):
        """
        Perform Durbin-Watson test for first-order autocorrelation.

        Returns:
        --------
        dict
            Test results and interpretation
        """
        # Use statsmodels' built-in durbin_watson function
        self.dw_statistic = durbin_watson(self.residuals)

        # Interpretation
        if self.dw_statistic < 1.5:
            interpretation = "Positive autocorrelation detected"
            significant = True
        elif self.dw_statistic > 2.5:
            interpretation = "Negative autocorrelation detected"
            significant = True
        else:
            interpretation = "No strong first-order autocorrelation"
            significant = False

        return {
            'dw_statistic': self.dw_statistic,
            'significant_autocorrelation': significant,
            'interpretation': interpretation
        }
    def calculate_acf(self, nlags=20):
        """
        Calculate autocorrelation function values.

        Parameters:
        -----------
        nlags : int
            Number of lags to calculate

        Returns:
        --------
        numpy.ndarray
            ACF values
        """
        self.acf_values = acf(self.residuals, nlags=nlags)
        return self.acf_values

    def print_diagnostics(self):
        """Print autocorrelation diagnostic results."""
        dw_results = self.durbin_watson_test()
        print(f"Durbin-Watson Statistic: {dw_results['dw_statistic']:.3f}")
        print(f"Interpretation: {dw_results['interpretation']}")

        if dw_results['significant_autocorrelation']:
            print("Warning: Significant autocorrelation detected in residuals.")
            print("Consider including lagged dependent variables in the model.")


class LaggedModelFitter:
    """Class for fitting models with lagged dependent variables."""

    def __init__(self, data, base_model_results):
        """
        Initialize with data and base model results.

        Parameters:
        -----------
        data : pd.DataFrame
            Original data
        base_model_results : statsmodels results object
            Base model results to extend with lags
        """
        self.data = data.copy()
        self.base_model_results = base_model_results
        self.lagged_data = None
        self.lagged_results = None

    def create_lagged_data(self, lag_periods=1):
        """
        Create lagged versions of the dependent variable.

        Parameters:
        -----------
        lag_periods : int or list
            Number of lag periods or list of lag periods
        """
        if isinstance(lag_periods, int):
            lag_periods = [lag_periods]

        self.lagged_data = self.data.copy()

        for lag in lag_periods:
            lag_col = f'ObservedEntities_lag{lag}'
            self.lagged_data[lag_col] = self.lagged_data['ObservedEntities'].shift(lag)

        # Drop rows with NaN values
        self.lagged_data = self.lagged_data.dropna()

        return self.lagged_data

    def fit_lagged_model(self, lag_periods=1):
        """
        Fit model with lagged dependent variables.

        Parameters:
        -----------
        lag_periods : int or list
            Lag periods to include
        """
        if self.lagged_data is None:
            self.create_lagged_data(lag_periods)

        # Construct formula with lags
        base_formula = self.base_model_results.model.formula

        if isinstance(lag_periods, int):
            lag_periods = [lag_periods]

        lag_terms = [f'ObservedEntities_lag{lag}' for lag in lag_periods]
        lagged_formula = base_formula + ' + ' + ' + '.join(lag_terms)

        # Fit model
        lagged_model = smf.glm(
            formula=lagged_formula,
            data=self.lagged_data,
            family=self.base_model_results.model.family
        )
        self.lagged_results = lagged_model.fit()

        return self.lagged_results

    def analyze_lagged_residuals(self, save_path=None):
        """
        Analyze residuals of the lagged model.

        Parameters:
        -----------
        save_path : str, optional
            Path to save the ACF plot
        """
        if self.lagged_results is None:
            raise ValueError("Lagged model must be fitted first")

        analyzer = AutocorrelationAnalyzer(self.lagged_results)
        analyzer.plot_acf_residuals(save_path=save_path)
        analyzer.print_diagnostics()

        return analyzer

    def print_results(self):
        """Print lagged model results."""
        if self.lagged_results is None:
            raise ValueError("Lagged model must be fitted first")

        print("\n--- Regression Results with Lagged Dependent Variable ---")
        print(self.lagged_results.summary())


def load_data(filepath):
    """Load preprocessed data from CSV file."""
    return pd.read_csv(filepath, index_col='Year')


if __name__ == "__main__":
    # This script is meant to be used with results from other scripts
    try:
        from trend_modeling import TrendModelFitter

        data = load_data('/home/callebalik/EnviDis/data/processed/sample_time_series_data.csv')

        # Get best model from trend modeling
        trend_fitter = TrendModelFitter(data)
        trend_fitter.fit_all_models()
        comparison = trend_fitter.compare_all_models()
        best_model = comparison['best_model']

        # Analyze autocorrelation
        print("=== Autocorrelation Analysis ===")
        analyzer = AutocorrelationAnalyzer(best_model)
        analyzer.plot_acf_residuals()
        analyzer.print_diagnostics()

        # Fit lagged model if autocorrelation is detected
        dw_results = analyzer.durbin_watson_test()
        if dw_results['significant_autocorrelation']:
            print("\n=== Fitting Lagged Model ===")
            lagged_fitter = LaggedModelFitter(data, best_model)
            lagged_fitter.fit_lagged_model(lag_periods=1)
            lagged_fitter.print_results()

            print("\n=== Analyzing Lagged Model Residuals ===")
            lagged_fitter.analyze_lagged_residuals()

    except (FileNotFoundError, ImportError) as e:
        print(f"Error: {e}")
        print("Make sure to run the prerequisite scripts first.")
