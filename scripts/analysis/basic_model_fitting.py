"""
Basic Model Fitting Script
This script handles fitting Poisson and Negative Binomial regression models.
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


class BasicModelFitter:
    """Class for fitting and comparing basic regression models."""

    def __init__(self, data):
        """
        Initialize with data.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with required columns: ObservedEntities, Year_scaled, TotalDocuments
        """
        self.data = data
        self.poisson_results = None
        self.nb_results = None
        self.chosen_model = None

    def fit_poisson_model(self):
        """Fit Poisson regression model."""
        formula = 'ObservedEntities ~ Year_scaled + TotalDocuments'
        poisson_model = smf.glm(
            formula=formula,
            data=self.data,
            family=sm.families.Poisson()
        )
        self.poisson_results = poisson_model.fit()
        return self.poisson_results

    def fit_negative_binomial_model(self):
        """Fit Negative Binomial regression model."""
        formula = 'ObservedEntities ~ Year_scaled + TotalDocuments'
        nb_model = smf.glm(
            formula=formula,
            data=self.data,
            family=sm.families.NegativeBinomial()
        )
        self.nb_results = nb_model.fit()
        return self.nb_results

    def check_overdispersion(self):
        """
        Check for overdispersion in Poisson model.

        Returns:
        --------
        dict
            Dictionary with dispersion statistics and interpretation
        """
        if self.poisson_results is None:
            self.fit_poisson_model()

        deviance = self.poisson_results.deviance
        df_resid = self.poisson_results.df_resid
        dispersion_statistic = deviance / df_resid

        overdispersed = dispersion_statistic > 1.2

        return {
            'dispersion_statistic': dispersion_statistic,
            'overdispersed': overdispersed,
            'interpretation': (
                "Indication of overdispersion. Negative Binomial model might be more appropriate."
                if overdispersed
                else "No strong indication of overdispersion. Poisson model might be sufficient."
            )
        }

    def compare_models(self):
        """
        Compare Poisson and Negative Binomial models using AIC/BIC.

        Returns:
        --------
        dict
            Comparison results and chosen model
        """
        if self.poisson_results is None:
            self.fit_poisson_model()
        if self.nb_results is None:
            self.fit_negative_binomial_model()

        comparison = {
            'poisson_aic': self.poisson_results.aic,
            'poisson_bic': self.poisson_results.bic,
            'nb_aic': self.nb_results.aic,
            'nb_bic': self.nb_results.bic,
        }

        if self.nb_results.aic < self.poisson_results.aic:
            self.chosen_model = self.nb_results
            comparison['chosen_model'] = 'Negative Binomial'
            comparison['reason'] = 'Lower AIC'
        else:
            self.chosen_model = self.poisson_results
            comparison['chosen_model'] = 'Poisson'
            comparison['reason'] = 'Lower AIC'

        return comparison

    def print_results(self):
        """Print comprehensive results of model fitting."""
        print("--- Poisson Regression Results (Linear Trend) ---")
        print(self.poisson_results.summary())

        # Check overdispersion
        overdispersion = self.check_overdispersion()
        print(f"\nPoisson Model Dispersion Statistic (Deviance / DF): {overdispersion['dispersion_statistic']:.3f}")
        print(overdispersion['interpretation'])

        print("\n--- Negative Binomial Regression Results (Linear Trend) ---")
        print(self.nb_results.summary())

        # Compare models
        comparison = self.compare_models()
        print(f"\nPoisson AIC: {comparison['poisson_aic']:.2f}, BIC: {comparison['poisson_bic']:.2f}")
        print(f"Negative Binomial AIC: {comparison['nb_aic']:.2f}, BIC: {comparison['nb_bic']:.2f}")
        print(f"{comparison['chosen_model']} model has {comparison['reason']}, suggesting a better fit.")
        print(f"Proceeding with {comparison['chosen_model']} model for further analysis.")


def load_data(filepath):
    """Load preprocessed data from CSV file."""
    return pd.read_csv(filepath, index_col='Year')


if __name__ == "__main__":
    # Load data
    try:
        data = load_data('/home/callebalik/EnviDis/data/processed/sample_time_series_data.csv')

        # Fit models
        fitter = BasicModelFitter(data)
        fitter.fit_poisson_model()
        fitter.fit_negative_binomial_model()

        # Print results
        fitter.print_results()

    except FileNotFoundError:
        print("Sample data not found. Please run data_generation.py first.")
