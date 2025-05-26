"""Autocorrelation Analysis Script
This script handles autocorrelation analysis and lagged variable modeling.
"""

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf


class AutocorrelationAnalyzer:
    """Class for analyzing autocorrelation in model residuals."""

    def __init__(self, model_results):
        """Initialize with fitted model results.

        Parameters
        ----------
        model_results : statsmodels results object
            Fitted model results

        """
        self.model_results = model_results
        self.residuals = model_results.resid_pearson
        self.dw_statistic = None
        self.acf_values = None

    def plot_acf_residuals(self, lags=None, save_path=None) -> None:
        """Plot autocorrelation function of residuals.

        Parameters
        ----------
        lags : int, optional
            Number of lags to plot
        save_path : str, optional
            Path to save the plot

        """
        if lags is None:
            lags = min(20, len(self.residuals) // 2 - 1)

        plot_acf(self.residuals, lags=lags, alpha=0.05)
        plt.title("Autocorrelation Function (ACF) of Model Residuals")
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

    def durbin_watson_test(self):
        """Perform Durbin-Watson test for first-order autocorrelation.

        Returns
        -------
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
            "dw_statistic": self.dw_statistic,
            "significant_autocorrelation": significant,
            "interpretation": interpretation,
        }

    def calculate_acf(self, nlags=20):
        """Calculate autocorrelation function values.

        Parameters
        ----------
        nlags : int
            Number of lags to calculate

        Returns
        -------
        numpy.ndarray
            ACF values

        """
        self.acf_values = acf(self.residuals, nlags=nlags)
        return self.acf_values

    def print_diagnostics(self) -> None:
        """Print autocorrelation diagnostic results."""
        dw_results = self.durbin_watson_test()
        print(f"Durbin-Watson Statistic: {dw_results['dw_statistic']:.3f}")
        print(f"Interpretation: {dw_results['interpretation']}")

        if dw_results["significant_autocorrelation"]:
            print("Warning: Significant autocorrelation detected in residuals.")
            print("Consider including lagged dependent variables in the model.")

    def plot_fitted_vs_actual(self, data, save_path=None) -> None:
        """Plot fitted values vs actual data over time.

        Parameters
        ----------
        data : pd.DataFrame
            Original data with Year as index and co_count column
        save_path : str, optional
            Path to save the plot

        """
        # Get fitted values
        fitted_values = self.model_results.fittedvalues

        # Align actual values with fitted values using the same index
        actual_values = data.loc[fitted_values.index, "co_count"]
        years = fitted_values.index

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Plot actual data
        plt.plot(
            years,
            actual_values,
            "o-",
            label="Actual Data",
            color="blue",
            markersize=6,
            linewidth=2,
        )

        # Plot fitted values
        plt.plot(
            years,
            fitted_values,
            "s-",
            label="Fitted Model",
            color="red",
            markersize=4,
            linewidth=2,
            alpha=0.8,
        )

        # Add confidence intervals if available
        try:
            # Get prediction intervals
            predictions = self.model_results.get_prediction()
            conf_int = predictions.conf_int()

            # Handle both numpy arrays and pandas DataFrames
            if hasattr(conf_int, "iloc"):
                # pandas DataFrame
                plt.fill_between(
                    years,
                    conf_int.iloc[:, 0],
                    conf_int.iloc[:, 1],
                    alpha=0.2,
                    color="red",
                    label="95% Confidence Interval",
                )
            else:
                # numpy array
                plt.fill_between(
                    years,
                    conf_int[:, 0],
                    conf_int[:, 1],
                    alpha=0.2,
                    color="red",
                    label="95% Confidence Interval",
                )
        except Exception:
            # If confidence intervals are not available, continue without them
            pass

        plt.title("Fitted Model vs Actual Data Over Time")
        plt.xlabel("Year")
        plt.ylabel("Identified DIS-PNM co-occurrences")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add model summary text
        r_squared = getattr(self.model_results, "rsquared", "N/A")
        aic = self.model_results.aic
        plt.text(
            0.02,
            0.98,
            f"AIC: {aic:.2f}\nPseudo R²: {r_squared}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

    def plot_residuals_vs_fitted(self, save_path=None) -> None:
        """Plot residuals vs fitted values to check for patterns.

        Parameters
        ----------
        save_path : str, optional
            Path to save the plot

        """
        fitted_values = self.model_results.fittedvalues
        residuals = self.residuals

        plt.figure(figsize=(10, 6))
        plt.scatter(fitted_values, residuals, alpha=0.6)
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.8)
        plt.xlabel("Fitted Values")
        plt.ylabel("Pearson Residuals")
        plt.title("Residuals vs Fitted Values")
        plt.grid(True, alpha=0.3)

        # Add a lowess smoother to show trends
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            smoothed = lowess(residuals, fitted_values, frac=0.3)
            plt.plot(
                smoothed[:, 0],
                smoothed[:, 1],
                color="orange",
                linewidth=2,
                label="LOWESS Smoother",
            )
            plt.legend()
        except ImportError:
            pass

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

    def create_diagnostic_plots(self, data, output_dir=None) -> None:
        """Create a comprehensive set of diagnostic plots.

        Parameters
        ----------
        data : pd.DataFrame
            Original data
        output_dir : str, optional
            Directory to save plots

        """
        if output_dir:
            import os

            os.makedirs(output_dir, exist_ok=True)

            # Plot fitted vs actual
            self.plot_fitted_vs_actual(data, f"{output_dir}/fitted_vs_actual.png")

            # Plot residuals vs fitted
            self.plot_residuals_vs_fitted(f"{output_dir}/residuals_vs_fitted.png")

            # Plot ACF of residuals
            self.plot_acf_residuals(save_path=f"{output_dir}/acf_residuals.png")
        else:
            # Show all plots
            self.plot_fitted_vs_actual(data)
            self.plot_residuals_vs_fitted()
            self.plot_acf_residuals()


class LaggedModelFitter:
    """Class for fitting models with lagged dependent variables."""

    def __init__(self, data, base_model_results):
        """Initialize with data and base model results.

        Parameters
        ----------
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
        """Create lagged versions of the dependent variable.

        Parameters
        ----------
        lag_periods : int or list
            Number of lag periods or list of lag periods

        """
        if isinstance(lag_periods, int):
            lag_periods = [lag_periods]

        self.lagged_data = self.data.copy()

        for lag in lag_periods:
            lag_col = f"co_count_lag{lag}"
            self.lagged_data[lag_col] = self.lagged_data["co_count"].shift(lag)

        # Drop rows with NaN values
        self.lagged_data = self.lagged_data.dropna()

        return self.lagged_data

    def fit_lagged_model(self, lag_periods=1):
        """Fit model with lagged dependent variables.

        Parameters
        ----------
        lag_periods : int or list
            Lag periods to include

        """
        if self.lagged_data is None:
            self.create_lagged_data(lag_periods)

        # Construct formula with lags
        base_formula = self.base_model_results.model.formula

        if isinstance(lag_periods, int):
            lag_periods = [lag_periods]

        lag_terms = [f"co_count_lag{lag}" for lag in lag_periods]
        lagged_formula = base_formula + " + " + " + ".join(lag_terms)

        # Fit model
        lagged_model = smf.glm(
            formula=lagged_formula,
            data=self.lagged_data,
            family=self.base_model_results.model.family,
        )
        self.lagged_results = lagged_model.fit()

        return self.lagged_results

    def analyze_lagged_residuals(self, save_path=None):
        """Analyze residuals of the lagged model.

        Parameters
        ----------
        save_path : str, optional
            Path to save the ACF plot

        """
        if self.lagged_results is None:
            msg = "Lagged model must be fitted first"
            raise ValueError(msg)

        analyzer = AutocorrelationAnalyzer(self.lagged_results)
        analyzer.plot_acf_residuals(save_path=save_path)
        analyzer.print_diagnostics()

        return analyzer

    def print_results(self) -> None:
        """Print lagged model results."""
        if self.lagged_results is None:
            msg = "Lagged model must be fitted first"
            raise ValueError(msg)

        print("\n--- Regression Results with Lagged Dependent Variable ---")
        print(self.lagged_results.summary())

    def plot_lagged_model_fit(self, save_path=None) -> None:
        """Plot fitted values of lagged model vs actual data.

        Parameters
        ----------
        save_path : str, optional
            Path to save the plot

        """
        if self.lagged_results is None:
            msg = "Lagged model must be fitted first"
            raise ValueError(msg)

        # Get data for plotting (excluding NaN values from lagging)
        plot_data = self.lagged_data.dropna()
        fitted_values = self.lagged_results.fittedvalues
        actual_values = plot_data["co_count"]
        years = plot_data.index

        plt.figure(figsize=(12, 8))

        # Plot actual data
        plt.plot(
            years,
            actual_values,
            "o-",
            label="Actual Data",
            color="blue",
            markersize=6,
            linewidth=2,
        )

        # Plot fitted values from lagged model
        plt.plot(
            years,
            fitted_values,
            "s-",
            label="Lagged Model Fitted",
            color="green",
            markersize=4,
            linewidth=2,
            alpha=0.8,
        )

        plt.title("Lagged Model: Fitted vs Actual Data Over Time")
        plt.xlabel("Year")
        plt.ylabel("Identified DIS-PNM co-occurrences")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add model summary text
        aic = self.lagged_results.aic
        plt.text(
            0.02,
            0.98,
            f"Lagged Model AIC: {aic:.2f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

    def compare_models_plot(self, save_path=None) -> None:
        """Compare base model vs lagged model visually.

        Parameters
        ----------
        save_path : str, optional
            Path to save the plot

        """
        if self.lagged_results is None:
            msg = "Lagged model must be fitted first"
            raise ValueError(msg)

        # Get data for plotting
        plot_data = self.lagged_data.dropna()

        # Get fitted values from both models
        base_fitted = self.base_model_results.fittedvalues
        lagged_fitted = self.lagged_results.fittedvalues
        actual_values = plot_data["co_count"]
        years = plot_data.index

        # Align base model fitted values with lagged data
        base_fitted_aligned = base_fitted.loc[years]

        plt.figure(figsize=(14, 8))

        # Plot actual data
        plt.plot(
            years,
            actual_values,
            "o-",
            label="Actual Data",
            color="blue",
            markersize=6,
            linewidth=2,
        )

        # Plot base model fitted values
        plt.plot(
            years,
            base_fitted_aligned,
            "s-",
            label="Base Model Fitted",
            color="red",
            markersize=4,
            linewidth=2,
            alpha=0.7,
        )

        # Plot lagged model fitted values
        plt.plot(
            years,
            lagged_fitted,
            "^-",
            label="Lagged Model Fitted",
            color="green",
            markersize=4,
            linewidth=2,
            alpha=0.7,
        )

        plt.title("Model Comparison: Base vs Lagged Model")
        plt.xlabel("Year")
        plt.ylabel("Identified DIS-PNM co-occurrences")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add comparison statistics
        base_aic = self.base_model_results.aic
        lagged_aic = self.lagged_results.aic
        improvement = base_aic - lagged_aic
        plt.text(
            0.02,
            0.92,
            f"Base Model AIC: {base_aic:.2f}\nLagged Model AIC: {lagged_aic:.2f}\nImprovement: {improvement:.2f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
