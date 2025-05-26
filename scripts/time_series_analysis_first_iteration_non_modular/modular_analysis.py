#!/usr/bin/env python3
"""
Modular Analysis Framework for Time Series Data
This module provides reusable analysis functions and classes for different analysis modes.
Implements proper GLM modeling with offset variables.
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings('ignore')


class ModelFitter:
    """Modular class for fitting various statistical models with proper offset handling."""

    def __init__(self, data, offset_var='TotalDocuments', response_var='ObservedEntities'):
        """
        Initialize the model fitter.

        Parameters:
        -----------
        data : pandas.DataFrame
            Data for modeling
        offset_var : str
            Name of the offset variable (e.g., 'TotalDocuments')
        response_var : str
            Name of the response variable (e.g., 'ObservedEntities')
        """
        self.data = data.copy()
        self.offset_var = offset_var
        self.response_var = response_var
        self.models = {}

        # Create log offset for GLM models
        self.data['log_offset'] = np.log(self.data[offset_var] + 1)  # Add 1 to avoid log(0)

    def fit_basic_models(self, predictors=['DaysSinceStart_scaled']):
        """
        Fit basic Poisson and Negative Binomial models with offset.

        Parameters:
        -----------
        predictors : list
            List of predictor variable names

        Returns:
        --------
        dict : Dictionary of fitted models
        """
        predictor_str = ' + '.join(predictors)

        # Sample data if too large for computational efficiency
        sample_size = min(5000, len(self.data))
        modeling_data = self.data.sample(n=sample_size, random_state=42)

        # Filter extreme values for model stability
        q95 = modeling_data[self.response_var].quantile(0.95)
        modeling_data = modeling_data[modeling_data[self.response_var] <= q95]

        try:
            # Poisson model with offset
            formula = f'{self.response_var} ~ {predictor_str}'
            self.models['poisson'] = smf.poisson(
                formula,
                data=modeling_data,
                offset=modeling_data['log_offset']
            ).fit(disp=0)

            # Negative Binomial model with offset
            self.models['negative_binomial'] = smf.negativebinomial(
                formula,
                data=modeling_data,
                offset=modeling_data['log_offset']
            ).fit(disp=0)

            print(f"   ✓ Basic models fitted with offset on {len(modeling_data)} observations")

        except Exception as e:
            print(f"   ⚠ Basic model fitting encountered issues: {e}")
            # Fallback without offset if needed
            try:
                formula_fallback = f'{self.response_var} ~ {predictor_str} + {self.offset_var}'
                self.models['poisson'] = smf.poisson(formula_fallback, data=modeling_data).fit(disp=0)
                self.models['negative_binomial'] = smf.negativebinomial(formula_fallback, data=modeling_data).fit(disp=0)
                print("   ✓ Basic models fitted without offset (fallback)")
            except Exception as e2:
                print(f"   ✗ Model fitting failed: {e2}")

        return self.models

    def fit_trend_models(self, time_var='Year_scaled'):
        """
        Fit polynomial trend models with offset.

        Parameters:
        -----------
        time_var : str
            Name of the time variable for trend modeling

        Returns:
        --------
        dict : Updated dictionary of fitted models
        """
        # Sample data for computational efficiency
        sample_size = min(5000, len(self.data))
        modeling_data = self.data.sample(n=sample_size, random_state=42)

        # Filter extreme values
        q95 = modeling_data[self.response_var].quantile(0.95)
        modeling_data = modeling_data[modeling_data[self.response_var] <= q95]

        try:
            # Linear trend with offset
            formula_linear = f'{self.response_var} ~ {time_var}'
            self.models['linear'] = smf.poisson(
                formula_linear,
                data=modeling_data,
                offset=modeling_data['log_offset']
            ).fit(disp=0)

            # Quadratic trend with offset
            formula_quad = f'{self.response_var} ~ {time_var} + I({time_var}**2)'
            self.models['quadratic'] = smf.poisson(
                formula_quad,
                data=modeling_data,
                offset=modeling_data['log_offset']
            ).fit(disp=0)

            # Cubic trend with offset
            formula_cubic = f'{self.response_var} ~ {time_var} + I({time_var}**2) + I({time_var}**3)'
            self.models['cubic'] = smf.poisson(
                formula_cubic,
                data=modeling_data,
                offset=modeling_data['log_offset']
            ).fit(disp=0)

            print(f"   ✓ Trend models fitted with offset on {len(modeling_data)} observations")

        except Exception as e:
            print(f"   ⚠ Trend model fitting encountered issues: {e}")
            # Fallback without offset
            try:
                formula_linear = f'{self.response_var} ~ {time_var} + {self.offset_var}'
                formula_quad = f'{self.response_var} ~ {time_var} + I({time_var}**2) + {self.offset_var}'
                formula_cubic = f'{self.response_var} ~ {time_var} + I({time_var}**2) + I({time_var}**3) + {self.offset_var}'

                self.models['linear'] = smf.poisson(formula_linear, data=modeling_data).fit(disp=0)
                self.models['quadratic'] = smf.poisson(formula_quad, data=modeling_data).fit(disp=0)
                self.models['cubic'] = smf.poisson(formula_cubic, data=modeling_data).fit(disp=0)
                print("   ✓ Trend models fitted without offset (fallback)")
            except Exception as e2:
                print(f"   ✗ Trend model fitting failed: {e2}")

        return self.models

    def get_model_comparison(self):
        """
        Compare fitted models using AIC, BIC, and other metrics.

        Returns:
        --------
        pandas.DataFrame : Model comparison table
        """
        if not self.models:
            print("No models fitted yet. Run fit_basic_models() or fit_trend_models() first.")
            return pd.DataFrame()

        comparison_data = []

        for name, model in self.models.items():
            try:
                # Calculate metrics
                aic = model.aic
                bic = model.bic

                # Pseudo R-squared (McFadden's)
                ll_null = model.llnull if hasattr(model, 'llnull') else np.nan
                ll_model = model.llf
                pseudo_r2 = 1 - (ll_model / ll_null) if not np.isnan(ll_null) else np.nan

                # Number of parameters
                n_params = len(model.params)

                # Deviance
                deviance = model.deviance if hasattr(model, 'deviance') else np.nan

                comparison_data.append({
                    'Model': name.replace('_', ' ').title(),
                    'AIC': aic,
                    'BIC': bic,
                    'Pseudo_R2': pseudo_r2,
                    'N_Params': n_params,
                    'Deviance': deviance,
                    'Log_Likelihood': ll_model
                })

            except Exception as e:
                print(f"Error calculating metrics for {name}: {e}")
                continue

        df = pd.DataFrame(comparison_data)

        # Sort by AIC (lower is better)
        if not df.empty:
            df = df.sort_values('AIC')

            # Add ranking
            df['AIC_Rank'] = range(1, len(df) + 1)

        return df


class TemporalAnalyzer:
    """Class for analyzing temporal patterns and seasonality."""

    def __init__(self, data, date_col=None):
        """
        Initialize temporal analyzer.

        Parameters:
        -----------
        data : pandas.DataFrame
            Data with datetime index or date column
        date_col : str, optional
            Name of date column if not using index
        """
        self.data = data.copy()

        if date_col is not None:
            self.data.index = pd.to_datetime(self.data[date_col])
        elif not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex or provide date_col")

    def get_monthly_patterns(self, value_cols=['ObservedEntities', 'TotalDocuments']):
        """
        Analyze monthly patterns in the data.

        Parameters:
        -----------
        value_cols : list
            Columns to aggregate by month

        Returns:
        --------
        pandas.DataFrame : Monthly aggregated data
        """
        agg_dict = {col: 'sum' for col in value_cols}
        monthly_data = self.data.groupby([self.data.index.year, self.data.index.month]).agg(agg_dict)
        monthly_data.index.names = ['Year', 'Month']
        monthly_data = monthly_data.reset_index()

        # Create date for plotting
        monthly_data['Date'] = pd.to_datetime(monthly_data[['Year', 'Month']].assign(day=1))

        return monthly_data

    def get_yearly_patterns(self, value_cols=['ObservedEntities', 'TotalDocuments']):
        """
        Analyze yearly patterns in the data.

        Parameters:
        -----------
        value_cols : list
            Columns to aggregate by year

        Returns:
        --------
        pandas.DataFrame : Yearly aggregated data
        """
        agg_dict = {col: 'sum' for col in value_cols}
        yearly_data = self.data.groupby(self.data.index.year).agg(agg_dict)
        yearly_data.index.name = 'Year'
        yearly_data = yearly_data.reset_index()

        return yearly_data

    def detect_seasonality(self, value_col='ObservedEntities'):
        """
        Detect seasonal patterns using statistical tests.

        Parameters:
        -----------
        value_col : str
            Column to test for seasonality

        Returns:
        --------
        dict : Seasonality analysis results
        """
        try:
            # Monthly seasonality test
            monthly_means = self.data.groupby(self.data.index.month)[value_col].mean()

            # Kruskal-Wallis test for monthly differences
            monthly_groups = [self.data[self.data.index.month == month][value_col].values
                            for month in range(1, 13)]
            monthly_groups = [group for group in monthly_groups if len(group) > 0]

            if len(monthly_groups) > 1:
                kw_stat, kw_p = stats.kruskal(*monthly_groups)
            else:
                kw_stat, kw_p = np.nan, np.nan

            # Coefficient of variation for monthly means
            monthly_cv = monthly_means.std() / monthly_means.mean() if monthly_means.mean() > 0 else np.nan

            results = {
                'monthly_kruskal_stat': kw_stat,
                'monthly_kruskal_p': kw_p,
                'monthly_cv': monthly_cv,
                'has_monthly_pattern': kw_p < 0.05 if not np.isnan(kw_p) else False,
                'monthly_means': monthly_means.to_dict()
            }

            return results

        except Exception as e:
            print(f"Error in seasonality detection: {e}")
            return {'error': str(e)}


class DiagnosticAnalyzer:
    """Class for model diagnostics and validation."""

    def __init__(self, models, data):
        """
        Initialize diagnostic analyzer.

        Parameters:
        -----------
        models : dict
            Dictionary of fitted models
        data : pandas.DataFrame
            Original data used for modeling
        """
        self.models = models
        self.data = data

    def calculate_residuals(self, model_name):
        """
        Calculate various types of residuals for a model.

        Parameters:
        -----------
        model_name : str
            Name of the model to analyze

        Returns:
        --------
        dict : Dictionary of residual arrays
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]

        try:
            # Get fitted values and residuals
            fitted = model.fittedvalues
            residuals = model.resid

            # Pearson residuals
            pearson_resid = model.resid_pearson if hasattr(model, 'resid_pearson') else residuals

            # Deviance residuals
            deviance_resid = model.resid_deviance if hasattr(model, 'resid_deviance') else residuals

            return {
                'fitted': fitted,
                'residuals': residuals,
                'pearson': pearson_resid,
                'deviance': deviance_resid
            }

        except Exception as e:
            print(f"Error calculating residuals for {model_name}: {e}")
            return {}

    def check_zero_inflation(self, response_var='ObservedEntities'):
        """
        Check for zero-inflation in the response variable.

        Parameters:
        -----------
        response_var : str
            Name of the response variable

        Returns:
        --------
        dict : Zero-inflation analysis results
        """
        observed_zeros = (self.data[response_var] == 0).sum()
        total_obs = len(self.data)
        observed_zero_prop = observed_zeros / total_obs

        # Expected zeros under Poisson distribution
        mean_rate = self.data[response_var].mean()
        expected_zero_prop = np.exp(-mean_rate)

        # Excess zeros
        excess_zeros = observed_zero_prop - expected_zero_prop

        return {
            'observed_zeros': observed_zeros,
            'total_observations': total_obs,
            'observed_zero_proportion': observed_zero_prop,
            'expected_zero_proportion': expected_zero_prop,
            'excess_zero_proportion': excess_zeros,
            'is_zero_inflated': excess_zeros > 0.1  # Threshold for considering zero-inflation
        }

    def model_diagnostics_summary(self):
        """
        Generate comprehensive diagnostics summary for all models.

        Returns:
        --------
        dict : Summary of diagnostics for all models
        """
        summary = {}

        # Zero-inflation check
        summary['zero_inflation'] = self.check_zero_inflation()

        # Model-specific diagnostics
        summary['models'] = {}

        for model_name, model in self.models.items():
            try:
                residuals = self.calculate_residuals(model_name)

                if residuals:
                    # Basic residual statistics
                    summary['models'][model_name] = {
                        'residual_mean': np.mean(residuals['residuals']),
                        'residual_std': np.std(residuals['residuals']),
                        'residual_min': np.min(residuals['residuals']),
                        'residual_max': np.max(residuals['residuals']),
                        'fitted_min': np.min(residuals['fitted']),
                        'fitted_max': np.max(residuals['fitted'])
                    }

            except Exception as e:
                print(f"Error in diagnostics for {model_name}: {e}")
                summary['models'][model_name] = {'error': str(e)}

        return summary


class ResultsExporter:
    """Class for exporting analysis results to files."""

    def __init__(self, output_dir):
        """
        Initialize results exporter.

        Parameters:
        -----------
        output_dir : str
            Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_model_comparison(self, comparison_df, filename='model_comparison.csv'):
        """Export model comparison table to CSV."""
        filepath = os.path.join(self.output_dir, filename)
        comparison_df.to_csv(filepath, index=False)
        print(f"Model comparison exported to: {filepath}")

    def export_coefficients(self, models, filename='model_coefficients.csv'):
        """Export model coefficients to CSV."""
        coeff_data = []

        for model_name, model in models.items():
            try:
                for param, coeff in model.params.items():
                    pvalue = model.pvalues[param] if hasattr(model, 'pvalues') else np.nan
                    stderr = model.bse[param] if hasattr(model, 'bse') else np.nan

                    coeff_data.append({
                        'Model': model_name,
                        'Parameter': param,
                        'Coefficient': coeff,
                        'Std_Error': stderr,
                        'P_Value': pvalue,
                        'Significant': pvalue < 0.05 if not np.isnan(pvalue) else False
                    })
            except Exception as e:
                print(f"Error exporting coefficients for {model_name}: {e}")

        if coeff_data:
            coeff_df = pd.DataFrame(coeff_data)
            filepath = os.path.join(self.output_dir, filename)
            coeff_df.to_csv(filepath, index=False)
            print(f"Model coefficients exported to: {filepath}")

    def export_diagnostics(self, diagnostics, filename='model_diagnostics.txt'):
        """Export model diagnostics to text file."""
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            f.write("MODEL DIAGNOSTICS SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            # Zero-inflation
            if 'zero_inflation' in diagnostics:
                zi = diagnostics['zero_inflation']
                f.write("ZERO-INFLATION ANALYSIS:\n")
                f.write(f"  Total observations: {zi.get('total_observations', 'N/A')}\n")
                f.write(f"  Observed zeros: {zi.get('observed_zeros', 'N/A')}\n")
                f.write(f"  Observed zero proportion: {zi.get('observed_zero_proportion', 'N/A'):.3f}\n")
                f.write(f"  Expected zero proportion: {zi.get('expected_zero_proportion', 'N/A'):.3f}\n")
                f.write(f"  Excess zero proportion: {zi.get('excess_zero_proportion', 'N/A'):.3f}\n")
                f.write(f"  Zero-inflated: {zi.get('is_zero_inflated', 'N/A')}\n\n")

            # Model diagnostics
            if 'models' in diagnostics:
                f.write("MODEL-SPECIFIC DIAGNOSTICS:\n")
                for model_name, diag in diagnostics['models'].items():
                    f.write(f"\n{model_name.upper()}:\n")
                    for key, value in diag.items():
                        f.write(f"  {key}: {value}\n")

        print(f"Model diagnostics exported to: {filepath}")


# Convenience function for complete analysis
def run_complete_analysis(data, output_dir, offset_var='TotalDocuments', response_var='ObservedEntities'):
    """
    Run complete modular analysis pipeline.

    Parameters:
    -----------
    data : pandas.DataFrame
        Input data for analysis
    output_dir : str
        Directory to save results
    offset_var : str
        Name of offset variable
    response_var : str
        Name of response variable

    Returns:
    --------
    dict : Complete analysis results
    """
    print("Starting complete modular analysis...")

    # Initialize components
    fitter = ModelFitter(data, offset_var=offset_var, response_var=response_var)
    temporal = TemporalAnalyzer(data)
    exporter = ResultsExporter(output_dir)

    # Fit models
    print("1. Fitting basic models...")
    fitter.fit_basic_models()

    print("2. Fitting trend models...")
    fitter.fit_trend_models()

    # Model comparison
    print("3. Comparing models...")
    comparison = fitter.get_model_comparison()

    # Temporal analysis
    print("4. Analyzing temporal patterns...")
    monthly_data = temporal.get_monthly_patterns()
    yearly_data = temporal.get_yearly_patterns()
    seasonality = temporal.detect_seasonality()

    # Diagnostics
    print("5. Running diagnostics...")
    diagnostics = DiagnosticAnalyzer(fitter.models, data)
    diag_summary = diagnostics.model_diagnostics_summary()

    # Export results
    print("6. Exporting results...")
    exporter.export_model_comparison(comparison)
    exporter.export_coefficients(fitter.models)
    exporter.export_diagnostics(diag_summary)

    results = {
        'models': fitter.models,
        'comparison': comparison,
        'monthly_data': monthly_data,
        'yearly_data': yearly_data,
        'seasonality': seasonality,
        'diagnostics': diag_summary,
        'fitter': fitter,
        'temporal': temporal,
        'diagnostics_analyzer': diagnostics
    }

    print("Complete modular analysis finished!")
    return results
