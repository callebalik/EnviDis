#!/usr/bin/env python3
"""Trend Models Module
Implements polynomial and spline trend models with proper offset handling.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from envidis.time_series_analysis.modeling.base_models import GLMModelFitter

warnings.filterwarnings("ignore")


class TrendModelFitter(GLMModelFitter):
    """Trend model fitter that extends GLMModelFitter with polynomial and spline models."""

    def __init__(
        self,
        data,
        response_col="ObservedEntities",
        offset_col="TotalDocuments",
        time_var="Year_scaled",
    ) -> None:
        """Initialize trend model fitter.

        Parameters
        ----------
        data : pandas.DataFrame
            Modeling data
        response_col : str
            Name of response variable
        offset_col : str
            Name of offset variable
        time_var : str
            Name of time variable for trends

        """
        super().__init__(data, response_col, offset_col)
        self.time_var = time_var
        self.trend_models = {}

    def fit_linear_trend(self):
        """Fit linear trend model.

        Returns
        -------
        Fitted linear trend model

        """
        try:
            # Prepare data
            y = self.data[self.response_col]
            x_data = self.data[["const", self.time_var]]
            offset = self.data["log_offset"]

            # Fit model
            model = sm.Poisson(y, x_data, offset=offset)
            results = model.fit(disp=0)

            self.trend_models["linear"] = results
            print(f"✓ Linear trend model fitted (AIC: {results.aic:.2f})")
            return results

        except Exception as e:
            print(f"⚠ Linear trend model fitting failed: {e}")
            return None

    def fit_quadratic_trend(self):
        """Fit quadratic trend model.

        Returns
        -------
        Fitted quadratic trend model

        """
        try:
            # Create quadratic term
            self.data[f"{self.time_var}_squared"] = self.data[self.time_var] ** 2

            # Prepare data
            y = self.data[self.response_col]
            x_data = self.data[["const", self.time_var, f"{self.time_var}_squared"]]
            offset = self.data["log_offset"]

            # Fit model
            model = sm.Poisson(y, x_data, offset=offset)
            results = model.fit(disp=0)

            self.trend_models["quadratic"] = results
            print(f"✓ Quadratic trend model fitted (AIC: {results.aic:.2f})")
            return results

        except Exception as e:
            print(f"⚠ Quadratic trend model fitting failed: {e}")
            return None

    def fit_cubic_trend(self):
        """Fit cubic trend model.

        Returns
        -------
        Fitted cubic trend model

        """
        try:
            # Create polynomial terms
            self.data[f"{self.time_var}_squared"] = self.data[self.time_var] ** 2
            self.data[f"{self.time_var}_cubed"] = self.data[self.time_var] ** 3

            # Prepare data
            y = self.data[self.response_col]
            x_data = self.data[
                [
                    "const",
                    self.time_var,
                    f"{self.time_var}_squared",
                    f"{self.time_var}_cubed",
                ]
            ]
            offset = self.data["log_offset"]

            # Fit model
            model = sm.Poisson(y, x_data, offset=offset)
            results = model.fit(disp=0)

            self.trend_models["cubic"] = results
            print(f"✓ Cubic trend model fitted (AIC: {results.aic:.2f})")
            return results

        except Exception as e:
            print(f"⚠ Cubic trend model fitting failed: {e}")
            return None

    def fit_spline_trend(self, df=5):
        """Fit spline trend model.

        Parameters
        ----------
        df : int
            Degrees of freedom for the spline

        Returns
        -------
        Fitted spline trend model

        """
        try:
            # Create spline basis
            time_values = self.data[self.time_var].values
            spline_formula = f"bs({self.time_var}, df={df})"
            spline_basis = dmatrix(spline_formula, {"Year_scaled": time_values})

            # Convert to DataFrame and add to data
            spline_df = pd.DataFrame(
                spline_basis,
                columns=[f"spline_{i}" for i in range(spline_basis.shape[1])],
            )
            spline_df.index = self.data.index

            # Prepare data
            y = self.data[self.response_col]
            x_data = spline_df
            offset = self.data["log_offset"]

            # Fit model
            model = sm.Poisson(y, x_data, offset=offset)
            results = model.fit(disp=0)

            self.trend_models[f"spline_df{df}"] = results
            print(f"✓ Spline trend model fitted (AIC: {results.aic:.2f}, df={df})")
            return results

        except Exception as e:
            print(f"⚠ Spline trend model fitting failed: {e}")
            return None

    def fit_all_trend_models(self):
        """Fit all trend models.

        Returns
        -------
        dict
            Dictionary of fitted trend models

        """
        print("Fitting all trend models...")

        # Ensure constant term exists
        if "const" not in self.data.columns:
            self.data["const"] = 1

        # Fit all models
        self.fit_linear_trend()
        self.fit_quadratic_trend()
        self.fit_cubic_trend()
        self.fit_spline_trend(df=3)
        self.fit_spline_trend(df=5)

        return self.trend_models

    def get_best_trend_model(self):
        """Get the best trend model based on AIC.

        Returns
        -------
        tuple
            (model_name, fitted_model)

        """
        if not self.trend_models:
            return None, None

        best_aic = float("inf")
        best_name = None
        best_model = None

        for name, model in self.trend_models.items():
            if hasattr(model, "aic") and model.aic < best_aic:
                best_aic = model.aic
                best_name = name
                best_model = model

        return best_name, best_model

    def predict_trend(self, model_name, time_values=None):
        """Predict using trend model.

        Parameters
        ----------
        model_name : str
            Name of the trend model to use
        time_values : array-like, optional
            Time values for prediction. If None, uses original data.

        Returns
        -------
        array
            Predicted values

        """
        if model_name not in self.trend_models:
            print(f"Model {model_name} not found")
            return None

        model = self.trend_models[model_name]

        if time_values is None:
            # Use original data
            predictions = model.predict()
        else:
            # Create prediction data
            pred_data = pd.DataFrame({self.time_var: time_values})
            pred_data["const"] = 1

            if "squared" in model_name:
                pred_data[f"{self.time_var}_squared"] = pred_data[self.time_var] ** 2
            if "cubic" in model_name:
                pred_data[f"{self.time_var}_cubed"] = pred_data[self.time_var] ** 3

            # For spline models, would need to recreate basis
            # This is simplified for now
            predictions = model.predict(pred_data)

        return predictions

    def interpret_trend_coefficients(self, model_name):
        """Interpret trend model coefficients.

        Parameters
        ----------
        model_name : str
            Name of the trend model

        Returns
        -------
        dict
            Interpretation of coefficients

        """
        if model_name not in self.trend_models:
            return {}

        model = self.trend_models[model_name]

        interpretation = {
            "model_name": model_name,
            "aic": model.aic,
            "bic": model.bic,
            "coefficients": model.params.to_dict(),
            "p_values": model.pvalues.to_dict(),
        }

        # Add specific interpretations based on model type
        if model_name == "linear":
            if self.time_var in model.params.index:
                coef = model.params[self.time_var]
                interpretation["trend_direction"] = (
                    "increasing" if coef > 0 else "decreasing"
                )
                interpretation["percent_change_per_unit"] = (np.exp(coef) - 1) * 100

        return interpretation

    def compare_trend_models(self):
        """Compare all fitted trend models.

        Returns
        -------
        pd.DataFrame
            Comparison table of trend models

        """
        if not self.trend_models:
            return pd.DataFrame()

        comparison_data = []

        for name, model in self.trend_models.items():
            comparison_data.append(
                {
                    "Model": name,
                    "AIC": model.aic,
                    "BIC": model.bic,
                    "LogLikelihood": model.llf,
                    "N_params": len(model.params),
                },
            )

        df = pd.DataFrame(comparison_data)
        df = df.sort_values("AIC")
        df["Delta_AIC"] = df["AIC"] - df["AIC"].min()

        return df
