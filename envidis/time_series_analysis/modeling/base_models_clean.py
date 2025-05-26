#!/usr/bin/env python3
"""Base GLM models with proper offset handling."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson
from statsmodels.discrete.discrete_model import NegativeBinomial


class GLMModelFitter:
    """Base GLM model fitter with proper offset handling.

    Uses TotalDocuments as offset variable rather than predictor.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        response_col: str = "ObservedEntities",
        offset_col: str = "TotalDocuments",
    ) -> None:
        """Initialize the GLM model fitter.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        response_col : str
            Name of response variable column
        offset_col : str
            Name of offset variable column

        """
        self.data = data.copy()
        self.response_col = response_col
        self.offset_col = offset_col
        self.fitted_models = {}

        # Create log offset variable
        self.data["log_offset"] = np.log(self.data[offset_col] + 1)

    def fit_poisson_model(
        self,
        predictors: list[str] | None = None,
        model_name: str = "poisson",
    ):
        """Fit a Poisson GLM model.

        Parameters
        ----------
        predictors : list[str] | None
            List of predictor variable names
        model_name : str
            Name to store the model under

        Returns
        -------
        Fitted model or None if fitting fails

        """
        if predictors is None:
            predictors = ["Year_scaled"]

        try:
            y = self.data[self.response_col]
            x_data = self.data[predictors]
            offset = self.data["log_offset"]

            model = sm.Poisson(y, x_data, offset=offset)
            fitted_model = model.fit(disp=0)

            self.fitted_models[model_name] = fitted_model
            return fitted_model

        except Exception as e:
            print(f"Error fitting Poisson model: {e}")
            return None

    def fit_negative_binomial_model(
        self,
        predictors: list[str] | None = None,
        model_name: str = "negative_binomial",
    ):
        """Fit a Negative Binomial GLM model.

        Parameters
        ----------
        predictors : list[str] | None
            List of predictor variable names
        model_name : str
            Name to store the model under

        Returns
        -------
        Fitted model or None if fitting fails

        """
        if predictors is None:
            predictors = ["Year_scaled"]

        try:
            y = self.data[self.response_col]
            x_data = self.data[predictors]
            offset = self.data["log_offset"]

            model = NegativeBinomial(y, x_data, offset=offset)
            fitted_model = model.fit(disp=0)

            self.fitted_models[model_name] = fitted_model
            return fitted_model

        except Exception as e:
            print(f"Error fitting Negative Binomial model: {e}")
            return None

    def fit_zero_inflated_poisson(
        self,
        predictors: list[str] | None = None,
        zi_predictors: list[str] | None = None,
        model_name: str = "zero_inflated_poisson",
    ):
        """Fit a Zero-Inflated Poisson model.

        Parameters
        ----------
        predictors : list[str] | None
            List of predictor variable names
        zi_predictors : list[str] | None
            Predictors for zero-inflation component
        model_name : str
            Name to store the model under

        Returns
        -------
        Fitted model or None if fitting fails

        """
        if predictors is None:
            predictors = ["Year_scaled"]

        try:
            y = self.data[self.response_col]
            x_data = self.data[predictors]
            offset = self.data["log_offset"]

            if zi_predictors is None:
                zi_predictors = ["const"]

            zi_x_data = self.data[zi_predictors]

            model = ZeroInflatedPoisson(y, x_data, exog_infl=zi_x_data, offset=offset)
            fitted_model = model.fit(disp=0)

            self.fitted_models[model_name] = fitted_model
            return fitted_model

        except Exception as e:
            print(f"Error fitting Zero-Inflated Poisson model: {e}")
            return None

    def fit_hurdle_model(
        self,
        predictors: list[str] | None = None,
        hurdle_predictors: list[str] | None = None,
        model_name: str = "hurdle",
    ):
        """Fit a hurdle model (two-part model).

        Parameters
        ----------
        predictors : list[str] | None
            List of predictor variable names for count part
        hurdle_predictors : list[str] | None
            Predictors for hurdle (binary) part
        model_name : str
            Name to store the model under

        Returns
        -------
        Dict with both model parts or None if fitting fails

        """
        if predictors is None:
            predictors = ["Year_scaled"]

        try:
            y = self.data[self.response_col]
            x_data = self.data[predictors]

            if hurdle_predictors is None:
                hurdle_predictors = predictors

            hurdle_x_data = self.data[hurdle_predictors]

            # Binary part (whether count > 0)
            y_binary = (y > 0).astype(int)
            binary_model = sm.Logit(y_binary, hurdle_x_data)
            binary_fitted = binary_model.fit(disp=0)

            # Count part (count | count > 0)
            y_positive = y[y > 0]
            x_positive = x_data.loc[y > 0]
            offset_positive = self.data.loc[y > 0, "log_offset"]

            count_model = sm.Poisson(y_positive, x_positive, offset=offset_positive)
            count_fitted = count_model.fit(disp=0)

            hurdle_model = {
                "binary_part": binary_fitted,
                "count_part": count_fitted,
                "model_type": "hurdle",
            }

            self.fitted_models[model_name] = hurdle_model
            return hurdle_model

        except Exception as e:
            print(f"Error fitting hurdle model: {e}")
            return None

    def fit_all_models(self, predictors: list[str] | None = None):
        """Fit all available models.

        Parameters
        ----------
        predictors : list[str] | None
            List of predictor variable names

        Returns
        -------
        dict
            Dictionary of fitted models

        """
        if predictors is None:
            predictors = ["Year_scaled"]

        print("Fitting all models...")

        # Add constant term if not present
        if "const" not in predictors and "const" not in self.data.columns:
            self.data["const"] = 1
            predictors = ["const"] + predictors
        elif "const" not in predictors:
            predictors = ["const"] + predictors

        self.fit_poisson_model(predictors)
        self.fit_negative_binomial_model(predictors)
        self.fit_zero_inflated_poisson(predictors)
        self.fit_hurdle_model(predictors)

        return self.fitted_models

    def get_model_summary(self, model_name: str):
        """Get summary statistics for a fitted model.

        Parameters
        ----------
        model_name : str
            Name of the fitted model

        Returns
        -------
        dict
            Model summary statistics

        """
        if model_name not in self.fitted_models:
            return {}

        model = self.fitted_models[model_name]

        if isinstance(model, dict) and model.get("model_type") == "hurdle":
            # Handle hurdle model
            return {
                "model_type": "hurdle",
                "binary_aic": model["binary_part"].aic,
                "count_aic": model["count_part"].aic,
                "combined_aic": model["binary_part"].aic + model["count_part"].aic,
                "binary_params": model["binary_part"].params,
                "count_params": model["count_part"].params,
            }
        else:
            # Handle regular models
            return {
                "aic": model.aic,
                "bic": model.bic,
                "llf": model.llf,
                "params": model.params,
                "pvalues": model.pvalues,
                "conf_int": model.conf_int(),
            }

    def compare_models(self) -> pd.DataFrame:
        """Compare fitted models using information criteria.

        Returns
        -------
        pd.DataFrame
            Model comparison table

        """
        if not self.fitted_models:
            return pd.DataFrame()

        comparison_data = []

        for name, model in self.fitted_models.items():
            if isinstance(model, dict) and model.get("model_type") == "hurdle":
                aic = model["binary_part"].aic + model["count_part"].aic
                bic = model["binary_part"].bic + model["count_part"].bic
                llf = model["binary_part"].llf + model["count_part"].llf
            else:
                aic = model.aic
                bic = model.bic
                llf = model.llf

            comparison_data.append(
                {
                    "Model": name,
                    "AIC": aic,
                    "BIC": bic,
                    "LogLikelihood": llf,
                    "Delta_AIC": 0,  # Will be calculated after
                }
            )

        df = pd.DataFrame(comparison_data)
        df = df.sort_values("AIC")
        df["Delta_AIC"] = df["AIC"] - df["AIC"].min()

        return df

    def get_best_model(self, criterion: str = "aic"):
        """Get the best model based on specified criterion.

        Parameters
        ----------
        criterion : str
            Criterion for model selection ('aic' or 'bic')

        Returns
        -------
        tuple
            (model_name, fitted_model)

        """
        if not self.fitted_models:
            return None, None

        best_score = float("inf")
        best_name = None
        best_model = None

        for name, model in self.fitted_models.items():
            if isinstance(model, dict) and model.get("model_type") == "hurdle":
                if criterion.lower() == "aic":
                    score = model["binary_part"].aic + model["count_part"].aic
                else:
                    score = model["binary_part"].bic + model["count_part"].bic
            else:
                score = getattr(model, criterion.lower(), float("inf"))

            if score < best_score:
                best_score = score
                best_name = name
                best_model = model

        return best_name, best_model
