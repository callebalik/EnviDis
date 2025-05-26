#!/usr/bin/env python3
"""Modeling utilities for time series analysis."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def create_offset_variable(
    data: pd.DataFrame,
    offset_col: str = "TotalDocuments",
) -> pd.DataFrame:
    """Create log offset variable for GLM modeling.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    offset_col : str
        Name of column to use as offset

    Returns
    -------
    pd.DataFrame
        Data with log_offset column added

    """
    data_copy = data.copy()
    data_copy["log_offset"] = np.log(data_copy[offset_col] + 1)
    return data_copy


def prepare_modeling_data(
    data: pd.DataFrame,
    response_col: str = "ObservedEntities",
    offset_col: str = "TotalDocuments",
    predictors: list = None,
) -> dict[str, Any]:
    """Prepare data for GLM modeling with proper variable transformations.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    response_col : str
        Name of response variable
    offset_col : str
        Name of offset variable
    predictors : list
        List of predictor variables

    Returns
    -------
    dict
        Dictionary containing prepared modeling components

    """
    if predictors is None:
        predictors = ["Year_scaled"]

    # Create working copy
    modeling_data = data.copy()

    # Create offset variable
    modeling_data = create_offset_variable(modeling_data, offset_col)

    # Extract components
    y = modeling_data[response_col]
    X = modeling_data[predictors]
    offset = modeling_data["log_offset"]

    return {
        "data": modeling_data,
        "y": y,
        "X": X,
        "offset": offset,
        "response_col": response_col,
        "offset_col": offset_col,
        "predictors": predictors,
    }


def calculate_model_diagnostics(fitted_model, data: pd.DataFrame) -> dict[str, float]:
    """Calculate comprehensive model diagnostics.

    Parameters
    ----------
    fitted_model : statsmodels fitted model
        Fitted GLM model
    data : pd.DataFrame
        Original data

    Returns
    -------
    dict
        Dictionary of diagnostic statistics

    """
    diagnostics = {}

    try:
        # Basic model statistics
        diagnostics["aic"] = fitted_model.aic
        diagnostics["bic"] = fitted_model.bic
        diagnostics["log_likelihood"] = fitted_model.llf
        diagnostics["n_obs"] = fitted_model.nobs

        # Pseudo R-squared
        if hasattr(fitted_model, "pseudo_rsquared"):
            diagnostics["pseudo_r2"] = fitted_model.pseudo_rsquared()
        else:
            # Calculate McFadden's pseudo R-squared
            null_ll = fitted_model.llnull if hasattr(fitted_model, "llnull") else None
            if null_ll is not None:
                diagnostics["pseudo_r2"] = 1 - (fitted_model.llf / null_ll)
            else:
                diagnostics["pseudo_r2"] = np.nan

        # Convergence info
        diagnostics["converged"] = (
            fitted_model.mle_retvals["converged"]
            if hasattr(fitted_model, "mle_retvals")
            else True
        )

        # Residual diagnostics
        if hasattr(fitted_model, "resid_pearson"):
            residuals = fitted_model.resid_pearson
            diagnostics["residual_mean"] = np.mean(residuals)
            diagnostics["residual_std"] = np.std(residuals)
            diagnostics["residual_skew"] = float(pd.Series(residuals).skew())
            diagnostics["residual_kurt"] = float(pd.Series(residuals).kurtosis())

        # Overdispersion test (for Poisson)
        if (
            hasattr(fitted_model, "resid_pearson")
            and fitted_model.family.__class__.__name__ == "Poisson"
        ):
            pearson_chi2 = np.sum(fitted_model.resid_pearson**2)
            df_resid = fitted_model.df_resid
            overdispersion = pearson_chi2 / df_resid
            diagnostics["overdispersion"] = overdispersion
            diagnostics["overdispersion_pvalue"] = 1 - stats.chi2.cdf(
                pearson_chi2,
                df_resid,
            )

    except Exception as e:
        print(f"Warning: Could not calculate some diagnostics: {e}")

    return diagnostics


def compare_models(models: dict[str, Any]) -> pd.DataFrame:
    """Compare multiple fitted models.

    Parameters
    ----------
    models : dict
        Dictionary of model_name: fitted_model pairs

    Returns
    -------
    pd.DataFrame
        Comparison table with model statistics

    """
    comparison_data = []

    for name, model in models.items():
        if model is None:
            continue

        try:
            row = {"Model": name}
            diagnostics = calculate_model_diagnostics(model, None)
            row.update(diagnostics)
            comparison_data.append(row)
        except Exception as e:
            print(f"Warning: Could not extract statistics for {name}: {e}")

    if not comparison_data:
        return pd.DataFrame()

    df = pd.DataFrame(comparison_data)

    # Sort by AIC (lower is better)
    if "aic" in df.columns:
        df = df.sort_values("aic")

    return df


def extract_coefficients_table(fitted_model, model_name: str = None) -> pd.DataFrame:
    """Extract coefficients table from fitted model.

    Parameters
    ----------
    fitted_model : statsmodels fitted model
        Fitted GLM model
    model_name : str
        Name of the model

    Returns
    -------
    pd.DataFrame
        Coefficients table with confidence intervals

    """
    try:
        # Get summary table
        coef_table = fitted_model.summary2().tables[1]

        # Add model name if provided
        if model_name:
            coef_table["Model"] = model_name

        # Ensure standard columns
        expected_cols = ["Coef.", "Std.Err.", "z", "P>|z|", "[0.025", "0.975]"]
        for col in expected_cols:
            if col not in coef_table.columns:
                coef_table[col] = np.nan

        return coef_table

    except Exception as e:
        print(f"Warning: Could not extract coefficients for {model_name}: {e}")
        return pd.DataFrame()


def validate_data_for_modeling(
    data: pd.DataFrame,
    response_col: str = "ObservedEntities",
    offset_col: str = "TotalDocuments",
) -> dict[str, Any]:
    """Validate data for GLM modeling and return diagnostic information.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    response_col : str
        Name of response variable
    offset_col : str
        Name of offset variable

    Returns
    -------
    dict
        Validation results and recommendations

    """
    validation = {"valid": True, "warnings": [], "errors": [], "recommendations": []}

    # Check required columns
    required_cols = [response_col, offset_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        validation["errors"].append(f"Missing required columns: {missing_cols}")
        validation["valid"] = False

    if not validation["valid"]:
        return validation

    # Check response variable
    y = data[response_col]
    if y.isnull().any():
        validation["warnings"].append(
            f"Response variable {response_col} contains missing values",
        )

    if (y < 0).any():
        validation["errors"].append(
            f"Response variable {response_col} contains negative values",
        )
        validation["valid"] = False

    if y.dtype.kind not in "iufc":
        validation["warnings"].append(
            f"Response variable {response_col} is not numeric",
        )

    # Check offset variable
    offset = data[offset_col]
    if offset.isnull().any():
        validation["warnings"].append(
            f"Offset variable {offset_col} contains missing values",
        )

    if (offset <= 0).any():
        validation["warnings"].append(
            f"Offset variable {offset_col} contains zero or negative values",
        )
        validation["recommendations"].append(
            "Consider adding small constant to offset variable",
        )

    # Check for extreme values
    y_99 = y.quantile(0.99)
    y_extreme = (y > y_99 * 10).sum()
    if y_extreme > 0:
        validation["warnings"].append(
            f"Found {y_extreme} extreme outliers in response variable",
        )
        validation["recommendations"].append(
            "Consider removing or transforming extreme outliers",
        )

    # Check zero inflation
    zero_prop = (y == 0).mean()
    if zero_prop > 0.3:
        validation["warnings"].append(
            f"High proportion of zeros in response ({zero_prop:.1%})",
        )
        validation["recommendations"].append("Consider zero-inflated models")

    # Check temporal structure
    if "Year" in data.columns or "Date" in data.columns:
        time_col = "Date" if "Date" in data.columns else "Year"
        if data[time_col].duplicated().any():
            validation["warnings"].append(f"Duplicate time points found in {time_col}")
            validation["recommendations"].append(
                "Consider aggregating duplicate time points",
            )

    return validation
