#!/usr/bin/env python3
"""
Model diagnostics and validation utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from scipy import stats
import warnings


class ModelDiagnostics:
    """Comprehensive model diagnostics and validation."""

    def __init__(self):
        """Initialize model diagnostics."""
        pass

    def run_comprehensive_diagnostics(
        self, fitted_model, data: pd.DataFrame, target_col: str
    ) -> Dict[str, Any]:
        """Run comprehensive model diagnostics."""
        diagnostics = {}

        try:
            # Basic model information
            diagnostics["model_info"] = self._extract_model_info(fitted_model)

            # Residual analysis
            diagnostics["residuals"] = self._analyze_residuals(fitted_model)

            # Goodness of fit
            diagnostics["goodness_of_fit"] = self._assess_goodness_of_fit(
                fitted_model, data, target_col
            )

            # Influence diagnostics
            diagnostics["influence"] = self._analyze_influence(fitted_model)

            # Assumption checks
            diagnostics["assumptions"] = self._check_assumptions(fitted_model, data)

            # Overall assessment
            diagnostics["overall_assessment"] = self._generate_overall_assessment(
                diagnostics
            )

        except Exception as e:
            diagnostics["error"] = str(e)
            warnings.warn(f"Diagnostics failed: {e}")

        return diagnostics

    def _extract_model_info(self, fitted_model) -> Dict[str, Any]:
        """Extract basic model information."""
        info = {}

        try:
            info["model_type"] = fitted_model.__class__.__name__
            info["family"] = (
                fitted_model.family.__class__.__name__
                if hasattr(fitted_model, "family")
                else "Unknown"
            )
            info["link_function"] = (
                fitted_model.family.link.__class__.__name__
                if hasattr(fitted_model, "family")
                else "Unknown"
            )
            info["n_observations"] = fitted_model.nobs
            info["n_parameters"] = fitted_model.df_model
            info["df_residuals"] = fitted_model.df_resid
            info["converged"] = getattr(fitted_model, "converged", True)

        except Exception as e:
            info["extraction_error"] = str(e)

        return info

    def _analyze_residuals(self, fitted_model) -> Dict[str, Any]:
        """Comprehensive residual analysis."""
        residual_analysis = {}

        try:
            # Get different types of residuals
            residuals = {
                "pearson": fitted_model.resid_pearson,
                "deviance": fitted_model.resid_deviance,
                "raw": fitted_model.resid,
            }

            for resid_type, resid_values in residuals.items():
                residual_analysis[resid_type] = {
                    "mean": np.mean(resid_values),
                    "std": np.std(resid_values),
                    "min": np.min(resid_values),
                    "max": np.max(resid_values),
                    "skewness": stats.skew(resid_values),
                    "kurtosis": stats.kurtosis(resid_values),
                }

                # Normality test
                if len(resid_values) > 3:
                    shapiro_stat, shapiro_p = stats.shapiro(
                        resid_values[:5000]
                    )  # Limit for large datasets
                    residual_analysis[resid_type]["normality_test"] = {
                        "shapiro_stat": shapiro_stat,
                        "shapiro_p": shapiro_p,
                        "is_normal": shapiro_p > 0.05,
                    }

            # Durbin-Watson test for autocorrelation (if applicable)
            try:
                from statsmodels.stats.diagnostic import durbin_watson

                dw_stat = durbin_watson(fitted_model.resid)
                residual_analysis["autocorrelation"] = {
                    "durbin_watson": dw_stat,
                    "interpretation": self._interpret_durbin_watson(dw_stat),
                }
            except Exception:
                pass

        except Exception as e:
            residual_analysis["error"] = str(e)

        return residual_analysis

    def _assess_goodness_of_fit(
        self, fitted_model, data: pd.DataFrame, target_col: str
    ) -> Dict[str, Any]:
        """Assess model goodness of fit."""
        goodness = {}

        try:
            # Basic fit statistics
            goodness["aic"] = fitted_model.aic
            goodness["bic"] = fitted_model.bic
            goodness["log_likelihood"] = fitted_model.llf
            goodness["deviance"] = fitted_model.deviance
            goodness["null_deviance"] = fitted_model.null_deviance

            # Pseudo R-squared
            goodness["pseudo_r2"] = (
                1 - (fitted_model.deviance / fitted_model.null_deviance)
                if fitted_model.null_deviance > 0
                else 0
            )

            # Predictions vs actual
            actual = data[target_col].values
            predicted = fitted_model.predict()

            # Correlation between actual and predicted
            correlation = np.corrcoef(actual, predicted)[0, 1]
            goodness["correlation"] = correlation
            goodness["r_squared"] = correlation**2

            # Mean absolute error and RMSE
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))

            goodness["mae"] = mae
            goodness["rmse"] = rmse

            # Relative errors
            mean_actual = np.mean(actual)
            goodness["relative_mae"] = mae / mean_actual if mean_actual > 0 else np.inf
            goodness["relative_rmse"] = (
                rmse / mean_actual if mean_actual > 0 else np.inf
            )

        except Exception as e:
            goodness["error"] = str(e)

        return goodness

    def _analyze_influence(self, fitted_model) -> Dict[str, Any]:
        """Analyze influential observations."""
        influence_analysis = {}

        try:
            influence = fitted_model.get_influence()

            # Cook's distance
            cooks_d = influence.cooks_distance[0]
            threshold_cooks = 4 / fitted_model.nobs

            influence_analysis["cooks_distance"] = {
                "values": cooks_d,
                "threshold": threshold_cooks,
                "influential_obs": np.where(cooks_d > threshold_cooks)[0].tolist(),
                "n_influential": np.sum(cooks_d > threshold_cooks),
                "max_value": np.max(cooks_d),
                "mean_value": np.mean(cooks_d),
            }

            # Leverage
            leverage = influence.hat_matrix_diag
            threshold_leverage = 2 * fitted_model.df_model / fitted_model.nobs

            influence_analysis["leverage"] = {
                "values": leverage,
                "threshold": threshold_leverage,
                "high_leverage_obs": np.where(leverage > threshold_leverage)[
                    0
                ].tolist(),
                "n_high_leverage": np.sum(leverage > threshold_leverage),
                "max_value": np.max(leverage),
                "mean_value": np.mean(leverage),
            }

            # DFFITS
            try:
                dffits = influence.dffits[0]
                threshold_dffits = 2 * np.sqrt(
                    fitted_model.df_model / fitted_model.nobs
                )

                influence_analysis["dffits"] = {
                    "values": dffits,
                    "threshold": threshold_dffits,
                    "influential_obs": np.where(np.abs(dffits) > threshold_dffits)[
                        0
                    ].tolist(),
                    "n_influential": np.sum(np.abs(dffits) > threshold_dffits),
                }
            except Exception:
                influence_analysis["dffits"] = {"error": "Could not calculate DFFITS"}

        except Exception as e:
            influence_analysis["error"] = str(e)

        return influence_analysis

    def _check_assumptions(self, fitted_model, data: pd.DataFrame) -> Dict[str, Any]:
        """Check model assumptions."""
        assumptions = {}

        try:
            # Overdispersion test (for Poisson models)
            if hasattr(fitted_model, "family") and "Poisson" in str(
                fitted_model.family
            ):
                dispersion = fitted_model.deviance / fitted_model.df_resid
                assumptions["overdispersion"] = {
                    "dispersion_parameter": dispersion,
                    "is_overdispersed": dispersion > 1.5,
                    "interpretation": (
                        "Overdispersed"
                        if dispersion > 1.5
                        else "Appropriate dispersion"
                    ),
                }

            # Linearity assumption (residuals vs fitted)
            residuals = fitted_model.resid_pearson
            fitted_values = fitted_model.fittedvalues

            # Simple trend test
            correlation_resid_fitted = np.corrcoef(residuals, fitted_values)[0, 1]
            assumptions["linearity"] = {
                "residual_fitted_correlation": correlation_resid_fitted,
                "assumption_met": abs(correlation_resid_fitted) < 0.1,
                "interpretation": (
                    "Linear"
                    if abs(correlation_resid_fitted) < 0.1
                    else "Non-linear pattern detected"
                ),
            }

            # Homoscedasticity (constant variance)
            # Breusch-Pagan test
            try:
                from statsmodels.stats.diagnostic import het_breuschpagan

                bp_stat, bp_p, _, _ = het_breuschpagan(
                    residuals, fitted_model.model.exog
                )
                assumptions["homoscedasticity"] = {
                    "breusch_pagan_stat": bp_stat,
                    "breusch_pagan_p": bp_p,
                    "homoscedastic": bp_p > 0.05,
                    "interpretation": (
                        "Constant variance"
                        if bp_p > 0.05
                        else "Heteroscedasticity detected"
                    ),
                }
            except Exception:
                # Fallback: simple variance analysis
                fitted_sorted = np.argsort(fitted_values)
                n_groups = 3
                group_size = len(fitted_sorted) // n_groups

                group_vars = []
                for i in range(n_groups):
                    start_idx = i * group_size
                    end_idx = (
                        (i + 1) * group_size if i < n_groups - 1 else len(fitted_sorted)
                    )
                    group_residuals = residuals[fitted_sorted[start_idx:end_idx]]
                    group_vars.append(np.var(group_residuals))

                var_ratio = (
                    max(group_vars) / min(group_vars) if min(group_vars) > 0 else np.inf
                )
                assumptions["homoscedasticity"] = {
                    "variance_ratio": var_ratio,
                    "homoscedastic": var_ratio < 3,
                    "interpretation": (
                        "Constant variance"
                        if var_ratio < 3
                        else "Heteroscedasticity detected"
                    ),
                }

        except Exception as e:
            assumptions["error"] = str(e)

        return assumptions

    def _generate_overall_assessment(
        self, diagnostics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate overall model assessment."""
        assessment = {
            "model_quality": "Unknown",
            "key_issues": [],
            "recommendations": [],
            "overall_score": 0.0,
        }

        try:
            score = 0.0
            max_score = 0.0

            # Model info assessment
            if (
                "model_info" in diagnostics
                and "extraction_error" not in diagnostics["model_info"]
            ):
                if diagnostics["model_info"].get("converged", True):
                    score += 1
                max_score += 1

            # Goodness of fit assessment
            if (
                "goodness_of_fit" in diagnostics
                and "error" not in diagnostics["goodness_of_fit"]
            ):
                gof = diagnostics["goodness_of_fit"]

                # Pseudo R-squared assessment
                pseudo_r2 = gof.get("pseudo_r2", 0)
                if pseudo_r2 > 0.7:
                    score += 1
                elif pseudo_r2 > 0.5:
                    score += 0.7
                elif pseudo_r2 > 0.3:
                    score += 0.4
                max_score += 1

                # Correlation assessment
                correlation = gof.get("correlation", 0)
                if abs(correlation) > 0.8:
                    score += 1
                elif abs(correlation) > 0.6:
                    score += 0.7
                max_score += 1

            # Residual analysis assessment
            if "residuals" in diagnostics and "error" not in diagnostics["residuals"]:
                residuals = diagnostics["residuals"]

                # Normality assessment
                if "pearson" in residuals and "normality_test" in residuals["pearson"]:
                    if residuals["pearson"]["normality_test"]["is_normal"]:
                        score += 1
                    max_score += 1

                # Autocorrelation assessment
                if "autocorrelation" in residuals:
                    dw_stat = residuals["autocorrelation"].get("durbin_watson", 2.0)
                    if 1.5 < dw_stat < 2.5:  # Acceptable range
                        score += 1
                    max_score += 1

            # Assumptions assessment
            if (
                "assumptions" in diagnostics
                and "error" not in diagnostics["assumptions"]
            ):
                assumptions = diagnostics["assumptions"]

                # Overdispersion
                if "overdispersion" in assumptions:
                    if not assumptions["overdispersion"]["is_overdispersed"]:
                        score += 1
                    else:
                        assessment["key_issues"].append("Model shows overdispersion")
                        assessment["recommendations"].append(
                            "Consider Negative Binomial model"
                        )
                    max_score += 1

                # Linearity
                if "linearity" in assumptions:
                    if assumptions["linearity"]["assumption_met"]:
                        score += 1
                    else:
                        assessment["key_issues"].append(
                            "Non-linear pattern in residuals"
                        )
                        assessment["recommendations"].append(
                            "Consider polynomial or spline terms"
                        )
                    max_score += 1

                # Homoscedasticity
                if "homoscedasticity" in assumptions:
                    if assumptions["homoscedasticity"]["homoscedastic"]:
                        score += 1
                    else:
                        assessment["key_issues"].append("Heteroscedasticity detected")
                        assessment["recommendations"].append(
                            "Consider robust standard errors"
                        )
                    max_score += 1

            # Calculate overall score
            assessment["overall_score"] = score / max_score if max_score > 0 else 0.0

            # Determine model quality
            if assessment["overall_score"] >= 0.8:
                assessment["model_quality"] = "Excellent"
            elif assessment["overall_score"] >= 0.6:
                assessment["model_quality"] = "Good"
            elif assessment["overall_score"] >= 0.4:
                assessment["model_quality"] = "Fair"
            else:
                assessment["model_quality"] = "Poor"

            # General recommendations
            if not assessment["key_issues"]:
                assessment["recommendations"].append("Model appears to fit well")
            else:
                assessment["recommendations"].append(
                    "Address identified issues for better model fit"
                )

        except Exception as e:
            assessment["error"] = str(e)

        return assessment

    def _interpret_durbin_watson(self, dw_stat: float) -> str:
        """Interpret Durbin-Watson statistic."""
        if dw_stat < 1.5:
            return "Positive autocorrelation detected"
        elif dw_stat > 2.5:
            return "Negative autocorrelation detected"
        else:
            return "No significant autocorrelation"

    def generate_diagnostic_report(self, diagnostics: Dict[str, Any]) -> str:
        """Generate a formatted diagnostic report."""
        report = ["=== MODEL DIAGNOSTICS REPORT ===\n"]

        # Model info
        if "model_info" in diagnostics:
            report.append("MODEL INFORMATION:")
            model_info = diagnostics["model_info"]
            for key, value in model_info.items():
                if key != "extraction_error":
                    report.append(f"  {key.replace('_', ' ').title()}: {value}")
            report.append("")

        # Overall assessment
        if "overall_assessment" in diagnostics:
            assessment = diagnostics["overall_assessment"]
            report.append("OVERALL ASSESSMENT:")
            report.append(f"  Model Quality: {assessment['model_quality']}")
            report.append(f"  Overall Score: {assessment['overall_score']:.2f}")

            if assessment["key_issues"]:
                report.append("  Key Issues:")
                for issue in assessment["key_issues"]:
                    report.append(f"    - {issue}")

            if assessment["recommendations"]:
                report.append("  Recommendations:")
                for rec in assessment["recommendations"]:
                    report.append(f"    - {rec}")
            report.append("")

        # Goodness of fit
        if (
            "goodness_of_fit" in diagnostics
            and "error" not in diagnostics["goodness_of_fit"]
        ):
            gof = diagnostics["goodness_of_fit"]
            report.append("GOODNESS OF FIT:")
            report.append(f"  AIC: {gof.get('aic', 'N/A'):.3f}")
            report.append(f"  BIC: {gof.get('bic', 'N/A'):.3f}")
            report.append(f"  Log-Likelihood: {gof.get('log_likelihood', 'N/A'):.3f}")
            report.append(f"  Pseudo R²: {gof.get('pseudo_r2', 'N/A'):.4f}")
            report.append(
                f"  Correlation (actual vs predicted): {gof.get('correlation', 'N/A'):.4f}"
            )
            report.append(f"  MAE: {gof.get('mae', 'N/A'):.3f}")
            report.append(f"  RMSE: {gof.get('rmse', 'N/A'):.3f}")
            report.append("")

        # Residual analysis
        if "residuals" in diagnostics and "error" not in diagnostics["residuals"]:
            residuals = diagnostics["residuals"]
            report.append("RESIDUAL ANALYSIS:")

            for resid_type in ["pearson", "deviance"]:
                if resid_type in residuals:
                    resid_data = residuals[resid_type]
                    report.append(f"  {resid_type.title()} Residuals:")
                    report.append(f"    Mean: {resid_data.get('mean', 'N/A'):.4f}")
                    report.append(f"    Std: {resid_data.get('std', 'N/A'):.4f}")
                    report.append(
                        f"    Skewness: {resid_data.get('skewness', 'N/A'):.4f}"
                    )
                    report.append(
                        f"    Kurtosis: {resid_data.get('kurtosis', 'N/A'):.4f}"
                    )

                    if "normality_test" in resid_data:
                        norm_test = resid_data["normality_test"]
                        report.append(
                            f"    Normality (Shapiro): p={norm_test.get('shapiro_p', 'N/A'):.4f} "
                            f"({'Normal' if norm_test.get('is_normal', False) else 'Non-normal'})"
                        )

            if "autocorrelation" in residuals:
                auto = residuals["autocorrelation"]
                report.append(
                    f"  Autocorrelation (Durbin-Watson): {auto.get('durbin_watson', 'N/A'):.3f} "
                    f"({auto.get('interpretation', 'N/A')})"
                )
            report.append("")

        # Assumptions
        if "assumptions" in diagnostics and "error" not in diagnostics["assumptions"]:
            assumptions = diagnostics["assumptions"]
            report.append("ASSUMPTION CHECKS:")

            if "overdispersion" in assumptions:
                od = assumptions["overdispersion"]
                report.append(
                    f"  Overdispersion: {od.get('dispersion_parameter', 'N/A'):.3f} "
                    f"({od.get('interpretation', 'N/A')})"
                )

            if "linearity" in assumptions:
                lin = assumptions["linearity"]
                report.append(
                    f"  Linearity: {lin.get('interpretation', 'N/A')} "
                    f"(correlation: {lin.get('residual_fitted_correlation', 'N/A'):.3f})"
                )

            if "homoscedasticity" in assumptions:
                homo = assumptions["homoscedasticity"]
                report.append(
                    f"  Homoscedasticity: {homo.get('interpretation', 'N/A')}"
                )
            report.append("")

        # Influence analysis
        if "influence" in diagnostics and "error" not in diagnostics["influence"]:
            influence = diagnostics["influence"]
            report.append("INFLUENCE ANALYSIS:")

            if "cooks_distance" in influence:
                cooks = influence["cooks_distance"]
                report.append(
                    f"  Cook's Distance: {cooks.get('n_influential', 0)} influential observations "
                    f"(threshold: {cooks.get('threshold', 'N/A'):.4f})"
                )

            if "leverage" in influence:
                lev = influence["leverage"]
                report.append(
                    f"  High Leverage: {lev.get('n_high_leverage', 0)} observations "
                    f"(threshold: {lev.get('threshold', 'N/A'):.4f})"
                )
            report.append("")

        return "\n".join(report)
