#!/usr/bin/env python3
"""
Model selection and comparison utilities.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import warnings


class ModelSelector:
    """Model selection and comparison utilities."""

    def __init__(self):
        """Initialize model selector."""
        pass

    def compare_models(self, fitted_models: Dict[str, Any]) -> Dict[str, Any]:
        """Compare multiple fitted models using various criteria."""
        comparison_results = {}

        for model_name, model in fitted_models.items():
            try:
                model_stats = {
                    "aic": model.aic,
                    "bic": model.bic,
                    "log_likelihood": model.llf,
                    "deviance": model.deviance,
                    "null_deviance": model.null_deviance,
                    "pseudo_r2": (
                        1 - (model.deviance / model.null_deviance)
                        if model.null_deviance > 0
                        else 0
                    ),
                    "n_observations": model.nobs,
                    "n_parameters": model.df_model,
                    "df_residuals": model.df_resid,
                }

                # Add model-specific metrics
                if hasattr(model, "scale"):
                    model_stats["scale"] = model.scale

                comparison_results[model_name] = model_stats

            except Exception as e:
                warnings.warn(f"Error extracting stats for model {model_name}: {e}")
                comparison_results[model_name] = {"error": str(e)}

        # Determine best models
        valid_models = {k: v for k, v in comparison_results.items() if "error" not in v}

        if valid_models:
            best_models = {
                "best_aic": min(
                    valid_models.keys(), key=lambda x: valid_models[x]["aic"]
                ),
                "best_bic": min(
                    valid_models.keys(), key=lambda x: valid_models[x]["bic"]
                ),
                "best_pseudo_r2": max(
                    valid_models.keys(), key=lambda x: valid_models[x]["pseudo_r2"]
                ),
            }
        else:
            best_models = {}

        return {
            "model_stats": comparison_results,
            "best_models": best_models,
            "model_ranking": self._rank_models(valid_models),
        }

    def _rank_models(self, model_stats: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Rank models based on multiple criteria."""
        if not model_stats:
            return []

        models = list(model_stats.keys())
        rankings = []

        for model in models:
            stats = model_stats[model]

            # Calculate composite score (lower is better)
            # Normalize AIC and BIC (z-scores)
            aic_values = [model_stats[m]["aic"] for m in models]
            bic_values = [model_stats[m]["bic"] for m in models]
            r2_values = [model_stats[m]["pseudo_r2"] for m in models]

            if len(aic_values) > 1:
                aic_zscore = (stats["aic"] - np.mean(aic_values)) / np.std(aic_values)
                bic_zscore = (stats["bic"] - np.mean(bic_values)) / np.std(bic_values)
                r2_zscore = -(stats["pseudo_r2"] - np.mean(r2_values)) / np.std(
                    r2_values
                )  # Negative because higher R² is better
            else:
                aic_zscore = bic_zscore = r2_zscore = 0

            composite_score = aic_zscore + bic_zscore + r2_zscore

            rankings.append(
                {
                    "model": model,
                    "aic": stats["aic"],
                    "bic": stats["bic"],
                    "pseudo_r2": stats["pseudo_r2"],
                    "composite_score": composite_score,
                }
            )

        # Sort by composite score (lower is better)
        rankings.sort(key=lambda x: x["composite_score"])

        return rankings

    def select_best_model(
        self, fitted_models: Dict[str, Any], criterion: str = "aic"
    ) -> Tuple[str, Any]:
        """Select the best model based on specified criterion."""
        valid_models = {}

        for name, model in fitted_models.items():
            try:
                if criterion == "aic":
                    valid_models[name] = model.aic
                elif criterion == "bic":
                    valid_models[name] = model.bic
                elif criterion == "pseudo_r2":
                    pseudo_r2 = (
                        1 - (model.deviance / model.null_deviance)
                        if model.null_deviance > 0
                        else 0
                    )
                    valid_models[name] = (
                        -pseudo_r2
                    )  # Negative because we want to minimize
                else:
                    raise ValueError(f"Unknown criterion: {criterion}")
            except Exception as e:
                warnings.warn(f"Error evaluating model {name}: {e}")
                continue

        if not valid_models:
            raise ValueError("No valid models found")

        best_model_name = min(valid_models.keys(), key=lambda x: valid_models[x])
        best_model = fitted_models[best_model_name]

        return best_model_name, best_model

    def calculate_model_weights(
        self, fitted_models: Dict[str, Any], criterion: str = "aic"
    ) -> Dict[str, float]:
        """Calculate Akaike weights or similar for model averaging."""
        model_values = {}

        for name, model in fitted_models.items():
            try:
                if criterion == "aic":
                    model_values[name] = model.aic
                elif criterion == "bic":
                    model_values[name] = model.bic
                else:
                    raise ValueError(f"Unsupported criterion for weights: {criterion}")
            except Exception:
                continue

        if not model_values:
            return {}

        # Calculate weights
        min_value = min(model_values.values())
        delta_values = {name: value - min_value for name, value in model_values.items()}

        # Calculate relative likelihoods
        rel_likelihoods = {
            name: np.exp(-0.5 * delta) for name, delta in delta_values.items()
        }

        # Normalize to get weights
        total_likelihood = sum(rel_likelihoods.values())
        weights = {
            name: likelihood / total_likelihood
            for name, likelihood in rel_likelihoods.items()
        }

        return weights

    def generate_model_comparison_report(
        self, comparison_results: Dict[str, Any]
    ) -> str:
        """Generate a formatted report of model comparison."""
        report = ["=== MODEL COMPARISON REPORT ===\n"]

        model_stats = comparison_results["model_stats"]
        best_models = comparison_results["best_models"]
        ranking = comparison_results["model_ranking"]

        # Summary table
        report.append("MODEL SUMMARY:")
        report.append("=" * 80)
        report.append(
            f"{'Model':<20} {'AIC':<12} {'BIC':<12} {'Pseudo R²':<12} {'Log-Lik':<12}"
        )
        report.append("-" * 80)

        for model_name, stats in model_stats.items():
            if "error" not in stats:
                report.append(
                    f"{model_name:<20} {stats['aic']:<12.2f} {stats['bic']:<12.2f} "
                    f"{stats['pseudo_r2']:<12.4f} {stats['log_likelihood']:<12.2f}"
                )

        report.append("")

        # Best models
        if best_models:
            report.append("BEST MODELS BY CRITERION:")
            report.append(f"  Best AIC: {best_models.get('best_aic', 'N/A')}")
            report.append(f"  Best BIC: {best_models.get('best_bic', 'N/A')}")
            report.append(
                f"  Best Pseudo R²: {best_models.get('best_pseudo_r2', 'N/A')}"
            )
            report.append("")

        # Model ranking
        if ranking:
            report.append("MODEL RANKING (by composite score):")
            for i, model_info in enumerate(ranking, 1):
                report.append(
                    f"  {i}. {model_info['model']} (Score: {model_info['composite_score']:.3f})"
                )
            report.append("")

        # Detailed statistics
        report.append("DETAILED MODEL STATISTICS:")
        report.append("=" * 80)

        for model_name, stats in model_stats.items():
            if "error" not in stats:
                report.append(f"\n{model_name.upper()}:")
                report.append(f"  AIC: {stats['aic']:.3f}")
                report.append(f"  BIC: {stats['bic']:.3f}")
                report.append(f"  Log-Likelihood: {stats['log_likelihood']:.3f}")
                report.append(f"  Deviance: {stats['deviance']:.3f}")
                report.append(f"  Null Deviance: {stats['null_deviance']:.3f}")
                report.append(f"  Pseudo R²: {stats['pseudo_r2']:.4f}")
                report.append(f"  Observations: {stats['n_observations']}")
                report.append(f"  Parameters: {stats['n_parameters']}")
                report.append(f"  DF Residuals: {stats['df_residuals']}")
            else:
                report.append(f"\n{model_name.upper()}: ERROR - {stats['error']}")

        return "\n".join(report)
