#!/usr/bin/env python3
"""
Temporal analysis functions for time series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy import stats
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings("ignore")


class TemporalAnalyzer:
    """Comprehensive temporal analysis for time series data."""

    def __init__(self):
        """Initialize temporal analyzer."""
        pass

    def analyze_temporal_patterns(
        self, data: pd.DataFrame, date_col: str = "date", target_col: str = None
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in the data."""
        if target_col is None:
            # Try to find the target column
            possible_targets = ["ObservedEntities", "co_count", "count", "value"]
            target_col = next(
                (col for col in possible_targets if col in data.columns), None
            )
            if target_col is None:
                raise ValueError("Could not identify target column")

        data_copy = data.copy()

        # Handle different date formats
        if date_col in data_copy.columns:
            data_copy["date"] = pd.to_datetime(data_copy[date_col])
        elif data_copy.index.name in ["Year", "Date"] or hasattr(
            data_copy.index, "year"
        ):
            data_copy["date"] = pd.to_datetime(data_copy.index, errors="coerce")
        else:
            # Create a date from year if available
            if "Year" in data_copy.columns:
                data_copy["date"] = pd.to_datetime(data_copy["Year"], format="%Y")
            else:
                raise ValueError("Could not create date column")

        data_copy["year"] = data_copy["date"].dt.year
        data_copy["month"] = data_copy["date"].dt.month

        analysis_results = {}

        # Basic temporal statistics
        analysis_results["basic_stats"] = {
            "total_observations": len(data_copy),
            "date_range": (data_copy["date"].min(), data_copy["date"].max()),
            "years_span": data_copy["year"].max() - data_copy["year"].min(),
            "mean_value": data_copy[target_col].mean(),
            "std_value": data_copy[target_col].std(),
            "total_sum": data_copy[target_col].sum(),
        }

        # Trend analysis
        analysis_results["trend"] = self._analyze_trend(data_copy, "date", target_col)

        # Seasonal patterns
        if len(data_copy["month"].unique()) > 1:
            analysis_results["seasonality"] = self._detect_seasonal_pattern(
                data_copy, target_col
            )
        else:
            analysis_results["seasonality"] = {
                "detected": False,
                "reason": "Insufficient monthly data",
            }

        # Autocorrelation
        analysis_results["autocorrelation"] = self._calculate_autocorrelation(
            data_copy[target_col]
        )

        # Change point detection
        analysis_results["changepoints"] = self._detect_changepoints(
            data_copy[target_col]
        )

        # Year-over-year growth
        analysis_results["growth"] = self._calculate_growth_rates(data_copy, target_col)

        return analysis_results

    def _calculate_trend(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate linear trend statistics."""
        if len(x) < 2:
            return {
                "slope": 0,
                "intercept": 0,
                "r_value": 0,
                "p_value": 1.0,
                "std_err": 0,
            }

        # Convert to numeric if needed
        x_numeric = pd.to_numeric(x, errors="coerce")
        y_numeric = pd.to_numeric(y, errors="coerce")

        # Remove NaN values
        mask = ~(np.isnan(x_numeric) | np.isnan(y_numeric))
        x_clean = x_numeric[mask]
        y_clean = y_numeric[mask]

        if len(x_clean) < 2:
            return {
                "slope": 0,
                "intercept": 0,
                "r_value": 0,
                "p_value": 1.0,
                "std_err": 0,
            }

        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_value": float(r_value),
            "p_value": float(p_value),
            "std_err": float(std_err),
        }

    def _detect_seasonal_pattern(
        self, data: pd.DataFrame, target_col: str
    ) -> Dict[str, Any]:
        """Detect seasonal patterns in the data."""
        monthly_means = data.groupby("month")[target_col].mean()

        # Test for significant differences between months
        if len(monthly_means) > 1:
            # Kruskal-Wallis test for seasonal differences
            monthly_groups = [
                group[target_col].values for name, group in data.groupby("month")
            ]
            # Only include groups with enough data
            monthly_groups = [group for group in monthly_groups if len(group) > 0]

            if len(monthly_groups) > 1:
                try:
                    kw_stat, kw_p = stats.kruskal(*monthly_groups)
                    seasonal_detected = kw_p < 0.05
                except:
                    kw_stat, kw_p = 0, 1.0
                    seasonal_detected = False
            else:
                kw_stat, kw_p = 0, 1.0
                seasonal_detected = False
        else:
            kw_stat, kw_p = 0, 1.0
            seasonal_detected = False

        return {
            "detected": seasonal_detected,
            "kruskal_wallis_stat": float(kw_stat),
            "kruskal_wallis_p": float(kw_p),
            "monthly_means": monthly_means.to_dict(),
            "peak_month": int(monthly_means.idxmax()),
            "trough_month": int(monthly_means.idxmin()),
            "seasonal_amplitude": float(monthly_means.max() - monthly_means.min()),
        }

    def _calculate_autocorrelation(
        self, series: pd.Series, max_lags: int = 10
    ) -> Dict[str, Any]:
        """Calculate autocorrelation function."""
        try:
            from statsmodels.tsa.stattools import acf

            # Limit max_lags to reasonable value
            max_lags = min(max_lags, len(series) // 4)

            if max_lags < 1:
                return {"lag_1": 0, "significant_lags": [], "ljung_box_p": 1.0}

            acf_values = acf(series, nlags=max_lags, alpha=0.05)

            # Extract correlation values
            correlations = (
                acf_values[0] if isinstance(acf_values, tuple) else acf_values
            )

            # Find significantly autocorrelated lags (rough approximation)
            critical_value = 1.96 / np.sqrt(len(series))
            significant_lags = [
                i
                for i, val in enumerate(correlations[1:], 1)
                if abs(val) > critical_value
            ]

            # Ljung-Box test for autocorrelation
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox

                lb_result = acorr_ljungbox(
                    series, lags=min(10, max_lags), return_df=True
                )
                ljung_box_p = (
                    lb_result["lb_pvalue"].iloc[-1] if not lb_result.empty else 1.0
                )
            except:
                ljung_box_p = 1.0

            return {
                "lag_1": float(correlations[1]) if len(correlations) > 1 else 0,
                "significant_lags": significant_lags,
                "ljung_box_p": float(ljung_box_p),
                "autocorrelations": [float(x) for x in correlations[: max_lags + 1]],
            }

        except Exception as e:
            print(f"Warning: Autocorrelation calculation failed: {e}")
            return {"lag_1": 0, "significant_lags": [], "ljung_box_p": 1.0}

    def _analyze_trend(
        self, data: pd.DataFrame, date_col: str, target_col: str
    ) -> Dict[str, Any]:
        """Comprehensive trend analysis."""
        # Convert dates to numeric for regression
        data_sorted = data.sort_values(date_col)
        x = np.arange(len(data_sorted))
        y = data_sorted[target_col].values

        # Linear trend
        linear_trend = self._calculate_trend(x, y)

        # Calculate trend direction and strength
        trend_direction = "increasing" if linear_trend["slope"] > 0 else "decreasing"
        trend_strength = (
            "strong"
            if abs(linear_trend["r_value"]) > 0.7
            else "moderate" if abs(linear_trend["r_value"]) > 0.3 else "weak"
        )
        trend_significant = linear_trend["p_value"] < 0.05

        # Mann-Kendall trend test (non-parametric)
        try:
            from scipy.stats import kendalltau

            tau, mk_p = kendalltau(x, y)
            mk_significant = mk_p < 0.05
        except:
            tau, mk_p = 0, 1.0
            mk_significant = False

        return {
            "linear_slope": linear_trend["slope"],
            "linear_r_squared": linear_trend["r_value"] ** 2,
            "linear_p_value": linear_trend["p_value"],
            "direction": trend_direction,
            "strength": trend_strength,
            "significant": trend_significant,
            "mann_kendall_tau": float(tau),
            "mann_kendall_p": float(mk_p),
            "mann_kendall_significant": mk_significant,
        }

    def _detect_changepoints(
        self, series: pd.Series, min_size: int = 5
    ) -> Dict[str, Any]:
        """Detect potential change points in the time series."""
        if len(series) < 2 * min_size:
            return {"changepoints": [], "n_changepoints": 0}

        try:
            # Simple change point detection using rolling statistics
            window = max(min_size, len(series) // 10)
            rolling_mean = series.rolling(window=window, center=True).mean()
            rolling_std = series.rolling(window=window, center=True).std()

            # Look for abrupt changes in mean
            mean_diff = rolling_mean.diff().abs()
            threshold = mean_diff.quantile(0.9)  # Top 10% of changes

            potential_changepoints = mean_diff[mean_diff > threshold].index.tolist()

            # Filter out changepoints that are too close together
            filtered_changepoints = []
            for cp in potential_changepoints:
                if (
                    not filtered_changepoints
                    or cp - filtered_changepoints[-1] >= min_size
                ):
                    filtered_changepoints.append(cp)

            return {
                "changepoints": filtered_changepoints,
                "n_changepoints": len(filtered_changepoints),
                "detection_method": "rolling_statistics",
            }

        except Exception as e:
            print(f"Warning: Change point detection failed: {e}")
            return {"changepoints": [], "n_changepoints": 0}

    def _calculate_growth_rates(
        self, data: pd.DataFrame, target_col: str
    ) -> Dict[str, Any]:
        """Calculate year-over-year growth rates."""
        if "year" not in data.columns:
            return {"mean_growth": 0, "growth_rates": []}

        # Annual aggregation
        annual_data = data.groupby("year")[target_col].sum().sort_index()

        if len(annual_data) < 2:
            return {"mean_growth": 0, "growth_rates": []}

        # Calculate year-over-year growth rates
        growth_rates = annual_data.pct_change().dropna()

        return {
            "mean_growth": float(growth_rates.mean()),
            "median_growth": float(growth_rates.median()),
            "std_growth": float(growth_rates.std()),
            "min_growth": float(growth_rates.min()),
            "max_growth": float(growth_rates.max()),
            "growth_rates": growth_rates.to_dict(),
            "volatile_growth": float(growth_rates.std())
            > 0.5,  # High volatility threshold
        }

    def generate_temporal_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a human-readable summary of temporal analysis."""
        summary = []

        # Basic info
        basic = analysis_results.get("basic_stats", {})
        summary.append(f"Temporal Analysis Summary")
        summary.append("=" * 50)
        summary.append(f"Data span: {basic.get('years_span', 0)} years")
        summary.append(f"Total observations: {basic.get('total_observations', 0)}")
        summary.append(f"Mean value: {basic.get('mean_value', 0):.2f}")

        # Trend
        trend = analysis_results.get("trend", {})
        if trend.get("significant", False):
            direction = trend.get("direction", "unknown")
            strength = trend.get("strength", "unknown")
            summary.append(f"\nTrend: {strength} {direction} trend detected")
            summary.append(f"  - R²: {trend.get('linear_r_squared', 0):.3f}")
            summary.append(f"  - p-value: {trend.get('linear_p_value', 1):.3f}")
        else:
            summary.append(f"\nTrend: No significant trend detected")

        # Seasonality
        seasonal = analysis_results.get("seasonality", {})
        if seasonal.get("detected", False):
            peak_month = seasonal.get("peak_month", 0)
            summary.append(f"\nSeasonality: Seasonal pattern detected")
            summary.append(f"  - Peak month: {peak_month}")
            summary.append(
                f"  - Seasonal amplitude: {seasonal.get('seasonal_amplitude', 0):.2f}"
            )
        else:
            summary.append(f"\nSeasonality: No significant seasonal pattern")

        # Growth
        growth = analysis_results.get("growth", {})
        if growth:
            mean_growth = growth.get("mean_growth", 0)
            summary.append(f"\nGrowth: Average annual growth rate: {mean_growth:.1%}")
            if growth.get("volatile_growth", False):
                summary.append(f"  - Growth is highly volatile")

        # Autocorrelation
        autocorr = analysis_results.get("autocorrelation", {})
        if autocorr.get("ljung_box_p", 1) < 0.05:
            summary.append(
                f"\nAutocorrelation: Significant temporal dependence detected"
            )
        else:
            summary.append(f"\nAutocorrelation: No significant temporal dependence")

        # Change points
        changepoints = analysis_results.get("changepoints", {})
        n_cp = changepoints.get("n_changepoints", 0)
        if n_cp > 0:
            summary.append(
                f"\nChange Points: {n_cp} potential change point(s) detected"
            )

        return "\n".join(summary)
