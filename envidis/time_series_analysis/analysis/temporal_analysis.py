#!/usr/bin/env python3
"""
Temporal analysis functions for time series data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from scipy import stats
import warnings

class TemporalAnalyzer:
    """Comprehensive temporal analysis for time series data."""

    def __init__(self):
        """Initialize temporal analyzer."""
        pass

    def analyze_temporal_patterns(self, data: pd.DataFrame,
                                date_col: str = 'date',
                                target_col: str = None) -> Dict[str, Any]:
        """Analyze temporal patterns in the data."""
        if target_col is None:
            target_col = [col for col in data.columns if col != date_col][0]

        data_copy = data.copy()
        data_copy['date'] = pd.to_datetime(data_copy[date_col])
        data_copy['year'] = data_copy['date'].dt.year
        data_copy['month'] = data_copy['date'].dt.month
        data_copy['day_of_year'] = data_copy['date'].dt.dayofyear
        data_copy['day_of_week'] = data_copy['date'].dt.dayofweek

        results = {}

        # 1. Basic temporal statistics
        results['basic_stats'] = {
            'start_date': data_copy['date'].min(),
            'end_date': data_copy['date'].max(),
            'total_days': len(data_copy),
            'date_range_years': (data_copy['date'].max() - data_copy['date'].min()).days / 365.25,
            'mean_value': data_copy[target_col].mean(),
            'std_value': data_copy[target_col].std(),
            'zero_proportion': (data_copy[target_col] == 0).mean()
        }

        # 2. Yearly analysis
        yearly_stats = data_copy.groupby('year')[target_col].agg([
            'count', 'sum', 'mean', 'std', 'min', 'max'
        ]).reset_index()

        results['yearly_analysis'] = {
            'yearly_stats': yearly_stats,
            'yearly_trend': self._calculate_trend(yearly_stats['year'], yearly_stats['sum']),
            'yearly_correlation': stats.pearsonr(yearly_stats['year'], yearly_stats['sum'])[0] if len(yearly_stats) > 1 else 0
        }

        # 3. Monthly patterns
        monthly_stats = data_copy.groupby('month')[target_col].agg([
            'count', 'sum', 'mean', 'std'
        ]).reset_index()

        results['monthly_analysis'] = {
            'monthly_stats': monthly_stats,
            'seasonal_pattern': self._detect_seasonal_pattern(monthly_stats['mean'].values),
            'monthly_variation': monthly_stats['mean'].std() / monthly_stats['mean'].mean() if monthly_stats['mean'].mean() > 0 else 0
        }

        # 4. Day of week patterns (if applicable)
        if len(data_copy['day_of_week'].unique()) > 1:
            dow_stats = data_copy.groupby('day_of_week')[target_col].agg([
                'count', 'sum', 'mean', 'std'
            ]).reset_index()

            results['day_of_week_analysis'] = {
                'dow_stats': dow_stats,
                'weekend_effect': self._calculate_weekend_effect(data_copy, target_col)
            }

        # 5. Autocorrelation analysis
        results['autocorrelation'] = self._calculate_autocorrelation(data_copy[target_col])

        # 6. Trend analysis
        results['trend_analysis'] = self._analyze_trend(data_copy, date_col, target_col)

        # 7. Changepoint detection
        results['changepoints'] = self._detect_changepoints(data_copy[target_col])

        return results

    def _calculate_trend(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate trend statistics."""
        if len(x) < 2:
            return {'slope': 0, 'intercept': 0, 'r_value': 0, 'p_value': 1}

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        return {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value,
            'std_err': std_err
        }

    def _detect_seasonal_pattern(self, monthly_means: np.ndarray) -> Dict[str, Any]:
        """Detect seasonal patterns in monthly data."""
        if len(monthly_means) < 12:
            return {'pattern_detected': False, 'pattern_type': 'insufficient_data'}

        # Simple seasonal pattern detection
        summer_months = [5, 6, 7, 8]  # June, July, August, September (0-indexed)
        winter_months = [11, 0, 1, 2]  # December, January, February, March

        summer_mean = np.mean([monthly_means[i] for i in summer_months if i < len(monthly_means)])
        winter_mean = np.mean([monthly_means[i] for i in winter_months if i < len(monthly_means)])

        seasonal_amplitude = abs(summer_mean - winter_mean)
        overall_mean = np.mean(monthly_means)

        pattern_strength = seasonal_amplitude / overall_mean if overall_mean > 0 else 0

        return {
            'pattern_detected': pattern_strength > 0.1,
            'pattern_strength': pattern_strength,
            'summer_mean': summer_mean,
            'winter_mean': winter_mean,
            'seasonal_amplitude': seasonal_amplitude
        }

    def _calculate_weekend_effect(self, data: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """Calculate weekend vs weekday effects."""
        weekday_data = data[data['day_of_week'] < 5][target_col]  # Monday-Friday
        weekend_data = data[data['day_of_week'] >= 5][target_col]  # Saturday-Sunday

        weekday_mean = weekday_data.mean() if len(weekday_data) > 0 else 0
        weekend_mean = weekend_data.mean() if len(weekend_data) > 0 else 0

        weekend_effect = (weekend_mean - weekday_mean) / weekday_mean if weekday_mean > 0 else 0

        # Statistical test
        if len(weekday_data) > 0 and len(weekend_data) > 0:
            t_stat, p_value = stats.ttest_ind(weekday_data, weekend_data)
        else:
            t_stat, p_value = 0, 1

        return {
            'weekday_mean': weekday_mean,
            'weekend_mean': weekend_mean,
            'weekend_effect': weekend_effect,
            't_statistic': t_stat,
            'p_value': p_value
        }

    def _calculate_autocorrelation(self, series: pd.Series, max_lags: int = 30) -> Dict[str, Any]:
        """Calculate autocorrelation function."""
        from statsmodels.tsa.stattools import acf, pacf

        try:
            # Ensure we don't exceed series length
            max_lags = min(max_lags, len(series) - 1)

            # Calculate autocorrelation
            autocorr = acf(series, nlags=max_lags, alpha=0.05, fft=True)

            # Calculate partial autocorrelation
            partial_autocorr = pacf(series, nlags=max_lags, alpha=0.05)

            # Find significant lags
            significant_lags = []
            for i in range(1, len(autocorr[0])):
                if abs(autocorr[0][i]) > 2/np.sqrt(len(series)):
                    significant_lags.append(i)

            return {
                'autocorrelation': autocorr[0],
                'autocorr_confidence_intervals': autocorr[1] if len(autocorr) > 1 else None,
                'partial_autocorrelation': partial_autocorr[0],
                'partial_autocorr_confidence_intervals': partial_autocorr[1] if len(partial_autocorr) > 1 else None,
                'significant_lags': significant_lags,
                'max_autocorr': np.max(np.abs(autocorr[0][1:])) if len(autocorr[0]) > 1 else 0
            }

        except Exception as e:
            warnings.warn(f"Autocorrelation calculation failed: {e}")
            return {
                'autocorrelation': None,
                'error': str(e)
            }

    def _analyze_trend(self, data: pd.DataFrame, date_col: str, target_col: str) -> Dict[str, Any]:
        """Comprehensive trend analysis."""
        # Create time variable
        data_copy = data.copy()
        data_copy['time_numeric'] = (pd.to_datetime(data_copy[date_col]) -
                                    pd.to_datetime(data_copy[date_col]).min()).dt.days

        # Linear trend
        linear_trend = self._calculate_trend(data_copy['time_numeric'], data_copy[target_col])

        # Non-linear trend detection using polynomial fits
        trend_results = {'linear': linear_trend}

        # Test polynomial trends
        for degree in [2, 3]:
            try:
                coeffs = np.polyfit(data_copy['time_numeric'], data_copy[target_col], degree)
                poly_pred = np.polyval(coeffs, data_copy['time_numeric'])

                # Calculate R²
                ss_res = np.sum((data_copy[target_col] - poly_pred) ** 2)
                ss_tot = np.sum((data_copy[target_col] - np.mean(data_copy[target_col])) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                trend_results[f'polynomial_degree_{degree}'] = {
                    'coefficients': coeffs.tolist(),
                    'r_squared': r_squared
                }
            except Exception:
                pass

        return trend_results

    def _detect_changepoints(self, series: pd.Series, min_size: int = 30) -> Dict[str, Any]:
        """Simple changepoint detection using variance changes."""
        try:
            if len(series) < 2 * min_size:
                return {'changepoints': [], 'method': 'insufficient_data'}

            # Simple variance-based changepoint detection
            changepoints = []

            # Rolling variance to detect changes
            window_size = max(min_size, len(series) // 10)
            rolling_var = series.rolling(window=window_size).var()

            # Find points where variance changes significantly
            var_threshold = rolling_var.quantile(0.75)

            for i in range(window_size, len(series) - window_size):
                before_var = series.iloc[i-window_size:i].var()
                after_var = series.iloc[i:i+window_size].var()

                if abs(before_var - after_var) > var_threshold:
                    changepoints.append(i)

            # Remove close changepoints
            if changepoints:
                filtered_changepoints = [changepoints[0]]
                for cp in changepoints[1:]:
                    if cp - filtered_changepoints[-1] > min_size:
                        filtered_changepoints.append(cp)
                changepoints = filtered_changepoints

            return {
                'changepoints': changepoints,
                'num_changepoints': len(changepoints),
                'method': 'variance_based'
            }

        except Exception as e:
            return {
                'changepoints': [],
                'error': str(e),
                'method': 'failed'
            }

    def generate_temporal_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a text summary of temporal analysis."""
        summary = ["=== TEMPORAL ANALYSIS SUMMARY ===\n"]

        # Basic statistics
        basic = analysis_results['basic_stats']
        summary.append(f"Data Period: {basic['start_date'].strftime('%Y-%m-%d')} to {basic['end_date'].strftime('%Y-%m-%d')}")
        summary.append(f"Total Observations: {basic['total_days']:,}")
        summary.append(f"Time Span: {basic['date_range_years']:.1f} years")
        summary.append(f"Zero Proportion: {basic['zero_proportion']:.1%}")
        summary.append(f"Mean Value: {basic['mean_value']:.2f}")
        summary.append("")

        # Yearly trends
        if 'yearly_analysis' in analysis_results:
            yearly = analysis_results['yearly_analysis']
            trend = yearly['yearly_trend']
            summary.append("YEARLY TRENDS:")
            summary.append(f"  Linear Trend Slope: {trend['slope']:.4f}")
            summary.append(f"  Trend Significance: p = {trend['p_value']:.4f}")
            summary.append(f"  Yearly Correlation: r = {yearly['yearly_correlation']:.3f}")
            summary.append("")

        # Monthly patterns
        if 'monthly_analysis' in analysis_results:
            monthly = analysis_results['monthly_analysis']
            seasonal = monthly['seasonal_pattern']
            summary.append("SEASONAL PATTERNS:")
            summary.append(f"  Seasonal Pattern Detected: {seasonal['pattern_detected']}")
            if seasonal['pattern_detected']:
                summary.append(f"  Pattern Strength: {seasonal['pattern_strength']:.3f}")
                summary.append(f"  Summer vs Winter: {seasonal['summer_mean']:.2f} vs {seasonal['winter_mean']:.2f}")
            summary.append("")

        # Autocorrelation
        if 'autocorrelation' in analysis_results and analysis_results['autocorrelation'].get('autocorrelation') is not None:
            autocorr = analysis_results['autocorrelation']
            summary.append("AUTOCORRELATION:")
            summary.append(f"  Maximum Autocorrelation: {autocorr['max_autocorr']:.3f}")
            summary.append(f"  Significant Lags: {autocorr['significant_lags'][:5]}")  # Show first 5
            summary.append("")

        # Changepoints
        if 'changepoints' in analysis_results:
            cp = analysis_results['changepoints']
            summary.append("CHANGEPOINTS:")
            summary.append(f"  Number of Changepoints: {cp['num_changepoints']}")
            if cp['num_changepoints'] > 0:
                summary.append(f"  Changepoint Positions: {cp['changepoints'][:5]}")  # Show first 5
            summary.append("")

        return "\n".join(summary)
