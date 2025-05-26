#!/usr/bin/env python3
"""
Time Series Analysis Package
A modular framework for comprehensive time series analysis with proper GLM modeling.
"""

__version__ = "1.0.0"
__author__ = "Time Series Analysis Framework"

# Import main classes for easy access
from .modeling.base_models import GLMModelFitter
from .modeling.trend_models import TrendModelFitter
from .plotting.comprehensive_plots import ComprehensivePlotter
from .analysis.temporal_analysis import TemporalAnalyzer
from .utils.data_preparation import DataPreparator

__all__ = [
    'GLMModelFitter',
    'TrendModelFitter',
    'ComprehensivePlotter',
    'TemporalAnalyzer',
    'DataPreparator'
]
