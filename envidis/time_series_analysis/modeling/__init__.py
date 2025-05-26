#!/usr/bin/env python3
"""
Modeling module for GLM fitting with proper offset handling.
"""

from .base_models import GLMModelFitter
from .trend_models import TrendModelFitter

__all__ = ["GLMModelFitter", "TrendModelFitter"]
