#!/usr/bin/env python3
"""Plot Configuration for Time Series Analysis
This module contains configuration settings for plot styling and parameters.
"""

from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt

# Color schemes
COLOR_SCHEMES = {
    "default": {
        "primary": "blue",
        "secondary": "red",
        "tertiary": "green",
        "quaternary": "purple",
        "accent": "orange",
        "trend": "red",
        "observed": "blue",
        "zero": "lightcoral",
        "nonzero": "lightblue",
        "recent": "teal",
        "monthly": "purple",
    },
    "colorblind_friendly": {
        "primary": "#1f77b4",  # blue
        "secondary": "#ff7f0e",  # orange
        "tertiary": "#2ca02c",  # green
        "quaternary": "#d62728",  # red
        "accent": "#9467bd",  # purple
        "trend": "#ff7f0e",  # orange
        "observed": "#1f77b4",  # blue
        "zero": "#ffbb78",  # light orange
        "nonzero": "#aec7e8",  # light blue
        "recent": "#17becf",  # cyan
        "monthly": "#9467bd",  # purple
    },
    "grayscale": {
        "primary": "black",
        "secondary": "gray",
        "tertiary": "darkgray",
        "quaternary": "lightgray",
        "accent": "dimgray",
        "trend": "black",
        "observed": "darkgray",
        "zero": "lightgray",
        "nonzero": "darkgray",
        "recent": "black",
        "monthly": "gray",
    },
    "vibrant": {
        "primary": "#FF6B6B",  # coral
        "secondary": "#4ECDC4",  # teal
        "tertiary": "#45B7D1",  # blue
        "quaternary": "#96CEB4",  # mint
        "accent": "#FFEAA7",  # yellow
        "trend": "#FF6B6B",  # coral
        "observed": "#45B7D1",  # blue
        "zero": "#FFB3BA",  # light pink
        "nonzero": "#BAE1FF",  # light blue
        "recent": "#4ECDC4",  # teal
        "monthly": "#96CEB4",  # mint
    },
}

# Default plot parameters
DEFAULT_PLOT_PARAMS = {
    "figure": {
        "figsize": (8.0, 24),  # A4 width (8 inches) with proportional height
        "dpi": 300,
        "facecolor": "white",
        "edgecolor": "none",
    },
    "subplot": {
        "layout": (6, 3),  # 6 rows, 3 columns for comprehensive layout
        "hspace": 0.4,  # Increased vertical spacing for readability
        "wspace": 0.3,  # Optimized horizontal spacing for A4 width
    },
    "fonts": {
        "title_size": 9,  # Reduced for A4 width
        "label_size": 8,  # Reduced for A4 width
        "tick_size": 7,  # Reduced for A4 width
        "legend_size": 7,  # Reduced for A4 width
        "table_size": 7,  # Reduced for A4 width
        "title_weight": "bold",
    },
    "lines": {
        "linewidth": 0.5,
        "markersize": 3,
        "alpha": 0.7,
        "trend_linewidth": 2,
    },
    "scatter": {
        "alpha": 0.6,
        "size": 1,
        "sample_size": 1000,
    },
    "histogram": {
        "bins": 50,
        "alpha": 0.7,
    },
    "table": {
        "fontsize": 9,
        "scale": (1.2, 1.5),
        "cellLoc": "center",
    },
    "sampling": {
        "timeseries_sample_rate": 100,  # Sample every nth point for raw timeseries
        "scatter_sample_size": 1000,  # Max points for scatter plots
    },
    "outlier_filtering": {
        "percentile_threshold": 99.7,  # Changed from 99 to 99.7 for less harsh filtering
        "enable_filtering": True,
        "y_axis_percentile": 95,  # For y-axis limits
    },
    "axes": {
        "major_tick_years": 5,  # Major tick and label every 5 years
        "minor_tick_years": 1,  # Minor tick every year
        "label_rotation": 45,   # Rotation for better readability
        "date_format": "%Y",    # Year format for labels
    },
}

# Plot-specific configurations
PLOT_CONFIGS = {
    "raw_timeseries": {
        "sample_rate": 50,  # Reduced from 100 for higher resolution
        "s": 2,  # marker size for scatter
        "alpha": 0.6,
        "color_key": "primary",
        "remove_outliers": True,  # Only this plot removes outliers
        "outlier_percentile": 99.7,  # Configurable outlier threshold
        "filter_sparse_years": True,  # Remove isolated early years
        "min_year_density": 0.01,  # Minimum data density threshold
        "x_axis": {
            "major_locator_years": 5,
            "minor_locator_years": 1,
            "label_rotation": 45,
            "date_format": "%Y",
        },
    },
    "yearly_aggregated": {
        "marker": "o",
        "markersize": 4,
        "alpha": 0.7,
        "show_trend": True,
        "trend_alpha": 0.5,
        "color_key": "secondary",
        "remove_outliers": False,  # Keep as False for yearly data
        "outlier_percentile": 99.7,  # Even when disabled, have threshold available
        "filter_sparse_years": True,  # Keep sparse year filtering
        "x_axis": {
            "major_locator_years": 5,
            "minor_locator_years": 1,
            "label_rotation": 45,
        },
    },
    "distribution": {
        "bins": 50,
        "alpha": 0.7,
        "color_key": "tertiary",
        "log_scale": True,
        "remove_outliers": False,  # Keep all data for distribution analysis
        "outlier_percentile": 99.7,  # Configurable threshold
    },
    "monthly_patterns": {
        "alpha": 0.7,
        "color_key": "monthly",
        "start_year": 2020,
        "outlier_percentile": 99.7,
    },
    "scatter": {
        "alpha": 0.6,
        "s": 1,
        "sample_size": 1000,
        "color_key": "accent",
        "remove_outliers": False,  # Keep all data for scatter analysis
        "outlier_percentile": 99.7,  # Configurable threshold
    },
    "trend_analysis": {
        "observed_color_key": "observed",
        "trend_color_key": "trend",
        "markersize": 4,
        "marker_alpha": 0.7,
        "trend_linewidth": 2,
        "trend_alpha": 0.8,
        "fit_trend": True,
        "trend_degree": 2,
        "remove_outliers": False,  # Keep all data for trend analysis
        "outlier_percentile": 99.7,  # Configurable threshold
        "y_limit_percentile": 95,  # Use 95th percentile for y-axis upper limit
        "x_axis": {
            "major_locator_years": 5,
            "minor_locator_years": 1,
            "label_rotation": 45,
        },
    },
    "zero_inflation": {
        "colors_keys": ["zero", "nonzero"],
        "autopct": "%1.1f%%",
    },
    "recent_trend": {
        "alpha": 0.7,
        "color_key": "recent",
        "start_year": 2020,
        "remove_outliers": False,  # Keep all data for recent trend analysis
        "outlier_percentile": 99.7,  # Configurable threshold
        "y_limit_percentile": 95,  # Use 95th percentile for y-axis upper limit
        "x_axis": {
            "major_locator_years": 5,
            "minor_locator_years": 1,
            "label_rotation": 45,
        },
    },
    "decomposition": {
        "original_alpha": 0.4,
        "original_markersize": 3,
        "trend_linewidth": 2.5,
        "trend_alpha": 0.8,
        "original_color_key": "observed",
        "trend_color_key": "trend",
        "window": None,  # Auto-calculate
        "remove_outliers": False,  # Keep all data for decomposition analysis
        "outlier_percentile": 99.7,  # Configurable threshold
        "y_limit_percentile": 95,  # Use 95th percentile for y-axis upper limit
        "x_axis": {
            "major_locator_years": 5,
            "minor_locator_years": 1,
            "label_rotation": 45,
        },
    },
    "publication_volume": {
        "doc_color": "steelblue",
        "rate_color": "orange",
        "alpha": 0.7,
        "remove_outliers": False,  # Keep all data for publication volume context
        "outlier_percentile": 99.7,  # Configurable threshold
        "x_axis": {
            "major_locator_years": 5,
            "minor_locator_years": 1,
            "label_rotation": 45,
        },
    },
}

# Layout configurations
LAYOUT_CONFIGS = {
    "comprehensive": {
        "layout": (6, 3),  # 6 rows, 3 columns
        "figsize": (8.0, 24),  # A4 width optimized
        "plots": [
            "raw_timeseries",  # Full width row 1
            "yearly_aggregated",  # Full width row 2
            "distribution",  # Row 3, col 1
            "monthly_patterns",  # Row 3, col 2
            "scatter",  # Row 3, col 3
            "model_summary",  # Row 4, col 1
            "trend_analysis",  # Row 4, col 2
            "zero_inflation",  # Row 4, col 3
            "recent_trend",  # Row 5, col 1
            "coefficients",  # Row 5, col 2
            "data_quality",  # Row 5, col 3
            "decomposition",  # Optional if space allows
        ],
    },
    "temporal_focus": {
        "layout": (2, 2),
        "figsize": (8.0, 6),  # A4 width for focused plots
        "plots": [
            "raw_timeseries",
            "yearly_aggregated",
            "monthly_patterns",
            "recent_trend",
        ],
    },
    "distribution_focus": {
        "layout": (2, 2),
        "figsize": (8.0, 6),  # A4 width for focused plots
        "plots": [
            "distribution_normal",
            "distribution_log",
            "zero_inflation",
            "scatter",
        ],
    },
    "trends_focus": {
        "layout": (2, 2),
        "figsize": (8.0, 6),  # A4 width for focused plots
        "plots": [
            "trend_analysis",
            "decomposition",
            "recent_trend",
            "monthly_patterns",
        ],
    },
    "quality_focus": {
        "layout": (1, 2),
        "figsize": (8.0, 4),  # A4 width for quality plots
        "plots": ["data_quality", "zero_inflation"],
    },
}


class PlotConfig:
    """Configuration manager for plotting parameters."""

    def __init__(
        self,
        color_scheme: str = "default",
        custom_params: dict[str, Any] = None,
    ):
        """Initialize plot configuration.

        Parameters
        ----------
        color_scheme : str
            Color scheme to use ('default', 'colorblind_friendly', 'grayscale', 'vibrant')
        custom_params : dict, optional
            Custom parameters to override defaults

        """
        self.color_scheme = color_scheme
        self.colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES["default"])
        self.params = DEFAULT_PLOT_PARAMS.copy()
        self.plot_configs = PLOT_CONFIGS.copy()
        self.layout_configs = LAYOUT_CONFIGS.copy()

        if custom_params:
            self._update_params(custom_params)

        self._apply_colors_to_configs()

    def _update_params(self, custom_params: dict[str, Any]):
        """Update parameters with custom values."""
        for key, value in custom_params.items():
            if key in self.params and isinstance(self.params[key], dict):
                self.params[key].update(value)
            else:
                self.params[key] = value

    def _apply_colors_to_configs(self):
        """Apply color scheme to plot configurations."""
        for plot_name, config in self.plot_configs.items():
            # Create a copy of config items to avoid RuntimeError during iteration
            config_items = list(config.items())

            # Handle single color key
            if "color_key" in config:
                color_key = config["color_key"]
                if color_key in self.colors:
                    config["color"] = self.colors[color_key]

            # Handle multiple color keys
            if "colors_keys" in config:
                colors = []
                for color_key in config["colors_keys"]:
                    if color_key in self.colors:
                        colors.append(self.colors[color_key])
                if colors:
                    config["colors"] = colors

            # Handle specific color mappings
            for key, value in config_items:  # Use the copy instead of config.items()
                if key.endswith("_color_key") and value in self.colors:
                    new_key = key.replace("_key", "")
                    config[new_key] = self.colors[value]

    def get_plot_config(self, plot_type: str) -> dict[str, Any]:
        """Get configuration for specific plot type."""
        return self.plot_configs.get(plot_type, {})

    def get_layout_config(self, layout_type: str) -> dict[str, Any]:
        """Get configuration for specific layout type."""
        return self.layout_configs.get(layout_type, {})

    def get_color(self, color_key: str) -> str:
        """Get color by key."""
        return self.colors.get(color_key, "black")

    def set_matplotlib_style(self) -> None:
        """Set matplotlib style based on configuration."""
        plt.rcParams.update(
            {
                "figure.figsize": self.params["figure"]["figsize"],
                "figure.dpi": self.params["figure"]["dpi"],
                "figure.facecolor": self.params["figure"]["facecolor"],
                "figure.edgecolor": self.params["figure"]["edgecolor"],
                "font.size": self.params["fonts"]["label_size"],
                "axes.titlesize": self.params["fonts"]["title_size"],
                "axes.labelsize": self.params["fonts"]["label_size"],
                "xtick.labelsize": self.params["fonts"]["tick_size"],
                "ytick.labelsize": self.params["fonts"]["tick_size"],
                "legend.fontsize": self.params["fonts"]["legend_size"],
                "lines.linewidth": self.params["lines"]["linewidth"],
                "lines.markersize": self.params["lines"]["markersize"],
            },
        )

    def create_custom_color_scheme(self, colors: dict[str, str]) -> None:
        """Create and apply custom color scheme."""
        self.colors.update(colors)
        self._apply_colors_to_configs()


# Predefined configurations for common use cases
PUBLICATION_CONFIG = PlotConfig(
    color_scheme="colorblind_friendly",
    custom_params={
        "figure": {"figsize": (8.0, 6), "dpi": 300},  # A4 compatible
        "fonts": {"title_size": 10, "label_size": 9, "tick_size": 8},
        "lines": {"linewidth": 1.2, "markersize": 3},  # Slightly thinner for A4
        "outlier_filtering": {
            "percentile_threshold": 99.5,
        },  # Even less harsh for publication
    },
)

PRESENTATION_CONFIG = PlotConfig(
    color_scheme="vibrant",
    custom_params={
        "figure": {"figsize": (8.0, 6), "dpi": 150},  # A4 compatible
        "fonts": {"title_size": 11, "label_size": 10, "tick_size": 9},
        "lines": {"linewidth": 1.5, "markersize": 4},
        "outlier_filtering": {
            "percentile_threshold": 99.8,
        },  # Very lenient for presentation
    },
)

QUICK_CONFIG = PlotConfig(
    color_scheme="default",
    custom_params={
        "figure": {"figsize": (8.0, 5), "dpi": 100},  # A4 compatible
        "sampling": {"timeseries_sample_rate": 200, "scatter_sample_size": 500},
        "outlier_filtering": {
            "percentile_threshold": 99.0,
        },  # More aggressive for quick analysis
    },
)


def get_config(config_name: str = "default") -> PlotConfig:
    """Get predefined configuration.

    Parameters
    ----------
    config_name : str
        Name of configuration ('default', 'publication', 'presentation', 'quick')

    Returns
    -------
    PlotConfig
        Configuration object

    """
    configs = {
        "default": PlotConfig(),
        "publication": PUBLICATION_CONFIG,
        "presentation": PRESENTATION_CONFIG,
        "quick": QUICK_CONFIG,
    }

    return configs.get(config_name, PlotConfig())
