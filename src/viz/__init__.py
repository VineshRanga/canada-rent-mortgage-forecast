"""Visualization module for forecasting results."""

from src.viz.plots import (
    plot_actual_vs_predicted,
    plot_rent_overall,
    plot_rent_by_unit_type,
    plot_rent_top_cmas,
    plot_mortgage_actual_vs_predicted,
    plot_all_rent_plots,
)

# Import rate story charts (optional, may not be available)
try:
    from src.viz.mpl_story import generate_rate_story_charts
except ImportError:
    generate_rate_story_charts = None

__all__ = [
    "plot_actual_vs_predicted",
    "plot_rent_overall",
    "plot_rent_by_unit_type",
    "plot_rent_top_cmas",
    "plot_mortgage_actual_vs_predicted",
    "plot_all_rent_plots",
]

if generate_rate_story_charts is not None:
    __all__.append("generate_rate_story_charts")

