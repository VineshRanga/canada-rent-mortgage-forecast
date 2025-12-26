"""IO module for loading Statistics Canada data files."""

from src.io.statcan_read import (
    load_tidy_statcan_csv,
    load_wide_pivot_csv,
    filter_window,
    coerce_value_to_float,
)

__all__ = [
    "load_tidy_statcan_csv",
    "load_wide_pivot_csv",
    "filter_window",
    "coerce_value_to_float",
]

