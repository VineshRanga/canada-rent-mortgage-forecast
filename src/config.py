"""
Configuration module for Canadian rent and mortgage forecasting pipeline.

This module defines paths, data file mappings, date windows, frequency conventions,
and utility functions for the forecasting pipeline.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List
import calendar


# ============================================================================
# Directory Paths
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.resolve()
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"


# ============================================================================
# Raw Data Files Mapping
# ============================================================================

RAW_FILES: Dict[str, str] = {
    "rent": "4610009201-noSymbol.csv",
    "mortgage": "1010013401-noSymbol.csv",
    "household_credit": "3810023801-noSymbol.csv",
    "financial_markets": "1010014501-noSymbol.csv",
    "labour_force": "1410028701-noSymbol.csv",
    "employees": "1410032001-noSymbol.csv",
    "housing_starts": "34100156.csv",
    "population": "17100009.csv",
    "migration": "17100040.csv",
    "non_permanent_residents": "17100121.csv",
}


# ============================================================================
# Project Window
# ============================================================================

PROJECT_WINDOW_START = "2019-01-01"
PROJECT_WINDOW_END = "2025-12-31"

# Note: Rent and mortgage targets are constrained to 2019Q1-2025Q3
# based on available data in the provided files.


# ============================================================================
# Frequency Conventions
# ============================================================================

# Quarterly index uses quarter-end Timestamp
# Weekly to quarterly: last observation in quarter
# Monthly to quarterly: mean (for labour/employment) or sum (for housing starts)


# ============================================================================
# Rent Unit Types
# ============================================================================

RENT_UNIT_TYPES: List[str] = [
    "Apartment - 1 bedroom",
    "Apartment - 2 bedrooms",
]


# ============================================================================
# Helper Functions
# ============================================================================

def parse_quarter_str_to_timestamp(quarter_str: str) -> datetime:
    """
    Parse a quarter string (e.g., "2019Q1") to a quarter-end Timestamp.
    
    Args:
        quarter_str: String in format "YYYYQN" where N is 1-4.
        
    Returns:
        datetime: Quarter-end Timestamp (last day of the quarter).
        
    Raises:
        ValueError: If quarter_str is not in the expected format.
        
    Examples:
        >>> parse_quarter_str_to_timestamp("2019Q1")
        datetime.datetime(2019, 3, 31, 0, 0)
        >>> parse_quarter_str_to_timestamp("2025Q3")
        datetime.datetime(2025, 9, 30, 0, 0)
    """
    if len(quarter_str) != 6 or quarter_str[4] != "Q":
        raise ValueError(
            f"Invalid quarter format: {quarter_str}. Expected format: YYYYQN (e.g., '2019Q1')"
        )
    
    try:
        year = int(quarter_str[:4])
        quarter_num = int(quarter_str[5])
        
        if quarter_num not in [1, 2, 3, 4]:
            raise ValueError(f"Quarter number must be 1-4, got {quarter_num}")
        
        # Calculate quarter-end month and day
        quarter_end_month = quarter_num * 3
        last_day = calendar.monthrange(year, quarter_end_month)[1]
        
        return datetime(year, quarter_end_month, last_day)
        
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse quarter string '{quarter_str}': {e}")


def quarter_end(dt: datetime) -> datetime:
    """
    Convert a datetime to the quarter-end Timestamp.
    
    Args:
        dt: Input datetime.
        
    Returns:
        datetime: Quarter-end Timestamp (last day of the quarter containing dt).
        
    Examples:
        >>> quarter_end(datetime(2019, 2, 15))
        datetime.datetime(2019, 3, 31, 0, 0)
        >>> quarter_end(datetime(2019, 7, 1))
        datetime.datetime(2019, 9, 30, 0, 0)
    """
    year = dt.year
    month = dt.month
    
    # Determine which quarter the month belongs to
    quarter_num = (month - 1) // 3 + 1
    quarter_end_month = quarter_num * 3
    last_day = calendar.monthrange(year, quarter_end_month)[1]
    
    return datetime(year, quarter_end_month, last_day)


def ensure_output_dirs() -> None:
    """
    Create required output directories if they do not exist.
    
    Creates the following directory structure:
    - outputs/
    - outputs/plots/
    - outputs/metrics/
    - outputs/forecasts/
    - data/processed/
    """
    output_dirs = [
        OUTPUT_DIR,
        OUTPUT_DIR / "plots",
        OUTPUT_DIR / "metrics",
        OUTPUT_DIR / "forecasts",
        PROCESSED_DIR,
    ]
    
    for dir_path in output_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

