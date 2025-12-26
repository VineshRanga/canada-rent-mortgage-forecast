"""
Statistics Canada CSV file readers.

This module provides functions to load both wide-pivot extracts and tidy
StatCan CSVs into a normalized long format with columns: date, geo, series, value.
"""

import re
from pathlib import Path
from typing import Optional
from csv import reader as csv_reader
import csv as _csv
import pandas as pd
import numpy as np
from datetime import datetime

from src.config import (
    parse_quarter_str_to_timestamp,
    quarter_end,
    PROJECT_WINDOW_START,
    PROJECT_WINDOW_END,
)


# ============================================================================
# Date Pattern Detection
# ============================================================================

# Quarterly patterns: "2019Q1", "Q1 2019", "2019 Q1"
QUARTERLY_PATTERNS = [
    re.compile(r"^\d{4}Q[1-4]$"),
    re.compile(r"^Q[1-4]\s+\d{4}$"),
    re.compile(r"^\d{4}\s*Q[1-4]$"),
]
QUARTERLY_PATTERN = QUARTERLY_PATTERNS[0]  # Keep for backward compatibility
WEEKLY_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MONTHLY_PATTERN = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}$")

MONTH_ABBR_TO_NUM = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


# ============================================================================
# Value Coercion
# ============================================================================

def coerce_value_to_float(value: str) -> float:
    """
    Convert a string value to float, handling StatCan special codes.
    
    Converts "F", "..", "x", "", "E" and non-numeric strings to NaN.
    Strips commas before conversion.
    
    Args:
        value: Input value (string, number, or NaN).
        
    Returns:
        float: Converted value or np.nan if invalid.
    """
    if pd.isna(value):
        return np.nan
    
    # Convert to string if not already
    value_str = str(value).strip()
    
    # Handle StatCan special codes
    if value_str in ["F", "..", "x", "", "E"]:
        return np.nan
    
    # Strip commas
    value_str = value_str.replace(",", "")
    
    # Try to convert to float
    try:
        return float(value_str)
    except (ValueError, TypeError):
        return np.nan


# ============================================================================
# Date Parsing Functions
# ============================================================================

def parse_quarterly_date(date_str: str) -> datetime:
    """
    Parse a quarterly date string (e.g., "2019Q1") to quarter-end Timestamp.
    
    Args:
        date_str: Quarterly date string in format "YYYYQN".
        
    Returns:
        datetime: Quarter-end Timestamp.
    """
    return parse_quarter_str_to_timestamp(date_str)


def parse_weekly_date(date_str: str) -> datetime:
    """
    Parse a weekly date string (e.g., "2019-01-02") to Timestamp.
    
    Args:
        date_str: Weekly date string in format "YYYY-MM-DD".
        
    Returns:
        datetime: Timestamp.
    """
    dt = pd.to_datetime(date_str)
    if isinstance(dt, pd.Timestamp):
        return dt.to_pydatetime()
    return dt


def parse_monthly_date(date_str: str) -> datetime:
    """
    Parse a monthly date string (e.g., "Jan-2019") to month-end Timestamp.
    
    Args:
        date_str: Monthly date string in format "MMM-YYYY".
        
    Returns:
        datetime: Month-end Timestamp.
    """
    parts = date_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid monthly date format: {date_str}")
    
    month_abbr = parts[0]
    year_str = parts[1]
    
    if month_abbr not in MONTH_ABBR_TO_NUM:
        raise ValueError(f"Invalid month abbreviation: {month_abbr}")
    
    month = MONTH_ABBR_TO_NUM[month_abbr]
    year = int(year_str)
    
    # Get last day of month
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    
    return datetime(year, month, last_day)


# ============================================================================
# Tidy CSV Loader
# ============================================================================

def load_tidy_statcan_csv(path: Path) -> pd.DataFrame:
    """
    Load a tidy StatCan CSV with REF_DATE, GEO, VALUE columns.
    
    Expects columns like REF_DATE, GEO, VALUE (case-insensitive).
    Parses REF_DATE into Timestamp.
    Constructs series from remaining categorical columns joined with " | ".
    
    Args:
        path: Path to the CSV file.
        
    Returns:
        DataFrame with columns: date, geo, series, value.
        
    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Read CSV
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    
    # Normalize column names (case-insensitive)
    df.columns = df.columns.str.strip()
    col_map = {col: col.lower() for col in df.columns}
    df = df.rename(columns=col_map)
    
    # Find required columns
    ref_date_col = None
    geo_col = None
    value_col = None
    
    for col in df.columns:
        if "ref_date" in col or col == "date":
            ref_date_col = col
        elif "geo" in col or col == "geography":
            geo_col = col
        elif "value" in col:
            value_col = col
    
    if ref_date_col is None:
        raise ValueError(f"REF_DATE column not found in {path}. Available columns: {list(df.columns)}")
    if value_col is None:
        raise ValueError(f"VALUE column not found in {path}. Available columns: {list(df.columns)}")
    
    # Identify metadata columns for series construction
    metadata_cols = [
        col for col in df.columns
        if col not in [ref_date_col, geo_col, value_col]
        and not col.startswith("symbol")
        and not col.startswith("vector")
        and not col.startswith("coordinate")
        and col.lower() not in ["symbol", "vector", "coordinate", "uom", "uom_id", "scalar_factor", "scalar_id"]
    ]
    
    # Parse dates
    df["date"] = pd.to_datetime(df[ref_date_col], errors="coerce")
    
    # Handle geo
    if geo_col:
        df["geo"] = df[geo_col].astype(str)
    else:
        df["geo"] = "Canada"
    
    # Construct series from metadata columns
    if metadata_cols:
        df["series"] = df[metadata_cols].apply(
            lambda row: " | ".join([str(val) for val in row if pd.notna(val) and str(val).strip() != ""]),
            axis=1
        )
    else:
        df["series"] = "Total"
    
    # Coerce value to float
    df["value"] = df[value_col].apply(coerce_value_to_float)
    
    # Select and return
    result = df[["date", "geo", "series", "value"]].copy()
    result = result.dropna(subset=["date"])  # Remove rows with invalid dates
    
    return result


# ============================================================================
# Quarter Label Normalization
# ============================================================================

def _quarter_label_to_period(label: str) -> str:
    """
    Normalize quarter label to canonical "YYYYQn" format.
    
    Converts:
    - "Q1 2019" -> "2019Q1"
    - "2019 Q1" -> "2019Q1"
    - "2019Q1" -> "2019Q1"
    
    Args:
        label: Quarter label string.
    
    Returns:
        Canonical "YYYYQn" format string.
    """
    label = str(label).strip()
    
    # Pattern: "Q1 2019" or "Q1  2019"
    match = re.match(r"^Q([1-4])\s+(\d{4})$", label)
    if match:
        quarter, year = match.groups()
        return f"{year}Q{quarter}"
    
    # Pattern: "2019 Q1" or "2019  Q1"
    match = re.match(r"^(\d{4})\s+Q([1-4])$", label)
    if match:
        year, quarter = match.groups()
        return f"{year}Q{quarter}"
    
    # Pattern: "2019Q1" (already canonical)
    match = re.match(r"^(\d{4})Q([1-4])$", label)
    if match:
        return label
    
    # If no match, return as-is (will fail parsing later)
    return label


# ============================================================================
# Header Row Detection
# ============================================================================

# Regex for finding weekly dates in raw text (word boundaries)
WEEKLY_DATE_TOKEN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

# Regex for full month-name dates: "January 2, 2019"
MONTH_NAME_DATE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
)

# Regex for month-name + year: "January 2019"
MONTH_NAME_YEAR = re.compile(r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}$")

# Regex for month-name + year token (for searching in text, case-insensitive)
MONTH_NAME_YEAR_TOKEN = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    re.IGNORECASE
)


def _detect_weekly_header_row_by_text(path: Path, max_lines: int = 20000, min_dates: int = 10) -> int | None:
    """
    Scan raw text lines for a header-like line containing many weekly date tokens (YYYY-MM-DD).
    Returns 0-based line index of that line, or None if not found.
    
    Useful for files with malformed quotes that break csv.reader parsing.
    """
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            n = len(WEEKLY_DATE_TOKEN.findall(line))
            if n >= min_dates:
                return i
    return None


def _extract_first_date_token(s: str) -> str | None:
    """
    Extract the first weekly date token (YYYY-MM-DD) from a string.
    
    Args:
        s: Input string (e.g., column name).
    
    Returns:
        First date token found, or None if not found.
    """
    m = WEEKLY_DATE_TOKEN.search(s)
    return m.group(0) if m else None


def _detect_header_row(path: Path, max_lines: int = 200) -> int:
    """
    Detect the true header row in StatCan CSV files.
    
    First tries month-name date detection, then raw-text weekly header detection,
    then falls back to csv.reader-based detection for quarterly/monthly headers.
    
    Args:
        path: Path to the CSV file.
        max_lines: Maximum number of lines to check.
    
    Returns:
        Zero-based index of the header row.
    """
    # First, scan for month-name dates with day (e.g., "January 2, 2019")
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            n = len(MONTH_NAME_DATE.findall(line))
            if n >= 10:
                return i
    
    # Second, scan for month-name + year (e.g., "January 2019")
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 5000:
                break
            n = len(MONTH_NAME_YEAR.findall(line))
            if n >= 10:
                return i
    
    # Third, try raw-text weekly header detection (handles malformed quotes)
    weekly_idx = _detect_weekly_header_row_by_text(path)
    if weekly_idx is not None:
        return weekly_idx
    
    # Fall back to csv.reader-based detection for quarterly/monthly headers
    # Read up to max_lines with encoding="utf-8-sig", errors="replace"
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
        lines = []
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            lines.append(line)
    
    # Date pattern regexes
    quarterly_patterns = [
        re.compile(r"^\d{4}Q[1-4]$"),
        re.compile(r"^Q[1-4]\s+\d{4}$"),
        re.compile(r"^\d{4}\s*Q[1-4]$"),
    ]
    weekly_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    monthly_pattern = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}$")
    
    def is_weekly_date_like(field: str) -> bool:
        """Check if a field looks like a weekly date column (YYYY-MM-DD)."""
        field = field.strip()
        return weekly_pattern.match(field) is not None
    
    def is_date_like(field: str) -> bool:
        """Check if a field looks like a date column (any type)."""
        field = field.strip()
        # Check quarterly patterns
        if any(pattern.match(field) for pattern in quarterly_patterns):
            return True
        # Check weekly pattern
        if weekly_pattern.match(field):
            return True
        # Check monthly pattern
        if monthly_pattern.match(field):
            return True
        return False
    
    # Step 1: Look for weekly header (weekly_date_like_count >= 10)
    for i in range(len(lines)):
        try:
            # Parse line with csv.reader to handle quoted fields correctly
            fields = next(csv_reader([lines[i]]))
            field_count = len(fields)
            weekly_date_like_count = sum(1 for field in fields if is_weekly_date_like(field))
            
            if weekly_date_like_count >= 10 and field_count >= (weekly_date_like_count + 1):
                print(f"[INFO] Detected weekly header at line {i+1} for {path.name} (date_like={weekly_date_like_count})")
                return i
        except Exception:
            # Skip lines that can't be parsed
            continue
    
    # Step 2: Fallback to existing logic (any date_like_count >= 4 AND field_count >= 6)
    for i in range(len(lines)):
        try:
            # Parse line with csv.reader to handle quoted fields correctly
            fields = next(csv_reader([lines[i]]))
            field_count = len(fields)
            date_like_count = sum(1 for field in fields if is_date_like(field))
            
            if date_like_count >= 4 and field_count >= 6:
                return i
        except Exception:
            # Skip lines that can't be parsed
            continue
    
    # Step 3: Fallback: find line with maximum field_count where field_count >= 6
    best_idx = 0
    best_count = 0
    
    for i in range(len(lines)):
        try:
            fields = next(csv_reader([lines[i]]))
            field_count = len(fields)
            
            if field_count >= 6 and field_count > best_count:
                best_count = field_count
                best_idx = i
        except Exception:
            continue
    
    return best_idx


# ============================================================================
# CSV Reading with Preamble Handling (deprecated, kept for compatibility)
# ============================================================================

def _read_csv_with_preamble_handling(path: Path) -> pd.DataFrame:
    """
    Read CSV file with automatic preamble detection and skipping.
    
    StatCan "noSymbol" exports can include preamble lines (1-column text)
    before the actual comma-separated header row. This function detects
    and skips these preamble lines.
    
    Args:
        path: Path to the CSV file.
    
    Returns:
        DataFrame with preamble lines skipped.
    """
    # Open file and read first ~80 lines to detect header
    with open(path, 'r', encoding='utf-8-sig') as f:
        lines = []
        for i, line in enumerate(f):
            if i >= 80:
                break
            lines.append(line)
    
    # Find header row: first line with >=5 commas where next few lines also have >=5 commas
    header_idx = None
    min_commas = 5
    check_ahead = 3  # Check next 3 lines to confirm it's the header
    
    for i in range(len(lines) - check_ahead):
        comma_count = lines[i].count(',')
        
        if comma_count >= min_commas:
            # Check that next few lines also have enough commas
            all_have_commas = all(
                lines[i + j].count(',') >= min_commas
                for j in range(1, min(check_ahead + 1, len(lines) - i))
            )
            
            if all_have_commas:
                header_idx = i
                break
    
    # If no header found, default to 0
    if header_idx is None:
        header_idx = 0
    
    # Log if preamble was detected
    if header_idx > 0:
        print(f"Detected preamble of {header_idx} lines in {path.name}")
    
    # Try reading with detected header
    try:
        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            skiprows=header_idx,
            engine="python",
            sep=",",
            quotechar='"'
        )
        return df
    except Exception as e:
        # Fallback: try with on_bad_lines="skip"
        try:
            df = pd.read_csv(
                path,
                encoding="utf-8-sig",
                skiprows=header_idx,
                engine="python",
                sep=",",
                quotechar='"',
                on_bad_lines="skip"
            )
            return df
        except Exception:
            # Last resort: try without skiprows
            df = pd.read_csv(
                path,
                encoding="utf-8-sig",
                engine="python",
                on_bad_lines="skip"
            )
            return df


# ============================================================================
# Wide Pivot CSV Loader
# ============================================================================

def load_wide_pivot_csv(path: Path) -> pd.DataFrame:
    """
    Load a wide-pivot StatCan CSV extract and convert to long format.
    
    Detects date columns by regex patterns:
    - Quarterly: "YYYYQN" (e.g., "2019Q1")
    - Weekly: "YYYY-MM-DD" (e.g., "2019-01-02")
    - Monthly: "MMM-YYYY" (e.g., "Jan-2019")
    
    Args:
        path: Path to the CSV file.
        
    Returns:
        DataFrame with columns: date, geo, series, value.
        
    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If no date columns are detected.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Detect header row
    header_idx = _detect_header_row(path)
    if header_idx > 0:
        print(f"Detected preamble of {header_idx} lines in {path.name}")
    
    # Read CSV with detected header (try C engine first, fall back to Python with QUOTE_NONE)
    try:
        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            skiprows=header_idx,
            engine="c",
            sep=",",
            quotechar='"',
            low_memory=False,
        )
    except Exception:
        # Fall back to Python engine with QUOTE_NONE for tolerance
        print(f"[WARN] Falling back to python parser with QUOTE_NONE for {path.name}")
        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            skiprows=header_idx,
            engine="python",
            sep=",",
            quoting=_csv.QUOTE_NONE,
            on_bad_lines="skip",
        )
    
    # Sanitize columns: strip quotes and whitespace
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    
    # Identify date columns with expanded quarterly patterns
    date_cols = []
    date_parsers = {}
    date_col_mapping = {}  # Map original column to normalized quarterly label
    weekly_col_extraction = {}  # Map weekly column to extracted date token
    
    for col in df.columns:
        col_str = str(col).strip()
        
        # Check quarterly patterns (including "Q1 2019" format)
        is_quarterly = False
        for pattern in QUARTERLY_PATTERNS:
            if pattern.match(col_str):
                is_quarterly = True
                # Normalize to canonical format
                normalized = _quarter_label_to_period(col_str)
                date_cols.append(col)
                date_parsers[col] = parse_quarterly_date
                date_col_mapping[col] = normalized
                break
        
        if is_quarterly:
            continue
        
        # Check weekly pattern (YYYY-MM-DD): use search instead of match to handle extra spaces/annotations
        if WEEKLY_DATE_TOKEN.search(col_str):
            date_token = _extract_first_date_token(col_str)
            if date_token:
                date_cols.append(col)
                date_parsers[col] = parse_weekly_date
                weekly_col_extraction[col] = date_token
        # Check monthly pattern (MMM-YYYY)
        if MONTHLY_PATTERN.match(col_str):
            date_cols.append(col)
            date_parsers[col] = parse_monthly_date
        # Check month-name + year token (e.g., "January 2019") - use search to handle extra whitespace/BOM
        elif MONTH_NAME_YEAR_TOKEN.search(col_str):
            date_cols.append(col)
            date_parsers[col] = None  # Will use pd.to_datetime
        # Check month-name date pattern (e.g., "January 2, 2019")
        elif MONTH_NAME_DATE.search(col_str):
            date_cols.append(col)
            date_parsers[col] = None  # Will use pd.to_datetime
    
    # Fallback: try pd.to_datetime for remaining columns (only if we don't have enough date columns)
    if len(date_cols) < 10:
        pd_datetime_cols = []
        for col in df.columns:
            if col not in date_cols:
                col_str = str(col).strip()
                try:
                    parsed = pd.to_datetime(col_str, errors="coerce")
                    if pd.notna(parsed):
                        pd_datetime_cols.append(col)
                except Exception:
                    pass
        
        # Only use pd.to_datetime results if we found at least 10 date columns
        if len(pd_datetime_cols) >= 10:
            for col in pd_datetime_cols:
                if col not in date_cols:
                    date_cols.append(col)
                    date_parsers[col] = None  # Will use pd.to_datetime
    
    # Header promotion fallback: if no date columns detected, try to find header in data rows
    if not date_cols:
        # Re-read with header=None
        try:
            df_raw = pd.read_csv(
                path,
                encoding="utf-8-sig",
                skiprows=header_idx,
                header=None,
                engine="python",
                sep=",",
                quoting=_csv.QUOTE_NONE,
                on_bad_lines="skip",
            )
        except Exception:
            # If re-reading fails, raise error with first 50 columns and first 3 rows
            raise ValueError(
                f"No date columns detected in {path}. "
                f"Expected patterns: YYYYQN, YYYY-MM-DD, MMM-YYYY, or month-name dates. "
                f"First 50 columns: {list(df.columns[:50])}\n"
                f"First 3 rows:\n{df.head(3).to_dict('records')}"
            )
        
        # Scan first 80 rows to find row with >= 10 month-name-year tokens
        found_header_row = None
        for k in range(min(80, len(df_raw))):
            row = df_raw.iloc[k]
            # Convert all cells to string and count MONTH_NAME_YEAR_TOKEN matches
            month_year_count = 0
            for cell in row:
                cell_str = str(cell)
                month_year_count += len(MONTH_NAME_YEAR_TOKEN.findall(cell_str))
            
            if month_year_count >= 10:
                found_header_row = k
                break
        
        if found_header_row is not None:
            # Promote row k to header
            new_header = df_raw.iloc[found_header_row].astype(str).tolist()
            df2 = df_raw.iloc[found_header_row + 1:].copy()
            df2.columns = [str(h).strip().strip('"') for h in new_header]
            df = df2
            
            print(f"[INFO] Promoted row {found_header_row} to header for {path.name} (month-year tokens detected)")
            
            # Re-detect date columns on the new dataframe
            date_cols = []
            date_parsers = {}
            date_col_mapping = {}
            weekly_col_extraction = {}
            
            for col in df.columns:
                col_str = str(col).strip()
                
                # Check quarterly patterns
                is_quarterly = False
                for pattern in QUARTERLY_PATTERNS:
                    if pattern.match(col_str):
                        is_quarterly = True
                        normalized = _quarter_label_to_period(col_str)
                        date_cols.append(col)
                        date_parsers[col] = parse_quarterly_date
                        date_col_mapping[col] = normalized
                        break
                
                if is_quarterly:
                    continue
                
                # Check weekly pattern (YYYY-MM-DD)
                if WEEKLY_DATE_TOKEN.search(col_str):
                    date_token = _extract_first_date_token(col_str)
                    if date_token:
                        date_cols.append(col)
                        date_parsers[col] = parse_weekly_date
                        weekly_col_extraction[col] = date_token
                # Check monthly pattern (MMM-YYYY)
                elif MONTHLY_PATTERN.match(col_str):
                    date_cols.append(col)
                    date_parsers[col] = parse_monthly_date
                # Check month-name + year token (e.g., "January 2019")
                elif MONTH_NAME_YEAR_TOKEN.search(col_str):
                    date_cols.append(col)
                    date_parsers[col] = None  # Will use pd.to_datetime
                # Check month-name date pattern (e.g., "January 2, 2019")
                elif MONTH_NAME_DATE.search(col_str):
                    date_cols.append(col)
                    date_parsers[col] = None  # Will use pd.to_datetime
    
    # If still no date columns detected, raise error with first 50 columns and first 3 rows
    if not date_cols:
        raise ValueError(
            f"No date columns detected in {path}. "
            f"Expected patterns: YYYYQN, YYYY-MM-DD, MMM-YYYY, or month-name dates. "
            f"First 50 columns: {list(df.columns[:50])}\n"
            f"First 3 rows:\n{df.head(3).to_dict('records')}"
        )
    
    # Identify metadata columns (non-date, non-excluded)
    excluded_cols = set(date_cols)
    excluded_patterns = [
        "symbol", "vector", "coordinate", "uom", "scalar", "status",
        "decimals", "terminated", "footnote"
    ]
    
    metadata_cols = [
        col for col in df.columns
        if col not in excluded_cols
        and not any(pattern in str(col).lower() for pattern in excluded_patterns)
    ]
    
    # 1) Identify stub columns (non-date columns)
    stub_cols = [c for c in df.columns if c not in date_cols]
    
    # 2) Clean + forward fill only within stub cols
    for c in stub_cols:
        s = df[c].astype(str).str.strip()
        s = s.replace({"": None, "nan": None, "None": None})
        # also treat ".." as missing only for stub text columns:
        s = s.replace({"..": None})
        df[c] = s
    
    df[stub_cols] = df[stub_cols].ffill()
    
    # Debug log
    print(f"[INFO] ffilled stub cols for {path.name}: {stub_cols}")
    
    # Identify geo column
    geo_col = None
    for col in ["Geography", "GEO", "geo", "geography"]:
        if col in df.columns:
            geo_col = col
            break
    
    # Melt the dataframe (use stub_cols as id_vars since they're already forward-filled)
    id_vars = stub_cols.copy()
    
    # Melt to long format
    df_long = pd.melt(
        df,
        id_vars=id_vars if id_vars else [],
        value_vars=date_cols,
        var_name="date_str",
        value_name="value_raw"
    )
    
    # Parse dates (normalize quarterly labels first, extract weekly tokens, use pd.to_datetime for others)
    def parse_date_str(date_str):
        # If this is a quarterly column, normalize the label first
        if date_str in date_col_mapping:
            normalized_label = date_col_mapping[date_str]
            parser = parse_quarterly_date
            try:
                return parser(normalized_label)
            except Exception:
                return pd.NaT
        # If this is a weekly column with extracted token, use the token
        elif date_str in weekly_col_extraction:
            date_token = weekly_col_extraction[date_str]
            parser = parse_weekly_date
            try:
                return parser(date_token)
            except Exception:
                return pd.NaT
        else:
            parser = date_parsers.get(date_str)
            # If parser is None, use pd.to_datetime (for month-name dates and pd.to_datetime-detected columns)
            if parser is None:
                try:
                    parsed = pd.to_datetime(date_str, errors="coerce")
                    return parsed if pd.notna(parsed) else pd.NaT
                except Exception:
                    return pd.NaT
            elif parser:
                try:
                    return parser(date_str)
                except Exception:
                    return pd.NaT
        return pd.NaT
    
    df_long["date"] = df_long["date_str"].apply(parse_date_str)
    
    # Handle geo
    if geo_col:
        df_long["geo"] = df_long[geo_col].astype(str)
    else:
        df_long["geo"] = "Canada"
    
    # Construct series from stub columns (excluding geo)
    series_cols = [c for c in stub_cols if c != geo_col]
    if series_cols:
        df_long["series"] = df_long[series_cols].apply(
            lambda row: " | ".join([str(val) for val in row if pd.notna(val) and str(val).strip() != ""]),
            axis=1
        )
    else:
        df_long["series"] = "Total"
    
    # Coerce value to float
    df_long["value"] = df_long["value_raw"].apply(coerce_value_to_float)
    
    # Select and return
    result = df_long[["date", "geo", "series", "value"]].copy()
    result = result.dropna(subset=["date"])  # Remove rows with invalid dates
    
    return result


# ============================================================================
# Window Filtering
# ============================================================================

def filter_window(
    df: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Filter DataFrame to a date window.
    
    Args:
        df: DataFrame with date column.
        start: Start date string or Timestamp (inclusive). Defaults to PROJECT_WINDOW_START.
        end: End date string or Timestamp (inclusive). Defaults to PROJECT_WINDOW_END.
        date_col: Name of date column. Defaults to "date".
        
    Returns:
        Filtered DataFrame.
        
    Raises:
        ValueError: If date_col does not exist in DataFrame.
    """
    # Validate date column exists
    if date_col not in df.columns:
        raise ValueError(
            f"Date column '{date_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    
    if start is None:
        start = PROJECT_WINDOW_START
    if end is None:
        end = PROJECT_WINDOW_END
    
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    
    mask = (df[date_col] >= start_dt) & (df[date_col] <= end_dt)
    return df[mask].copy()


# ============================================================================
# Main Guard Example
# ============================================================================

if __name__ == "__main__":
    """Example usage: load each raw file and print shape."""
    from src.config import RAW_DIR, RAW_FILES
    
    print("Loading StatCan CSV files...")
    print("=" * 60)
    
    for key, filename in RAW_FILES.items():
        filepath = RAW_DIR / filename
        
        if not filepath.exists():
            print(f"\n{key}: {filename}")
            print(f"  File not found: {filepath}")
            continue
        
        try:
            # Try wide pivot first (most common)
            try:
                df = load_wide_pivot_csv(filepath)
                print(f"\n{key}: {filename}")
                print(f"  Format: Wide pivot")
                print(f"  Shape: {df.shape}")
                print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
                print(f"  Unique geos: {df['geo'].nunique()}")
                print(f"  Unique series: {df['series'].nunique()}")
            except ValueError:
                # Try tidy format
                df = load_tidy_statcan_csv(filepath)
                print(f"\n{key}: {filename}")
                print(f"  Format: Tidy")
                print(f"  Shape: {df.shape}")
                print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
                print(f"  Unique geos: {df['geo'].nunique()}")
                print(f"  Unique series: {df['series'].nunique()}")
                
        except Exception as e:
            print(f"\n{key}: {filename}")
            print(f"  Error: {type(e).__name__}: {e}")

