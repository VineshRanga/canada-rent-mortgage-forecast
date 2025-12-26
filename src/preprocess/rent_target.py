"""
Extract rent target data from Statistics Canada CSV.

Loads asking rent prices by CMA and unit type, filters to target unit types,
and creates a quarterly target dataset for modeling.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import re

from src.config import RAW_DIR, PROCESSED_DIR, RAW_FILES, quarter_end

# Flag to include Canada aggregate (default: False, keep only CMAs)
INCLUDE_CANADA_AGGREGATE = False

from src.io.statcan_read import load_wide_pivot_csv


# ============================================================================
# Helper Functions for Series Parsing
# ============================================================================

def _norm_txt(x: str) -> str:
    """Normalize text: strip, remove quotes, normalize whitespace."""
    x = str(x).strip().strip('"').replace("\u00a0", " ")
    x = re.sub(r"\s+", " ", x)
    return x


def _is_cma(tok: str) -> bool:
    """Check if token is a CMA (contains 'census metropolitan area' or '(cma)')."""
    t = tok.lower()
    return ("census metropolitan area" in t) or ("(cma)" in t)


def _is_unit(tok: str) -> bool:
    """Check if token is a unit type (contains 'apartment' and 'bed', or is 'room')."""
    t = tok.lower()
    return ("apartment" in t and "bed" in t) or (t == "room")


def _extract_from_series(series_val: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract CMA and unit type from series string.
    
    Splits by " | ", normalizes parts, and identifies CMA and unit tokens.
    
    Args:
        series_val: Series string value.
    
    Returns:
        Tuple of (cma, unit_type) or (None, None) if not found.
    """
    parts = [_norm_txt(p) for p in re.split(r"\s\|\s", _norm_txt(series_val)) if _norm_txt(p)]
    cma = next((p for p in parts if _is_cma(p)), None)
    unit = next((p for p in parts if _is_unit(p)), None)
    return cma, unit


def extract_rent_target(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Extract rent target data from StatCan CSV.
    
    Loads asking rent prices, filters to target CMAs and unit types,
    and creates a quarterly target dataset.
    
    Args:
        input_path: Path to raw rent CSV file.
                   Defaults to RAW_DIR / RAW_FILES["rent"].
        output_path: Path to save output parquet file.
                     Defaults to PROCESSED_DIR / "rent_target_quarterly.parquet".
    
    Returns:
        DataFrame with columns:
        - quarter_end_date: Quarter-end Timestamp
        - cma: CMA name (string)
        - unit_type: Unit type (string)
        - y: Target rent value (float)
    
    Raises:
        FileNotFoundError: If input file does not exist.
        ValueError: If extraction produces 0 rows.
    """
    if input_path is None:
        input_path = RAW_DIR / RAW_FILES["rent"]
    if output_path is None:
        output_path = PROCESSED_DIR / "rent_target_quarterly.parquet"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Extracting rent target data...")
    print("=" * 60)
    print(f"Loading from: {input_path}")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Rent data file not found: {input_path}")
    
    # 1) Load wide pivot CSV
    df = load_wide_pivot_csv(input_path)
    print(f"  Loaded shape: {df.shape}")
    
    # 2) Clean data
    print("\nCleaning data...")
    if "series" in df.columns:
        df["series"] = df["series"].astype(str).str.strip()
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    
    # Drop rows where series=="" or value is NaN
    initial_rows = len(df)
    if "series" in df.columns:
        df = df[df["series"].ne("")].copy()
    if "value" in df.columns:
        df = df[df["value"].notna()].copy()
    print(f"  After cleaning: {len(df)} rows (dropped {initial_rows - len(df)})")
    
    # 1) Debug sample of series values
    s = df["series"].astype(str).str.strip()
    print("\nSERIES SAMPLE (first 30):", s.dropna().unique()[:30].tolist())
    
    # 3) Extract CMA and unit_type from series using helper functions
    print("\nExtracting CMA and unit_type from series...")
    tmp = df["series"].astype(str).apply(_extract_from_series)
    df["cma_raw"] = tmp.apply(lambda x: x[0])
    df["unit_raw"] = tmp.apply(lambda x: x[1])
    
    # 5) Drop rows without both
    before_extraction = len(df)
    df = df.dropna(subset=["cma_raw", "unit_raw"]).copy()
    print(f"  After extraction: {len(df)} rows (dropped {before_extraction - len(df)})")
    
    # 6) Clean CMA
    df["cma"] = (
        df["cma_raw"]
        .astype(str)
        .str.replace(", Census metropolitan area (CMA)", "", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.rstrip(",")
    )
    
    # 7) Map unit types (non-brittle)
    u = df["unit_raw"].astype(str).str.strip().str.strip('"').str.replace(r"\s+", " ", regex=True)
    mask_1 = u.str.contains(r"\b1\b.*bed", case=False, na=False)
    mask_2 = (
        u.str.contains(r"\b2\b.*bed", case=False, na=False) |
        u.str.contains(r"two[-\s]*bed", case=False, na=False)
    )
    
    before_unit_filter = len(df)
    df = df[mask_1 | mask_2].copy()
    print(f"  After unit_type filter: {len(df)} rows (dropped {before_unit_filter - len(df)})")
    
    # Recompute masks on filtered dataframe
    u_filtered = df["unit_raw"].astype(str).str.strip().str.strip('"').str.replace(r"\s+", " ", regex=True)
    mask_1_filtered = u_filtered.str.contains(r"\b1\b.*bed", case=False, na=False)
    mask_2_filtered = (
        u_filtered.str.contains(r"\b2\b.*bed", case=False, na=False) |
        u_filtered.str.contains(r"two[-\s]*bed", case=False, na=False)
    )
    
    df.loc[mask_1_filtered, "unit_type"] = "Apartment - 1 bedroom"
    df.loc[mask_2_filtered, "unit_type"] = "Apartment - 2 bedrooms"
    
    # 8) Build output with a single date column
    print("\nBuilding output schema...")
    if "date" not in df.columns:
        raise ValueError("'date' column not found in DataFrame")
    
    # Convert date to datetime and ensure quarter-end
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"].apply(quarter_end))
    
    out = df[["date", "cma", "unit_type", "value"]].copy()
    out = out.rename(columns={"date": "quarter_end_date", "value": "y"})
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["y"]).copy()
    
    # Sort by date, cma, unit_type
    out = out.sort_values(["quarter_end_date", "cma", "unit_type"]).reset_index(drop=True)
    
    # 9) Summary prints
    print("\n" + "=" * 60)
    print("RENT TARGET EXTRACTION SUMMARY")
    print("=" * 60)
    print("unit_type counts:", out["unit_type"].value_counts().to_dict())
    print("CMAs:", out["cma"].nunique(), "rows:", len(out))
    print("date min/max:", out["quarter_end_date"].min(), out["quarter_end_date"].max())
    
    # 10) Save and return
    if len(out) == 0:
        raise ValueError(
            "Rent target extraction produced 0 rows; inspect series sample printed above."
        )
    
    print(f"\nSaving to: {output_path}")
    out.to_parquet(output_path, index=False, engine="pyarrow")
    print("Rent target data saved successfully.")
    
    return out


if __name__ == "__main__":
    """Run extraction as standalone script."""
    df = extract_rent_target()
    print("\nFirst few rows:")
    print(df.head(10))

