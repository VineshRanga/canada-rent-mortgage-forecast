import pandas as pd
from pathlib import Path

def load_df():
    p_parquet = Path("data/processed/rent_target_quarterly.parquet")
    p_preds = Path("outputs/rent_predictions.csv")

    if p_parquet.exists():
        df = pd.read_parquet(p_parquet)
        source = str(p_parquet)
        # rent_target schema uses 'cma'
        cma_col = "cma"
    elif p_preds.exists():
        df = pd.read_csv(p_preds)
        source = str(p_preds)
        cma_col = "cma"
    else:
        raise FileNotFoundError("Neither data/processed/rent_target_quarterly.parquet nor outputs/rent_predictions.csv exists.")

    df[cma_col] = df[cma_col].astype(str).str.strip()
    return df, source, cma_col

def main():
    df, source, cma_col = load_df()
    cmas = sorted([c for c in df[cma_col].dropna().unique().tolist() if c and c.lower() != "nan"])

    print(f"Source: {source}")
    print(f"Unique CMAs: {len(cmas)}")
    print("\nCMA list (sorted):")
    for c in cmas:
        print(f"  - {c}")

    targets = ["Mississauga", "Brampton", "Oakville", "Burlington", "Oshawa", "Hamilton", "Toronto"]
    print("\nPresence checks (case-insensitive exact match):")
    cmas_lower = {c.lower(): c for c in cmas}
    for t in targets:
        hit = cmas_lower.get(t.lower())
        print(f"  {t}: {'YES -> ' + hit if hit else 'NO'}")

    print("\nSubstring matches (case-insensitive contains):")
    for t in targets:
        matches = [c for c in cmas if t.lower() in c.lower()]
        if matches:
            print(f"  {t}: {matches}")

    print("\nNote: This dataset is CMA-based (Census Metropolitan Areas), not municipalities.")
    print("Cities like Mississauga/Brampton/Oakville/Burlington are typically part of the Toronto CMA,")
    print("so they may not appear as separate entries.")

if __name__ == "__main__":
    main()

