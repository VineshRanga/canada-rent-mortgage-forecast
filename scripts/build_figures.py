"""Copy curated PNGs into figures/ for GitHub/README embedding."""

import sys
from pathlib import Path
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OUTPUT_DIR


def build_figures() -> None:
    """Copy curated PNGs into figures/ for GitHub/README embedding."""
    file_mappings = [
        (OUTPUT_DIR / "plots" / "plotly" / "rent_toronto_1bed.png", "rent_toronto_1bed_forecast.png"),
        (OUTPUT_DIR / "plots" / "plotly" / "rent_gta_proxy_1bed.png", "rent_gta_1bed_forecast.png"),
        (OUTPUT_DIR / "plots" / "mpl_story" / "toronto_rent_qoq_vs_goc5y.png", "toronto_rent_qoq_vs_goc5y.png"),
        (OUTPUT_DIR / "plots" / "mpl_story" / "gta_rent_qoq_vs_goc5y.png", "gta_rent_qoq_vs_goc5y.png"),
        (OUTPUT_DIR / "plots" / "mpl_eval" / "rent_uplift_lag4_heatmap.png", "rent_uplift_lag4_heatmap.png"),
        (OUTPUT_DIR / "plots" / "mpl_eval" / "rent_uplift_top15_cmas.png", "rent_uplift_top15_cmas.png"),
        (OUTPUT_DIR / "plots" / "mortgage_actual_vs_predicted.png", "mortgage_actual_vs_predicted.png"),
    ]
    
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("BUILDING FIGURES DIRECTORY")
    print("=" * 80)
    print(f"Destination: {figures_dir.absolute()}\n")
    
    copied_files = []
    missing_files = []
    
    for source_path, target_filename in file_mappings:
        target_path = figures_dir / target_filename
        
        if source_path.exists():
            try:
                shutil.copy2(source_path, target_path)
                copied_files.append(target_filename)
                print(f"✓ Copied: {target_filename}")
            except Exception as e:
                print(f"✗ Failed to copy {target_filename}: {e}")
                missing_files.append((target_filename, str(source_path)))
        else:
            print(f"[WARN] missing: {source_path}")
            missing_files.append((target_filename, str(source_path)))
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if len(copied_files) > 0:
        print(f"\nSuccessfully copied {len(copied_files)} file(s):")
        for filename in copied_files:
            print(f"  - {figures_dir / filename}")
    
    if len(missing_files) > 0:
        print(f"\nMissing {len(missing_files)} file(s):")
        for filename, source_path in missing_files:
            print(f"  - {filename} (source: {source_path})")
    
    print(f"\nFigures directory: {figures_dir.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    build_figures()

