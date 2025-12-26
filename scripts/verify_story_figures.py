"""Verify story figure files exist in both outputs and figures directories."""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OUTPUT_DIR

REQUIRED_FILES = [
    "toronto_rent_qoq_vs_goc5y.png",
    "gta_rent_qoq_vs_goc5y.png",
]

def format_file_info(filepath: Path) -> str:
    """Format file size and modified time."""
    if not filepath.exists():
        return "MISSING"
    
    stat = filepath.stat()
    size_kb = stat.st_size / 1024
    mtime = datetime.fromtimestamp(stat.st_mtime)
    return f"{size_kb:.1f} KB, modified {mtime.strftime('%Y-%m-%d %H:%M:%S')}"

def main():
    """Check that all story figures exist in both locations."""
    output_dir = OUTPUT_DIR / "plots" / "mpl_story"
    figures_dir = Path("figures")
    
    print("=" * 80)
    print("VERIFYING STORY FIGURES")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Figures directory: {figures_dir}\n")
    
    all_present = True
    
    for filename in REQUIRED_FILES:
        output_path = output_dir / filename
        figures_path = figures_dir / filename
        
        output_exists = output_path.exists()
        figures_exists = figures_path.exists()
        
        if output_exists and figures_exists:
            print(f"✓ {filename}")
            print(f"  Output: {format_file_info(output_path)}")
            print(f"  Figures: {format_file_info(figures_path)}")
        else:
            all_present = False
            print(f"✗ {filename}")
            if not output_exists:
                print(f"  MISSING in output: {output_path}")
            if not figures_exists:
                print(f"  MISSING in figures: {figures_path}")
        print()
    
    print("=" * 80)
    if all_present:
        print("SUCCESS: All story figures present in both locations")
        print("=" * 80)
        sys.exit(0)
    else:
        print("FAILURE: Some story figures are missing")
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    main()

