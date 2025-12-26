"""
Export LinkedIn-ready PNG charts to a dedicated directory.

Copies Plotly-generated PNG files from outputs/plots/plotly/ to outputs/linkedin_pack/
for easy sharing on LinkedIn.
"""

import sys
from pathlib import Path
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OUTPUT_DIR


def export_linkedin_pack() -> None:
    """
    Export LinkedIn-ready PNG charts to outputs/linkedin_pack/.
    
    Checks for required PNG files and copies them to the destination directory.
    Prints warnings if PNGs are missing (likely due to kaleido not being installed).
    """
    # Source and destination directories
    source_dir = OUTPUT_DIR / "plots" / "plotly"
    dest_dir = OUTPUT_DIR / "linkedin_pack"
    
    # Required PNG files
    required_pngs = [
        "rent_toronto_1bed.png",
        "rent_toronto_2bed.png",
        "rent_gta_proxy_1bed.png",
        "rent_gta_proxy_2bed.png"
    ]
    
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("EXPORTING LINKEDIN PACK")
    print("=" * 80)
    print(f"Source directory: {source_dir}")
    print(f"Destination directory: {dest_dir}\n")
    
    # Check and copy each PNG
    copied_files = []
    missing_files = []
    
    for png_file in required_pngs:
        source_path = source_dir / png_file
        dest_path = dest_dir / png_file
        
        if source_path.exists():
            try:
                shutil.copy2(source_path, dest_path)
                copied_files.append(png_file)
                print(f"✓ Copied: {dest_path}")
            except Exception as e:
                print(f"✗ Failed to copy {png_file}: {e}")
                missing_files.append(png_file)
        else:
            missing_files.append(png_file)
            print(f"✗ Missing: {source_path}")
    
    print()
    
    # Summary
    if len(copied_files) == len(required_pngs):
        print("=" * 80)
        print("SUCCESS: All PNG files exported")
        print("=" * 80)
        print(f"Exported {len(copied_files)} files to: {dest_dir}")
        print("\nFiles ready for LinkedIn:")
        for png_file in copied_files:
            print(f"  - {dest_dir / png_file}")
    elif len(copied_files) > 0:
        print("=" * 80)
        print("PARTIAL SUCCESS: Some PNG files exported")
        print("=" * 80)
        print(f"Exported {len(copied_files)}/{len(required_pngs)} files to: {dest_dir}")
        print("\nExported files:")
        for png_file in copied_files:
            print(f"  - {dest_dir / png_file}")
        print("\nMissing files:")
        for png_file in missing_files:
            print(f"  - {png_file}")
        print("\n[WARN] Some PNG files are missing. This usually means kaleido is not installed.")
        print("       To generate PNG files, install kaleido and rerun the pipeline:")
        print("       pip install kaleido")
        print("       python run_pipeline.py")
    else:
        print("=" * 80)
        print("WARNING: No PNG files found")
        print("=" * 80)
        print("[WARN] PNG files are missing. This usually means kaleido is not installed.")
        print("       To generate PNG files, install kaleido and rerun the pipeline:")
        print("       pip install kaleido")
        print("       python run_pipeline.py")
        print("\nNote: HTML files may still be available in:")
        print(f"      {source_dir}")
    
    print("=" * 80)


if __name__ == "__main__":
    export_linkedin_pack()

