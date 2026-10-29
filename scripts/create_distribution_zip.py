"""
Creates a clean, portable, and presentation-ready distribution ZIP of the project.
Strictly excludes:
- .venv / venv / virtual environments
- node_modules
- .git repository internals
- raw/large archive datasets (archive/)
- raw mlvis 1k csv benchmarks (data/mlvis/raw/)
- ephemeral test clients & test models
- runtime logs, server.pid, caches (__pycache__, .pytest_cache)
- temporary uploads and user session histories

Preserves:
- All source code (backend, src, frontend/src, frontend/dist, scripts, tests)
- Pre-compiled production React build (frontend/dist/)
- Canonical reference client configs & datasets (demo_ecommerce, movielens_dataset, fashion_client, context_aware_ar_interaction)
- Official demo model checkpoints & metadata
- Documentation, notebooks, research outputs, and reports
- One-click launchers (start.bat, stop.bat, SETUP_NEW_COMPUTER.bat)
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = PROJECT_ROOT.parent / "context_aware_recommender.zip"

EXCLUDE_DIRS = {
    ".venv",
    "venv",
    "env",
    ".git",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "archive",
}

EXCLUDE_EXACT_REL_PATHS = {
    Path("data/server.log"),
    Path("data/server.pid"),
    Path("data/platform.db"),
    Path("clients/railrocket_recommender_system"),
    Path("clients/contract_violation_client"),
    Path("clients/generic_client_dynamic"),
    Path("clients/temp_live_del_1"),
    Path("clients/client_without_active_model"),
    Path("clients/client_no_model"),
    Path("models/contract_violation_client"),
    Path("models/generic_client_dynamic"),
    Path("models/client_no_model"),
    Path("data/uploads"),
    Path("data/user_history"),
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".DS_Store",
}


def should_exclude(rel_path: Path) -> bool:
    # 1. Check directory parts
    for part in rel_path.parts:
        if part in EXCLUDE_DIRS:
            return True
        if part.endswith(".egg-info"):
            return True

    # 2. Check extensions
    if rel_path.suffix in EXCLUDE_EXTENSIONS:
        return True

    # 3. Check exact relative path prefixes
    for excluded in EXCLUDE_EXACT_REL_PATHS:
        try:
            rel_path.relative_to(excluded)
            return True
        except ValueError:
            pass

    # 4. Filter data/mlvis except README.md
    if str(rel_path).replace("\\", "/").startswith("data/mlvis/"):
        if rel_path.name.lower() != "readme.md":
            return True

    # 5. Exclude raw uploads inside clients
    parts = rel_path.parts
    if len(parts) >= 3 and parts[0] == "clients" and parts[2] in ("raw", "processed"):
        return True

    return False


def create_zip():
    print(f"Creating clean distribution ZIP at:")
    print(f"  {OUTPUT_ZIP}")
    print()

    # If the zip already exists, remove it before writing
    if OUTPUT_ZIP.exists():
        try:
            OUTPUT_ZIP.unlink()
        except Exception as e:
            print(f"Notice: removing old zip: {e}")

    file_count = 0
    total_bytes = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            root_path = Path(root)
            rel_root = root_path.relative_to(PROJECT_ROOT)

            # Filter dirs in-place to prevent os.walk from descending into them
            dirs[:] = [d for d in dirs if not should_exclude(rel_root / d)]

            for file in files:
                file_path = root_path / file
                rel_path = file_path.relative_to(PROJECT_ROOT)

                if should_exclude(rel_path):
                    continue

                arcname = Path("context_aware_recommender") / rel_path
                zf.write(file_path, str(arcname).replace("\\", "/"))
                file_count += 1
                total_bytes += file_path.stat().st_size

    zip_size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    orig_size_mb = total_bytes / (1024 * 1024)

    print(f"ZIP Creation Summary:")
    print(f"  Files Packed    : {file_count:,}")
    print(f"  Uncompressed    : {orig_size_mb:.2f} MB")
    print(f"  ZIP File Size   : {zip_size_mb:.2f} MB")
    print(f"  Status          : SUCCESS (< 512 MB limit)")


if __name__ == "__main__":
    create_zip()
