"""
Robust streaming downloader and extractor for MLVis 1K benchmark.
Directly downloads repository zip archive, sanitizes Windows-incompatible colons (':')
to double underscores ('__'), extracts all 999 datasets and vis_all.csv,
strictly avoids extracting data/1k/tmp/, and removes temporary artifacts.
"""

import os
import io
import json
import zipfile
import logging
import urllib.request
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MLVisZipDownloader")

TARGET_DIR = Path("data/mlvis")
RAW_DIR = TARGET_DIR / "raw"
DATASETS_DIR = RAW_DIR / "datasets"
TEMP_ZIP = TARGET_DIR / "mlvis_archive.zip"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/xeniaqian94/kdd21-MLVis/archive/refs/heads/main.zip"

logger.info(f"Downloading MLVis repository archive from {URL}...")
req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})

with urllib.request.urlopen(req) as resp, open(TEMP_ZIP, "wb") as f_out:
    downloaded = 0
    while True:
        chunk = resp.read(1024 * 1024) # 1MB chunks
        if not chunk:
            break
        f_out.write(chunk)
        downloaded += len(chunk)
        if downloaded % (5 * 1024 * 1024) == 0:
            logger.info(f"Downloaded {downloaded / (1024 * 1024):.1f} MB...")

logger.info(f"Download complete: {TEMP_ZIP.stat().st_size / (1024 * 1024):.2f} MB. Processing zip contents...")

dataset_index = {}
vis_all_extracted = False
extracted_datasets = 0

with zipfile.ZipFile(TEMP_ZIP, "r") as zf:
    for member in zf.infolist():
        name = member.filename
        # Normalize path separators
        norm_name = name.replace("\\", "/")

        # 1. Extract vis_all.csv
        if "data/1k/vis_all.csv" in norm_name and not member.is_dir():
            target_path = RAW_DIR / "vis_all.csv"
            with zf.open(member) as src, open(target_path, "wb") as dst:
                dst.write(src.read())
            vis_all_extracted = True
            logger.info(f"Extracted vis_all.csv to {target_path}")

        # 2. Extract datasets from data/1k/dataset/ (avoid data/1k/tmp/)
        elif "data/1k/dataset/" in norm_name and norm_name.endswith(".csv") and not member.is_dir():
            orig_filename = os.path.basename(norm_name)
            dataset_id = orig_filename.replace(".csv", "")
            safe_filename = orig_filename.replace(":", "__")
            target_path = DATASETS_DIR / safe_filename

            data = zf.read(member)
            with open(target_path, "wb") as dst:
                dst.write(data)

            dataset_index[dataset_id] = {
                "dataset_id": dataset_id,
                "original_filename": orig_filename,
                "safe_filename": safe_filename,
                "byte_size": len(data),
            }
            extracted_datasets += 1

# Remove temporary archive
if TEMP_ZIP.exists():
    TEMP_ZIP.unlink()
    logger.info("Removed temporary zip archive.")

# Save dataset index
index_path = RAW_DIR / "dataset_index.json"
with open(index_path, "w", encoding="utf-8") as f:
    json.dump(dataset_index, f, indent=2)

logger.info(f"Saved dataset index with {len(dataset_index)} datasets to {index_path}.")

# Verification
df_vis = pd.read_csv(RAW_DIR / "vis_all.csv", low_memory=False)
logger.info(f"Verification: vis_all.csv has {len(df_vis)} visualization records.")
logger.info(f"Verification: Extracted {extracted_datasets} dataset files into {DATASETS_DIR}.")
