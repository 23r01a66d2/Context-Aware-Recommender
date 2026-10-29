"""
Optimized extractor for MLVis 1K dataset and visualization records.
Batches requests and skips already-extracted datasets.
"""

import os
import io
import json
import logging
import subprocess
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MLVisExtractor")

REPO_DIR = Path("temp_mlvis_repo")
TARGET_DIR = Path("data/mlvis")
RAW_DIR = TARGET_DIR / "raw"
DATASETS_DIR = RAW_DIR / "datasets"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Ensure vis_all.csv is present
vis_all_path = RAW_DIR / "vis_all.csv"
if not vis_all_path.exists() or vis_all_path.stat().st_size == 0:
    logger.info("Extracting vis_all.csv...")
    vis_all_bytes = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "show", "HEAD:data/1k/vis_all.csv"]
    )
    with open(vis_all_path, "wb") as f:
        f.write(vis_all_bytes)

df_vis = pd.read_csv(vis_all_path, low_memory=False)
logger.info(f"Verified vis_all.csv: {df_vis.shape[0]} rows, {df_vis.shape[1]} columns.")

# 2. List all datasets in git tree
logger.info("Querying tree data/1k/dataset...")
tree_output = subprocess.check_output(
    ["git", "-C", str(REPO_DIR), "ls-tree", "HEAD:data/1k/dataset"]
).decode("utf-8")

lines = [line.strip() for line in tree_output.split("\n") if line.strip()]
entries = []
for line in lines:
    parts = line.split()
    if len(parts) >= 4:
        mode, obj_type, blob_hash, orig_filename = parts[0], parts[1], parts[2], parts[3]
        entries.append((blob_hash, orig_filename))

logger.info(f"Total entries in tree: {len(entries)}")

# 3. Check what is already downloaded
needed = []
dataset_index = {}

for blob_hash, orig_filename in entries:
    dataset_id = orig_filename.replace(".csv", "")
    safe_filename = orig_filename.replace(":", "__")
    safe_filepath = DATASETS_DIR / safe_filename

    if safe_filepath.exists() and safe_filepath.stat().st_size > 0:
        dataset_index[dataset_id] = {
            "dataset_id": dataset_id,
            "original_filename": orig_filename,
            "safe_filename": safe_filename,
            "blob_hash": blob_hash,
            "byte_size": safe_filepath.stat().st_size,
        }
    else:
        needed.append((blob_hash, orig_filename, dataset_id, safe_filename, safe_filepath))

logger.info(f"Already extracted: {len(dataset_index)}. Remaining to extract: {len(needed)}")

# 4. If any needed, fetch via git cat-file batch input
if needed:
    input_hashes = "\n".join(item[0] for item in needed) + "\n"
    proc = subprocess.Popen(
        ["git", "-C", str(REPO_DIR), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    stdout_data, stderr_data = proc.communicate(input=input_hashes.encode("utf-8"))
    if proc.returncode != 0:
        logger.error(f"git cat-file error: {stderr_data.decode('utf-8', errors='ignore')}")

    # Parse stdout_data stream
    stream = io.BytesIO(stdout_data)
    for blob_hash, orig_filename, dataset_id, safe_filename, safe_filepath in needed:
        header_line = stream.readline().decode("utf-8", errors="ignore").strip()
        if not header_line:
            break
        header_parts = header_line.split()
        if len(header_parts) < 3 or header_parts[1] != "blob":
            logger.warning(f"Failed to read blob for {blob_hash}: {header_line}")
            continue

        size = int(header_parts[2])
        data = stream.read(size)
        stream.read(1) # trailing newline

        with open(safe_filepath, "wb") as f_out:
            f_out.write(data)

        dataset_index[dataset_id] = {
            "dataset_id": dataset_id,
            "original_filename": orig_filename,
            "safe_filename": safe_filename,
            "blob_hash": blob_hash,
            "byte_size": size,
        }

index_path = RAW_DIR / "dataset_index.json"
with open(index_path, "w", encoding="utf-8") as f:
    json.dump(dataset_index, f, indent=2)

logger.info(
    f"All done! Extracted {len(dataset_index)}/{len(entries)} datasets to {DATASETS_DIR}. "
    f"Index saved to {index_path}."
)
