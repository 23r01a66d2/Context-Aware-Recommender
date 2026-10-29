"""
Script to execute Fashion Dataset Onboarding Flow from archive/data.csv and archive/data.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_onboarding import onboard_fashion_dataset

if __name__ == "__main__":
    print("Executing Fashion dataset onboarding flow...")
    result = onboard_fashion_dataset(
        csv_path="archive/data.csv",
        image_dir="archive/data",
        client_id="fashion_client",
        client_name="Fashion Multimodal Catalog",
        sample_size=1000
    )
    print(f"Fashion onboarding completed: {result}")
