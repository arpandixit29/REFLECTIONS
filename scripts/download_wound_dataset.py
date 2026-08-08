"""
Reflections - Kaggle Wound Dataset Downloader
=============================================
Downloads 'yasinpratomo/wound-dataset' using kagglehub and prepares
the dataset directory structure for Reflections model training.
"""

import os
import shutil
from pathlib import Path

def download_dataset():
    print("[Reflections] Initializing Kagglehub Dataset Downloader...")
    try:
        import kagglehub
    except ImportError:
        print("[Reflections] Installing kagglehub...")
        os.system("pip install kagglehub")
        import kagglehub

    print("[Reflections] Downloading dataset 'yasinpratomo/wound-dataset' from Kaggle...")
    path = kagglehub.dataset_download("yasinpratomo/wound-dataset")
    print(f"[Reflections] [OK] Dataset downloaded successfully to: {path}")

    # Prepare data target directory
    target_dir = Path("data/wound_dataset")
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Reflections] Processing downloaded files from {path} into {target_dir}...")
    
    # Symlink or copy dataset files into local project directory
    source_path = Path(path)
    if source_path.exists():
        for item in source_path.iterdir():
            dest = target_dir / item.name
            if not dest.exists():
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
        print(f"[Reflections] [OK] Dataset files ready at '{target_dir}'")
    
    return str(target_dir)

if __name__ == "__main__":
    download_dataset()
