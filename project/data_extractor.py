import json
import os
from pathlib import Path

# import pandas as pd

# Create .kaggle directory
kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(exist_ok=True)

# Create file kaggle.json
with open(kaggle_dir / "kaggle.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "username": os.getenv("KAGGLE_USERNAME"),
            "key": os.getenv("KAGGLE_KEY"),
        },
        f,
    )

# Define segure permissions (Linux/Mac)
if not os.name == "nt":  # Se não for Windows
    os.chmod(kaggle_dir / "kaggle.json", 0o600)

# flake8: noqa: E402
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

# Load .env variables
load_dotenv(dotenv_path="../.env")

# Authentication
api = KaggleApi()
api.authenticate()

# Load datasets
api.dataset_download_files(
    "olistbr/brazilian-ecommerce", path="./data/raw", unzip=True
)
