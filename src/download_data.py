"""Downloads and extracts Cricsheet match data.

Cricsheet (https://cricsheet.org) publishes free, ball-by-ball data for
international and major franchise cricket under the Open Database
License (ODbL). No API key or registration required.
"""

import io
import zipfile
from pathlib import Path

import requests

BASE_URL = "https://cricsheet.org/downloads"

# Cricsheet's zip filenames for men's international matches, by format.
FORMAT_FILES = {
    "test": "tests_male_json.zip",
    "odi": "odis_male_json.zip",
    "t20i": "t20s_male_json.zip",
}

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download_format(fmt: str) -> Path:
    """Download and extract one format's JSON archive. Returns the extracted folder."""
    if fmt not in FORMAT_FILES:
        raise ValueError(f"Unknown format '{fmt}'. Choose from {list(FORMAT_FILES)}.")

    filename = FORMAT_FILES[fmt]
    url = f"{BASE_URL}/{filename}"
    out_dir = RAW_DIR / fmt
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    print(f"Extracting to {out_dir} ...")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(out_dir)

    n_files = len(list(out_dir.glob("*.json")))
    print(f"Done: {n_files} match files in {out_dir}")
    return out_dir


if __name__ == "__main__":
    import sys

    formats = sys.argv[1:] or ["test"]
    for fmt in formats:
        download_format(fmt)