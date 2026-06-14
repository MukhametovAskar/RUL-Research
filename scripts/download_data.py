#!/usr/bin/env python3
"""Download NASA C-MAPSS dataset (FD001-FD004) and place into ./data/"""
import os
import zipfile
import urllib.request

URL = "https://s3-us-west-2.amazonaws.com/ailab-data-public/cmapss/NASA-CMAPSS.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "NASA-CMAPSS.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading C-MAPSS dataset ({URL})...")
        urllib.request.urlretrieve(URL, zip_path)
        print("Download complete.")
    else:
        print("ZIP already exists, skipping download.")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    print(f"Extracted to {DATA_DIR}")

    expected = ["FD001", "FD002", "FD003", "FD004"]
    for fd in expected:
        fd_dir = os.path.join(DATA_DIR, fd)
        if os.path.isdir(fd_dir):
            print(f"  {fd}/ OK")
        else:
            # Files might be at top level — try to reorganize
            for suffix in ["train", "test", "RUL"]:
                src = os.path.join(DATA_DIR, f"{suffix}_{fd}.txt")
                if os.path.exists(src):
                    os.makedirs(fd_dir, exist_ok=True)
                    dst = os.path.join(fd_dir, f"{suffix}.txt")
                    os.rename(src, dst)
                    print(f"  {fd}/{suffix}.txt moved into place")

    print("\nDone. Expected structure:")
    print("  data/FD001/train.txt, test.txt, RUL.txt")
    print("  data/FD002/...")
    print("  data/FD003/...")
    print("  data/FD004/...")


if __name__ == "__main__":
    download()
