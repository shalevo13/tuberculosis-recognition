"""
Download and preprocess NIH / NLM Montgomery County & Shenzhen Tuberculosis Chest X-ray Datasets.
Fetches high-resolution scans from the official NIH AWS repository, resizes them on-the-fly to 224x224,
and saves a clean CSV metadata table.
"""

import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from PIL import Image
import pandas as pd
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")

BASE_URL_MONTGOMERY = (
    "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/"
    "Montgomery-County-CXR-Set/MontgomerySet/CXR_png/"
)

BASE_URL_SHENZHEN = (
    "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/"
    "Shenzhen-Hospital-CXR-Set/CXR_png/"
)

URL_NIAID_BELARUS = (
    "https://raw.githubusercontent.com/frapa/tbcnn/master/belarus/belarus.zip"
)

DATASET_CONFIGS = [
    {
        "source": "Montgomery County",
        "base_url": BASE_URL_MONTGOMERY,
        "pattern": r"MCUCXR_\d+_[01]\.png",
    },
    {
        "source": "Shenzhen Hospital",
        "base_url": BASE_URL_SHENZHEN,
        "pattern": r"CHNCXR_\d+_[01]\.png",
    },
]


def get_image_list(base_url, pattern):
    """Scrape image filenames from an official NIH directory index."""
    req = urllib.request.Request(base_url + "index.html", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8")
    return sorted(list(set(re.findall(pattern, html))))


def download_and_process_image(item, target_size=(224, 224), max_retries=4):
    """Download a single high-res scan, convert to 8-bit grayscale, resize, and save."""
    fname = item["filename"]
    base_url = item["base_url"]
    source = item["source"]
    dest_path = os.path.join(IMAGES_DIR, fname)
    label = int(fname.split("_")[-1].replace(".png", ""))

    if os.path.exists(dest_path):
        return {"filename": fname, "label": label, "source": source, "status": "cached"}

    url = base_url + fname
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()

            img = Image.open(BytesIO(data))
            if img.mode != "L":
                img = img.convert("L")

            resized = img.resize(target_size, Image.Resampling.BILINEAR)
            resized.save(dest_path, format="PNG", optimize=True)
            return {"filename": fname, "label": label, "source": source, "status": "downloaded"}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"filename": fname, "error": str(e), "source": source, "status": "failed"}
            time.sleep(1.0 + attempt * 1.5)


def download_and_process_niaid_belarus(target_size=(224, 224), max_retries=4):
    """
    Download and process the NIAID TB Portals (Belarus Set) dataset.
    This clinical radiograph cohort contains culture-confirmed pulmonary Tuberculosis cases
    (MDR and drug-sensitive TB) collected by NIAID in collaboration with the Republican
    Scientific and Practical Centre for Pulmonology and Tuberculosis (Minsk, Belarus).
    Excludes non-frontal artifact scans (e.g. image 2) for clinical consistency.
    """
    import zipfile

    source = "NIAID TB Portals (Belarus)"
    zip_path = os.path.join(DATA_DIR, "belarus.zip")

    if not os.path.exists(zip_path):
        print(f"Fetching NIAID TB Portals (Belarus Set) archive from {URL_NIAID_BELARUS}...")
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(URL_NIAID_BELARUS, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                with open(zip_path, "wb") as f:
                    f.write(data)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error fetching NIAID archive: {e}")
                    return []
                time.sleep(2.0 + attempt * 2.0)

    print("Extracting and resizing NIAID TB Portals radiographs...")
    records = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        png_names = sorted([f for f in zf.namelist() if f.endswith(".png")])
        # Filter out known non-frontal image artifact (belarus/2.png)
        valid_names = [f for f in png_names if not f.endswith("/2.png")]
        print(f"  Discovered {len(valid_names)} confirmed TB chest radiographs from NIAID TB Portals.")

        for idx, internal_fname in enumerate(tqdm(valid_names, desc="Processing NIAID CXRs"), start=1):
            out_fname = f"BLRCXR_{idx:04d}_1.png"
            dest_path = os.path.join(IMAGES_DIR, out_fname)

            if not os.path.exists(dest_path):
                img_data = zf.read(internal_fname)
                img = Image.open(BytesIO(img_data))
                if img.mode != "L":
                    img = img.convert("L")
                resized = img.resize(target_size, Image.Resampling.BILINEAR)
                resized.save(dest_path, format="PNG", optimize=True)

            records.append({
                "filename": out_fname,
                "label": 1,
                "label_name": "Tuberculosis",
                "source": source
            })

    return records


def setup_dataset(max_workers=8):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    all_items = []

    for cfg in DATASET_CONFIGS:
        print(f"Fetching image catalogue for {cfg['source']}...")
        filenames = get_image_list(cfg["base_url"], cfg["pattern"])
        print(f"  Discovered {len(filenames)} chest radiographs.")
        for f in filenames:
            all_items.append({
                "filename": f,
                "base_url": cfg["base_url"],
                "source": cfg["source"]
            })

    print(f"\nProcessing NLM radiographs (Montgomery + Shenzhen): {len(all_items)}")

    records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_and_process_image, item): item for item in all_items}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing CXRs"):
            result = future.result()
            if result.get("status") != "failed":
                records.append({
                    "filename": result["filename"],
                    "label": result["label"],
                    "label_name": "Tuberculosis" if result["label"] == 1 else "Normal",
                    "source": result["source"]
                })
            else:
                print(f"Warning: Failed to fetch {result.get('filename')}: {result.get('error')}")

    # Process Option 3: NIAID TB Portals (Belarus Set)
    niaid_records = download_and_process_niaid_belarus()
    records.extend(niaid_records)

    df = pd.DataFrame(records).sort_values("filename").reset_index(drop=True)
    df.to_csv(METADATA_PATH, index=False)
    print(f"\nDataset saved successfully!")
    print(f"Metadata location: {METADATA_PATH}")
    print(f"Total images: {len(df)}")
    print(f"Distribution by Source:\n{df['source'].value_counts()}\n")
    print(f"Class distribution:\n{df['label_name'].value_counts()}\n")
    return df


if __name__ == "__main__":
    setup_dataset()
