import os
import requests
import zipfile
from pathlib import Path

# The Cancer Imaging Archive (TCIA) REST API URLs
TCIA_BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"

FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "tcia"

def get_series_list(collection, modality="CT"):
    """Get a list of series for a specific collection and modality."""
    url = f"{TCIA_BASE_URL}/getSeries?Collection={collection}&Modality={modality}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def download_series(series_uid, output_dir):
    """Download a specific series as a ZIP file and extract it."""
    url = f"{TCIA_BASE_URL}/getImage?SeriesInstanceUID={series_uid}"
    
    zip_path = output_dir / f"{series_uid}.zip"
    
    print(f"Downloading Series UID {series_uid} to {zip_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir / "images")
        
    # Clean up the zip file
    os.remove(zip_path)

def main():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    
    # RIDER Lung CT is a good dataset for radiomics benchmarking
    collection = "RIDER Lung CT"
    
    print(f"Fetching series list for {collection}...")
    series_list = get_series_list(collection)
    
    if not series_list:
        print("No series found.")
        return
        
    # Pick the first series
    target_series = series_list[0]
    series_uid = target_series["SeriesInstanceUID"]
    
    print(f"Selected Series UID: {series_uid}")
    print(f"Image Count: {target_series.get('ImageCount', 'Unknown')}")
    
    # Download the series
    download_series(series_uid, FIXTURE_DIR)
    
    # Create a dummy mask for benchmarking purposes.
    # In a real scenario, we'd download the SEG/RTSTRUCT, but for benchmarking speed,
    # a generated spherical mask over the center of the DICOM volume is sufficient and requires fewer dependencies to parse.
    print(f"Download complete. DICOM files extracted to {FIXTURE_DIR / 'images'}")
    print("Note: run_benchmark.py will generate a synthetic standard mask dynamically for this clinical volume.")

if __name__ == "__main__":
    main()
