import requests
from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm

ZENODO_RECORD_ID = "10245813"
API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
DATA_DIR = Path(__file__).parent.parent / "data"


def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(
        total=total_size_in_bytes, unit="iB", unit_scale=True, desc=dest_path.name
    )

    with open(dest_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching metadata for Zenodo record {ZENODO_RECORD_ID}...")
    response = requests.get(API_URL)
    if response.status_code != 200:
        print(f"Failed to fetch metadata: {response.status_code}")
        return

    data = response.json()
    files = data.get("files", [])

    print(f"Found {len(files)} files.")

    for file_info in files:
        file_url = file_info["links"]["self"]
        file_name = file_info["key"]
        dest_path = DATA_DIR / file_name

        if dest_path.exists():
            print(f"File {file_name} already exists. Skipping download.")
        else:
            print(f"Downloading {file_name}...")
            download_file(file_url, dest_path)

        if file_name.endswith(".zip"):
            print(f"Extracting {file_name}...")
            with ZipFile(dest_path, "r") as zip_ref:
                zip_ref.extractall(DATA_DIR)
            print(f"Extracted {file_name}.")


if __name__ == "__main__":
    main()
