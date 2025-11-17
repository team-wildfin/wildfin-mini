import requests
import yaml
import os
from config.data.datasets import DATASETS


def get_dataset_files(dataverse_url, doi, headers=None):
    url = f"{dataverse_url}/api/datasets/:persistentId/?persistentId={doi}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    files = data['data']['latestVersion']['files']
    return [
        {
            'file_id': f['dataFile']['id'],
            'filename': f['label']
        }
        for f in files
    ]

def download_file_by_id(dataverse_url, file_id, save_path, headers=None):
    url = f"{dataverse_url}/api/access/datafile/{file_id}"
    with requests.get(url, headers=headers, stream=True) as response:
        print("Request Body:", response.request.url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):  # 8 KB chunks

                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
            print(f"File saved to {save_path}")
        else:
            print(f"Failed to download file: {response.status_code} {response.text}")

if __name__ == "__main__":
    DATASET_NAMES = ['fishfollow', 'coralcam'] 
    dataverse = yaml.safe_load(open('config/actual/dataverse.yml', 'r'))
    DATAVERSE_URL = dataverse['url']
    API_KEY = dataverse['api_key']
    HEADERS = {"X-Dataverse-key": API_KEY}

    for DATASET_NAME in DATASET_NAMES:
        print(f"Processing dataset: {DATASET_NAME}")
        ROOT = DATASETS[DATASET_NAME].path
        RAW_DIR = os.path.join(ROOT, "raw")
        os.makedirs(RAW_DIR, exist_ok=True)
        files = get_dataset_files(DATAVERSE_URL, DATASETS[DATASET_NAME].doi, HEADERS)
        print(f"Found {len(files)} files in dataset.")
        for f in files:
            dest_path = os.path.join(RAW_DIR, f['filename'])
            print(f"Downloading {f['filename']}...")
            download_file_by_id(DATAVERSE_URL, f['file_id'], dest_path, HEADERS)
    print("✅ All downloads complete.")