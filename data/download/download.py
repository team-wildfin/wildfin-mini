import requests
import yaml
import os


DATASETS = ['fishfollow', 'coralcam'] 
config = yaml.safe_load(open('config/actual/dataset.yml', 'r'))
dataverse = yaml.safe_load(open('config/actual/dataverse.yml', 'r'))
DATAVERSE_URL = dataverse['url']
API_KEY = dataverse['api_key']
HEADERS = {"X-Dataverse-key": API_KEY}

def get_dataset_files(doi):
    url = f"{DATAVERSE_URL}/api/datasets/:persistentId/?persistentId={doi}"
    response = requests.get(url, headers=HEADERS)
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

def download_file_by_id(file_id, save_path):
    url = f"{DATAVERSE_URL}/api/access/datafile/{file_id}"
    with requests.get(url, headers=HEADERS, stream=True) as response:
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
    for DATASET in DATASETS:
        print(f"Processing dataset: {DATASET}")
        ROOT = config[DATASET]['path']
        RAW_DIR = os.path.join(ROOT, "raw")
        os.makedirs(RAW_DIR, exist_ok=True)
        files = get_dataset_files(config[DATASET]['doi'])
        print(f"Found {len(files)} files in dataset.")
        for f in files:
            dest_path = os.path.join(RAW_DIR, f['filename'])
            print(f"Downloading {f['filename']}...")
            download_file_by_id(f['file_id'], dest_path)
    print("✅ All downloads complete.")