import os
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

FILTERED_DIR = "detection/data/02_filtered"
RAW_DIR = "detection/data/03_raw"
MAX_WORKERS = 16
TIMEOUT = 10
MAX_RETRIES = 2


def load_selection(split: str) -> list[dict]:
    with open(f"{FILTERED_DIR}/{split}_selection.json") as f:
        return json.load(f)


def download_image(session: requests.Session, img_info: dict, output_dir: str) -> str | None:
    dest_path = os.path.join(output_dir, img_info["file_name"])

    if os.path.exists(dest_path):
        return None

    for attempt in range(MAX_RETRIES + 1):
        try:
            r = session.get(img_info["coco_url"], timeout=TIMEOUT)
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(r.content)
            return None
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                return img_info["file_name"]
            time.sleep(1)


def download_split(split: str, max_workers: int = MAX_WORKERS) -> list[str]:
    output_dir = os.path.join(RAW_DIR, split)
    os.makedirs(output_dir, exist_ok=True)

    images_info = load_selection(split)
    failures = []

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_image, session, img_info, output_dir): img_info
                for img_info in images_info
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=split):
                result = future.result()
                if result:
                    failures.append(result)

    return failures


if __name__ == "__main__":
    for split in ["train", "val"]:
        failures = download_split(split)
        print(f"{split}: {len(failures)} échecs")
        if failures:
            with open(f"{RAW_DIR}/{split}_failed.txt", "w") as f:
                f.write("\n".join(failures))
