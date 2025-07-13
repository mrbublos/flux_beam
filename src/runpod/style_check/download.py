import hashlib
import os
from logging import Logger

import requests

logger = Logger(__name__)

STYLES_FOLDER = os.getenv("STYLES_FOLDER")
if STYLES_FOLDER is None:
    raise Exception("STYLES_FOLDER is not set")

def download_file(url):
    os.makedirs(STYLES_FOLDER, exist_ok=True)

    m = hashlib.md5()
    m.update(url.encode('utf-8'))
    url_hash = m.hexdigest()

    file_name = f"{url_hash}.safetensors"
    url_file_name = f"{url_hash}.txt"
    url_file_path = os.path.join(STYLES_FOLDER, f"{url_hash}.txt")
    file_path = os.path.join(STYLES_FOLDER, file_name)

    if os.path.exists(file_path):
        logger.info(f"File already exists: {file_path}")
        return file_name, url_file_name

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with open(url_file_path, 'w') as f:
        f.write(url)

    logger.info(f"File {url} downloaded: {file_path}")

    return file_name, url_file_name
