import hashlib
import os

import requests

from src.app.logger import Logger
from huggingface_hub import snapshot_download

logger = Logger(__name__)

STYLES_FOLDER = os.getenv("STYLES_FOLDER", "")
if STYLES_FOLDER is None:
    raise Exception("STYLES_FOLDER is not set")

CIVIT_AI_TOKEN = os.getenv("CIVIT_AI_TOKEN", "")

def download_file(url):
    logger.info(f"Downloading file: {url}")
    try:
        os.makedirs(STYLES_FOLDER, exist_ok=True)

        logger.info(f"Downloading file: {url}")

        if "civitai.com" in url:
            return download_civitai(url)
        elif "huggingface.co" in url:
            return download_huggingface(url)
        else:
            raise Exception(f"Unknown url: {url}")

    except Exception as e:
        logger.error(f"Error downloading file: {url} {e}")
        raise e

def download_huggingface(url):
    url = url.replace("https://huggingface.co/", "")
    m = hashlib.md5()
    m.update(url.encode('utf-8'))
    url_hash = m.hexdigest()

    local_dir = f"{STYLES_FOLDER}/{url_hash}"
    snapshot_download(
        repo_id=url,
        local_dir=local_dir,
        allow_patterns=["*.safetensors"],
    )

    import shutil
    safetensors_files = [file for file in os.listdir(local_dir) if file.endswith('.safetensors')]
    if len(safetensors_files) > 1:
        raise Exception(f"More than one safetensors file found in {local_dir}")
    elif len(safetensors_files) == 0:
        raise Exception(f"No safetensors file found in {local_dir}")

    src_path = os.path.join(local_dir, safetensors_files[0])
    dest_path = os.path.join(STYLES_FOLDER, f"{url_hash}.safetensors")
    shutil.move(src_path, dest_path)
    shutil.rmtree(local_dir)

    url_file_path = os.path.join(STYLES_FOLDER, f"{url_hash}.txt")
    with open(url_file_path, 'w') as f:
        f.write(url)

    return f"{url_hash}.safetensors", url_file_path

def download_civitai(url):
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

    response = requests.get(url + '&token=' + CIVIT_AI_TOKEN, stream=True)
    response.raise_for_status()

    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with open(url_file_path, 'w') as f:
        f.write(url)

    logger.info(f"File {url} downloaded: {file_path}")

    return file_name, url_file_name