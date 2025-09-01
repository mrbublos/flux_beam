import base64
import os
import shutil
import uuid
from io import BytesIO

import modal
from PIL import Image

from src.app.logger import Logger

# Define persistent storage volumes
volume_raw = modal.Volume.from_name("raw-data", create_if_missing=True)
volume_processed = modal.Volume.from_name("processed-data", create_if_missing=True)

def validate_image(image_bytes: bytes) -> bool:
    """
    Validate if the given bytes represent a valid JPG or PNG image.

    Args:
        image_bytes: Bytes containing the image data

    Returns:
        bool: True if the image is a valid JPG or PNG, False otherwise
    """

    # First, try to open the image with PIL
    img = Image.open(BytesIO(image_bytes))

    # Check if the format is either JPEG or PNG
    if img.format not in ('JPEG', 'PNG'):
        return False

    # Try to load the image data to verify it's not corrupted
    img.verify()

    # For some images, we need to open it again after verify
    img = Image.open(BytesIO(image_bytes))
    img.load()

    return True


def _store_file(user_id: str, base64_data: str, extension: str):
    """Stores a file decoded from a base64 string."""
    volume_raw.reload()

    if not all([user_id, base64_data, extension]):
        raise ValueError("Missing required parameters: user_id, base64_data, or extension")

    user_folder_path = f"/mnt/code/raw_data/{user_id}"
    os.makedirs(user_folder_path, exist_ok=True)

    try:
        image_data = base64.b64decode(base64_data)
    except Exception as e:
        raise Exception(f"Failed to decode base64 data: {e}")

    try:
        validate_image(image_data)
        logger.info(f"Image validated {user_id} {extension}")
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

    file_name = f"{uuid.uuid4()}.{extension}"
    file_path = os.path.join(user_folder_path, file_name)

    try:
        with open(file_path, "wb") as f:
            f.write(image_data)
        volume_raw.commit()  # Persist changes
    except Exception as e:
        raise Exception(f"Failed to store file: {e}")

    return {"status": "success", "file_path": file_path}


def _clear_files(user_id: str):
    """Clears all files for a given user from both raw and processed volumes."""
    volume_raw.reload()
    volume_processed.reload()

    paths_to_clear = [
        f"/mnt/code/raw_data/{user_id}",
        f"/mnt/code/processed/{user_id}",
    ]
    cleared = False
    for path in paths_to_clear:
        if os.path.exists(path):
            shutil.rmtree(path)
            cleared = True

    if cleared:
        volume_raw.commit()
        volume_processed.commit()

    return {"status": "success"}


app = modal.App("FileManipulator")
image = (
    modal.Image
    .debian_slim(python_version="3.11")
    .pip_install(["Pillow==11.3.0"])
    .add_local_python_source("src")
)

logger = Logger(__name__)


@app.cls(
    image=image,
    volumes={
        "/mnt/code/raw_data": volume_raw,
        "/mnt/code/processed": volume_processed,
    },
    max_containers=10,
    scaledown_window=10,
)
class FileManipulator:

    @modal.method()
    def run(self, data: dict):
        user_id = data.get("user_id")
        image_data = data.get("image_data") if "image_data" in data else None
        extension = data.get("extension") if "extension" in data else None
        action = data.get("action") if "action" in data else "store"

        logger.info(f"Starting file manipulator...{user_id} {action} {extension}")

        """Main method to dispatch actions."""
        if action == "store":
            return _store_file(user_id, image_data, extension)
        elif action == "clear":
            return _clear_files(user_id)
        else:
            raise ValueError(f"Unknown action: {action}")


@app.local_entrypoint()
def main(data: dict):
    process_job = modal.Cls.from_name("FileManipulator", "FileManipulator")
    call = process_job.run.spawn(data)
    return call.object_id


if __name__ == "__main__":
    # main({
    #     "action": "store",
    #     "user_id": "test_arina",
    #     "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg==",
    #     "extension": "png"
    # })
    # main({
    #     "action": "clear",
    #     "user_id": "test_arina",
    # })
    directory = "/Users/ip/IdeaProjects/runpod-flux-serverless/tmp/raw_images"
    user_id = "test_arina"
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        raise Exception(f"Directory '{directory}' does not exist")

        # Get list of all files in the directory
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    except Exception as e:
        print(f"Error reading directory: {e}")
        raise Exception(f"Error reading directory: {e}")

    # files = ['corrupted.png']
    files = ['IMG_3268.jpeg']

    if not files:
        print(f"No files found in {directory}")
        raise Exception(f"No files found in {directory}")

    print(f"Found {len(files)} files to process...")

    # Process each file
    for filename in files:
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            file_data = f.read()

        # Get file extension
        _, extension = os.path.splitext(filename)
        extension = extension.lstrip('.').lower()

        # Skip files without extensions or with unsupported formats
        if not extension or extension not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            print(f"Skipping {filename}: Unsupported file format")
            continue

        # Convert to base64
        base64_data = base64.b64encode(file_data).decode('utf-8')

        # Prepare data for main function
        data = {
            "action": "store",
            "user_id": user_id,
            "image_data": base64_data,
            "extension": extension
        }

        # Call main function
        print(f"Processing {filename}...")
        result = main(data)
        print(f"Successfully processed {filename}. Result: {result}")
