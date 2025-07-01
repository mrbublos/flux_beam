import base64
import os
import uuid

from beam import task_queue, Image, Volume

def store_file(user_id: str, base64_data: str, extension: str):
    if not user_id or not base64_data or not extension:
        print("Missing required parameters user_id, base64_data, and extension")
        raise Exception("Missing required parameters")

    # Create a directory for the user if it doesn't exist
    # The path is relative to the volume mount path
    user_folder_path = f"./raw_data/{user_id}"
    os.makedirs(user_folder_path, exist_ok=True)

    # Decode the base64 string
    try:
        image_data = base64.b64decode(base64_data)
    except Exception as e:
        print(f"Failed to decode base64 data: {e}")
        raise Exception(f"Failed to decode base64 data: {str(e)}")

    # Generate a unique filename
    file_name = f"{uuid.uuid4()}.{extension}"
    file_path = os.path.join(user_folder_path, file_name)
    print(f"Storing file {file_path}")
    # Write the file to the persistent volume
    try:
        with open(file_path, "wb") as f:
            f.write(image_data)
    except Exception as e:
        print(f"Failed to store file: {e}")
        raise Exception(f"Failed to store file: {str(e)}")

    # Return the path to the stored file
    return {"status": "success", "file_path": file_path}

def clear_files(user_id: str):
    # The path is relative to the volume mount path
    try:
        user_folder_path = f"./raw_data/{user_id}"
        if os.path.exists(user_folder_path):
            import shutil
            shutil.rmtree(user_folder_path)
    except Exception as e:
        print(f"Failed to clear files for user {user_id}: {e}")
        raise Exception(f"Failed to clear files for user {user_id}: {str(e)}")

    try:
        processed_folder_path = f"./processed/{user_id}"
        if os.path.exists(processed_folder_path):
            import shutil
            shutil.rmtree(processed_folder_path)
    except Exception as e:
        print(f"Failed to clear files for user {user_id}: {e}")
        raise Exception(f"Failed to clear files for user {user_id}: {str(e)}")

    return {"status": "success"}

@task_queue(
    name="file_manipulator",
    cpu=0.125,
    memory="1Gi",
    image=Image(
        python_version="python3.9",
        python_packages=[],
    ),
    volumes=[
        Volume(name="raw_data", mount_path="/mnt/code/raw_data"),
        Volume(name="processed", mount_path=f"/mnt/code/processed"),
    ],
)
def file_manipulator(**inputs):
    """
    This endpoint receives a userId, a base64 encoded byte array, and an extension.
    It stores the file in a volume under a folder named userId.
    """
    if not "action" in inputs:
        return {"status": "error", "message": "action is required."}

    action = inputs["action"]
    user_id = inputs["user_id"] if "user_id" in inputs else None
    image_data = inputs["image_data"] if "image_data" in inputs else None
    extension = inputs["extension"] if "extension" in inputs else None

    if action == "store":
        return store_file(user_id, image_data, extension)
    elif action == "clear":
        return clear_files(user_id)
    else:
        return {"status": "error", "message": "Invalid action."}