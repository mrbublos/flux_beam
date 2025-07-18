import os

import runpod

from src.app.create_config import TrainConfig
from src.app.inference import get_generator
from src.app.logger import Logger
from src.app.preprocess_images import preprocess_images
from src.app.s3client import S3Client
from src.app.train import train_user_lora

import shutil

logger = Logger(__name__)

get_generator()

print("Starting inference handler...")
USER_DATA_PATH = os.getenv("USER_DATA_PATH", "/runpod-volume/user_data")
USER_PROCESSED_DATA_PATH = os.getenv("USER_PROCESSED_DATA_PATH", "/runpod-volume/processed")
MODEL = os.getenv("MODEL", "/runpod-volume/models/flux_dev")
LORAS_PATH = os.getenv("LORAS_PATH", "/runpod-volume/loras")

s3_client = S3Client()

def train(event):
    inputs = event["input"]
    user_id = inputs["user_id"]
    steps = inputs["steps"]

    raw_user_data_path = f"{USER_DATA_PATH}/{user_id}"
    processed_user_data_path = f"{USER_PROCESSED_DATA_PATH}/{user_id}"

    if os.path.exists(f"{processed_user_data_path}/{user_id}.safetensors"):
        logger.info(f"User {user_id} already trained. Skipping training.")
        return {"status": "success"}

    # clean up preprocessed files
    if os.path.exists(processed_user_data_path):
        shutil.rmtree(processed_user_data_path)
    os.makedirs(processed_user_data_path, exist_ok=True)

    print("Starting image preprocessing...")
    preprocess_images(input_dir=raw_user_data_path, output_dir=processed_user_data_path)

    print("Image preprocessing completed.")

    train_user_lora(TrainConfig(
        user_id=user_id,
        steps=steps,
        processed_images_dir=processed_user_data_path,
        default_config="./configs/dev.yaml",
        lora_output_dir=LORAS_PATH,
        raw_images_dir=raw_user_data_path,
        script_path="/app/character_training/start_training_beam.sh",
        model_path=MODEL,
    ))

    print(f"Lora train for user {user_id} completed.")

    return {"status": "success"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": train})