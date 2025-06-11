import os
import subprocess

from src.create_config import create_config, TrainConfig
from src.logger import Logger

logger = Logger(__name__)

def train_user_lora(event: TrainConfig):
    user_id = event.user_id

    try:
        os.makedirs(os.path.dirname(event.raw_images_dir + "/"), exist_ok=True)
        os.makedirs(os.path.dirname(event.processed_images_dir + "/"), exist_ok=True)

        config_name = create_config(event)
        script_path = f"/workspace/character_training/start_training_beam.sh"

        logger.debug(f"Starting training for {user_id}")

        # logger.info(f"Preprocessing images {user_id}")
        # start(input_dir=user_photos_raw,
        #       output_dir=user_photos_processed)

        logger.info(f"Learning model for {user_id}")
        result = subprocess.run([script_path, config_name], capture_output=False, text=True)
        logger.debug(f"Completed training for {user_id}")
        # Check the script's exit code
        if result.returncode == 0:
            return {
                'user_id': user_id,
                'success': True,
            }
        else:
            return {
                'user_id': user_id,
                'success': False,
            }


    except Exception as e:
        logger.error(f"Error training model file for {user_id}: {e}")
        return {
            "success": False,
        }
