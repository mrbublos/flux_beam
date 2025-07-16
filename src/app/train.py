import os
import subprocess

from src.app.create_config import create_config, TrainConfig
from src.app.inference import flush
from src.app.logger import Logger

logger = Logger(__name__)

def train_user_lora(event: TrainConfig):
    user_id = event.user_id

    try:
        flush()

        os.makedirs(os.path.dirname(event.raw_images_dir + "/"), exist_ok=True)
        os.makedirs(os.path.dirname(event.processed_images_dir + "/"), exist_ok=True)

        config_name = create_config(event)
        script_path = event.script_path

        logger.debug(f"Starting training for {user_id}")

        logger.info(f"Learning model for {user_id} {config_name}")
        result = subprocess.run([script_path, config_name], capture_output=False, text=True)
        logger.debug(f"Completed training for {user_id}")
        # Check the script's exit code
        if result.returncode == 0:
            return {
                'user_id': user_id,
                'success': True,
            }
        else:
            raise Exception(result.stderr)


    except Exception as e:
        logger.error(f"Error training model file for {user_id}: {e}")
        raise Exception(f"Error training model file for {user_id}: {str(e)}")
