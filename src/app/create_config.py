from src.app.logger import Logger
import yaml

logger = Logger("create_config")

class TrainConfig:
    def __init__(
            self,
            user_id,
            steps,
            processed_images_dir,
            default_config,
            lora_output_dir,
            raw_images_dir,
            script_path = "/workspace/character_training/start_training_beam.sh",
    ):
        self.user_id = user_id
        self.steps = steps
        self.processed_images_dir = processed_images_dir
        self.default_config = default_config
        self.lora_output_dir = lora_output_dir
        self.raw_images_dir = raw_images_dir
        self.script_path = script_path


def create_config(train_config: TrainConfig):
    logger.info(f"Updating config for {train_config.user_id}")
    with open(train_config.default_config, 'r') as file:
        config = yaml.safe_load(file)

    old_input_folder = config['config']['process'][0]['datasets'][0]['folder_path']
    input_folder = train_config.processed_images_dir
    config['config']['process'][0]['datasets'][0]['folder_path'] = input_folder
    logger.debug(f"Updated config input folder to {input_folder} from {old_input_folder}")

    old_output_folder = config['config']['name']
    output_folder =train_config.user_id
    config['config']['name'] = output_folder
    logger.debug(f"Updated config output folder to {output_folder} from {old_output_folder}")

    old_steps = config['config']['process'][0]['train']['steps']
    new_steps = train_config.steps or old_steps
    config['config']['process'][0]['train']['steps'] = new_steps
    logger.debug(f"Updated config steps from {old_steps} to {new_steps}")

    old_training_folder = config['config']['process'][0]['training_folder']
    new_training_folder = train_config.lora_output_dir
    config['config']['process'][0]['training_folder'] = new_training_folder
    logger.debug(f"Updated config training_folder from {old_training_folder} to {new_training_folder}")

    config_name = f"{train_config.raw_images_dir}/config.yaml"
    with open(config_name, 'w') as file:
        yaml.dump(config, file)

    return config_name
