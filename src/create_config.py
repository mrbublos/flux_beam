from src.logger import Logger
import yaml

logger = Logger("create_config")


class TrainConfig:
    def __init__(
            self,
            user_id,
            steps,
            processed_images_dir,
            default_config,
            model_name,
            lora_output_dir,
            raw_images_dir,
    ):
        self.user_id = user_id
        self.steps = steps
        self.model_name = model_name
        self.processed_images_dir = processed_images_dir
        self.lora_output_dir = lora_output_dir
        self.raw_images_dir = raw_images_dir
        self.default_config = default_config


def create_config(config: TrainConfig):
    logger.info(f"Updating config for {config.user_id}")
    with open(config.default_config, 'r') as file:
        config = yaml.safe_load(file)

    old_input_folder = config['config']['process'][0]['datasets'][0]['folder_path']
    input_folder = f"{config.processed_images_dir}/{config.user_id}"
    config['config']['process'][0]['datasets'][0]['folder_path'] = input_folder
    logger.debug(f"Updated config input folder to {input_folder} from {old_input_folder}")

    old_output_folder = config['config']['name']
    output_folder =config.user_id
    config['config']['name'] = output_folder
    logger.debug(f"Updated config output folder to {output_folder} from {old_output_folder}")

    old_steps = config['config']['process'][0]['train']['steps']
    new_steps = config.steps or old_steps
    config['config']['process'][0]['train']['steps'] = new_steps
    logger.debug(f"Updated config steps from {old_steps} to {new_steps}")

    old_training_folder = config['config']['process'][0]['training_folder']
    new_training_folder = config.lora_output_dir
    config['config']['process'][0]['training_folder'] = new_training_folder
    logger.debug(f"Updated config training_folder from {old_training_folder} to {new_training_folder}")

    old_model = config['config']['process'][0]['train']['model']['name_or_path']
    new_model = config.model_name
    config['config']['process'][0]['train']['model']['name_or_path'] = new_model
    logger.debug(f"Updated config model from {old_model} to {new_model}")

    config_name = f"{config.raw_images_dir}/{config.user_id}/config.yaml"
    with open(config_name, 'w') as file:
        yaml.dump(config, file)

    return config_name
