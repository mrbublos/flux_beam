import os

from beam import function, Volume, Image, env

from src.create_config import TrainConfig
from src.train import train_user_lora

VOLUME_PATH = "./models"
RAW_VOLUME_PATH = "./raw_data"
PROCESSED_VOLUME_PATH = "./processed"
LORAS_VOLUME_PATH = "./loras"


@function(
    image=Image(python_version="python3.11")
    .add_commands([
        "git clone https://github.com/mrbublos/character_training.git",
        "cd character_training && git submodule update --init --recursive",
        "chmod +x ./character_training/start_training.sh",
    ]).add_python_packages(
        [
            "accelerate",
            "albucore==0.0.16",
            "albumentations==1.4.15",
            "bitsandbytes",
            "controlnet_aux==0.0.7",
            "diffusers",
            "einops",
            "flatten_json",
            "gradio",
            "hf_transfer",
            "huggingface_hub",
            "invisible-watermark",
            "k-diffusion",
            "kornia",
            "lpips",
            "lycoris-lora==1.8.3",
            "omegaconf",
            "open_clip_torch",
            "optimum-quanto==0.2.4",
            "oyaml",
            "peft",
            "Pillow",
            "pillow-heif",
            "prodigyopt",
            "pydantic",
            "python-dotenv",
            "python-slugify",
            "pytorch_fid",
            "pyyaml",
            "safetensors",
            "sentencepiece",
            "tensorboard",
            "timm",
            "toml",
            "torch",
            "torchaudio",
            "torchvision",
            "tqdm",
            "transformers==4.49.0",
            "yaml",
        ]
    ),
    gpu="T4",
    secrets=["HF_TOKEN"],
    env={
        "PREPROCESSING_MODEL": "./models/florence_2_large",
        "TRAINING_MODEL": "./models/flux_schnell",
        "HF_OFFLINE": "1",
    },
    volumes=[
        Volume(name="models", mount_path=VOLUME_PATH),
        Volume(name="raw_data", mount_path=RAW_VOLUME_PATH),
        Volume(name="processed", mount_path=PROCESSED_VOLUME_PATH),
        Volume(name="loras", mount_path=LORAS_VOLUME_PATH),
    ],
)
def run():
    print("Starting lora train...")
    user_id="test_arina"
    train_user_lora(TrainConfig(
        user_id=user_id,
        steps=2,
        processed_images_dir=f"{PROCESSED_VOLUME_PATH}/{user_id}",
        default_config="./character_training/configs/train_config_1h100.yaml",
        model_name=os.getenv("TRAINING_MODEL"),
        lora_output_dir=f"./{LORAS_VOLUME_PATH}/{user_id}",
    ))
    print("Lora train completed.")


if __name__ == "__main__":
    run()
