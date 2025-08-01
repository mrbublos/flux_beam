import torch
from beam import function, Volume, Image

from src.app.create_config import TrainConfig
from src.app.train import train_user_lora

VOLUME_PATH = "/mnt/code/models"
RAW_VOLUME_PATH = "/mnt/code/raw_data"
PROCESSED_VOLUME_PATH = "/mnt/code/processed"
LORAS_VOLUME_PATH = "/mnt/code/loras"


@function(
    image=Image(python_version="pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime")
    .add_commands([
        "cd /workspace && git clone https://github.com/mrbublos/character_training.git",
        "cd /workspace/character_training && git submodule update --init --recursive",
        "cd /workspace/character_training && git checkout runpod_serverless",
        "chmod +x /workspace/character_training/start_training.sh",
        "echo 1",
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
            "setuptools<70",
            "tensorboard",
            "timm",
            "toml",
            # "torch",
            # "torchaudio",
            # "torchvision",
            "tqdm",
            "transformers==4.49.0",
        ]
    ),
    gpu="H100",
    memory="32Gi",
    cpu=2,
    secrets=["HF_TOKEN"],
    env={
        "PREPROCESSING_MODEL": "/mnt/code/models/florence_2_large",
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

    # files = os.listdir("/workspace/character_training")
    # print(files)

    user_id="test_arina"
    train_user_lora(TrainConfig(
        user_id=user_id,
        steps=500,
        processed_images_dir=f"{PROCESSED_VOLUME_PATH}/{user_id}",
        default_config="/mnt/code/configs/dev.yaml",
        lora_output_dir=LORAS_VOLUME_PATH,
        raw_images_dir=f"{RAW_VOLUME_PATH}/{user_id}",
    ))
    print("Lora train completed.")


if __name__ == "__main__":
    run()
