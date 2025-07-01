from beam import Image, Volume, task_queue

from src.app.create_config import TrainConfig
from src.app.preprocess_images import preprocess_images
from src.app.train import train_user_lora


@task_queue(
    # cpu=2,
    memory="32Gi",
    gpu="H100",
    image=Image(python_version="python3.11")
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
            "torch",
            "torchaudio",
            "torchvision",
            "tqdm",
            "transformers==4.49.0",
        ]
    ),
    env={
        "PREPROCESSING_MODEL": "./models/florence_2_large",
        "HF_OFFLINE": "1",
    },
    volumes=[
        Volume(name="models", mount_path="/mnt/code/models"),
        Volume(name="raw_data", mount_path="/mnt/code/raw_data"),
        Volume(name="processed", mount_path="/mnt/code/processed"),
        Volume(name="loras", mount_path="/mnt/code/loras"),
    ],
)
def train(user_id: str, steps: int = 500):
    print(f"Starting lora train for user {user_id}...")

    print("Starting image preprocessing...")
    preprocess_images(input_dir=f"./raw_data/{user_id}", output_dir=f"./processed/{user_id}")
    print("Image preprocessing completed.")

    train_user_lora(TrainConfig(
        user_id=user_id,
        steps=steps,
        processed_images_dir=f"/mnt/code/processed/{user_id}",
        default_config="/mnt/code/configs/schnell.yaml",
        lora_output_dir="/mnt/code/loras",
        raw_images_dir=f"/mnt/code/raw_data/{user_id}",
    ))

    print(f"Lora train for user {user_id} completed.")

    return {"status": "success"}
