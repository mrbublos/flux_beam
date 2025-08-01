import os

import modal

from src.app.create_config import TrainConfig
from src.app.inference import cuda_info
from src.app.logger import Logger
from src.app.preprocess_images import setup_preprocessing_model, preprocess_images
from src.app.train import train_user_lora

# Define the Modal App
app = modal.App("Train")

# Define Network File Systems for persistent storage
volume_models = modal.Volume.from_name("models", create_if_missing=True)
volume_raw = modal.Volume.from_name("raw-data", create_if_missing=True)
volume_processed = modal.Volume.from_name("processed-data", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .pip_install_from_requirements("src/modal/requirements.txt")
    .apt_install("git")
    .run_commands(
        "cd / && git clone https://github.com/mrbublos/character_training.git",
        "cd /character_training && git submodule update --init --recursive",
        "cd /character_training && git checkout runpod_serverless",
        "chmod +x /character_training/start_training_beam.sh",
    )
    .apt_install([
        "libgl1",
        "libglib2.0-0",
        "pciutils",
    ])
    .pip_install(["optimum-quanto==0.2.4"])
    .pip_install(["fastapi[standard]"])
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "$CUDA_HOME/bin:$PATH",
        "PREPROCESSING_MODEL": "/mnt/models/florence_2_large",
        "HF_OFFLINE": "1",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        "TORCH_CUDA_ARCH_LIST": "Turing",
    })
    .add_local_python_source("src")
    .add_local_dir("configs", "/mnt/code/configs")
)
volume_loras = modal.Volume.from_name("loras", create_if_missing=True)


# Define the container image

@app.cls(
    image=image,
    gpu="H100",
    cpu=2,
    memory=4096,
    volumes={
        "/mnt/models": volume_models,
        "/mnt/raw_data": volume_raw,
        "/mnt/processed": volume_processed,
        "/mnt/loras": volume_loras,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    max_containers=8,
    timeout=1500, # 25 minutes
)
class Train:
    @modal.enter()
    def setup(self):
        self.logger = Logger(__name__)
        setup_preprocessing_model()

    @modal.method()
    def run(self, inputs: dict):
        self.logger.info("Starting lora train...")

        user_id = inputs["user_id"]
        steps = inputs["steps"]

        if os.path.exists(f"/mnt/loras/{user_id}/{user_id}.safetensors"):
            self.logger.info(f"Lora for user {user_id} already exists, skipping training.")
            return {"status": "success"}

        volume_raw.reload()
        volume_processed.reload()

        cuda_info()

        print(f"Starting lora train for user {user_id}...")

        user_processed_images_dir = f"/mnt/processed/{user_id}"
        user_raw_images_dir = f"/mnt/raw_data/{user_id}"

        print("Starting image preprocessing...")
        preprocess_images(
            input_dir=user_raw_images_dir,
            output_dir=user_processed_images_dir,
            setup_model=False
        )
        print("Image preprocessing completed.")

        train_user_lora(TrainConfig(
            user_id=user_id,
            steps=steps,
            processed_images_dir=user_processed_images_dir,
            default_config="/mnt/code/configs/dev.yaml",
            lora_output_dir="/mnt/loras",
            raw_images_dir=user_raw_images_dir,
            script_path="/character_training/start_training_beam.sh",
            model_path="/mnt/models/flux_dev",
            run_path="/character_training/run.py",
        ))

        if os.path.exists(user_processed_images_dir):
            os.rmdir(user_processed_images_dir)

        # if os.path.exists(user_raw_images_dir):
        #     os.rmdir(user_raw_images_dir)

        volume_processed.commit()
        volume_loras.commit()
        print(f"Lora train for user {user_id} completed.")

        return {"status": "success"}


@app.local_entrypoint()
def local_train():
    Train().train.remote({ "user_id": "test_arina", "steps": 10 })
