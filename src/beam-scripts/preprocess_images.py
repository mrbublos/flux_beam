from beam import function, Volume, Image

from src.app.preprocess_images import preprocess_images

VOLUME_PATH = "./models"
RAW_VOLUME_PATH = "./raw_data"
PROCESSED_VOLUME_PATH = "./processed"


@function(
    image=Image(python_version="python3.11")
    .add_python_packages(
        [
            "diffusers",
            "transformers==4.49.0",
            "accelerate",
            "torch",
            "Pillow",
            "tqdm",
            "safetensors",
            "pillow_heif",
            "einops",
            "timm",
            "oyaml",
        ]
    ),
    gpu="T4",
    secrets=["HF_TOKEN"],
    env={
        "PREPROCESSING_MODEL": "./models/florence_2_large",
        "HF_OFFLINE": "1",
    },
    volumes=[
        Volume(name="models", mount_path=VOLUME_PATH),
        Volume(name="raw_data", mount_path=RAW_VOLUME_PATH),
        Volume(name="processed", mount_path=PROCESSED_VOLUME_PATH),
    ],
)
def run():
    print("Starting image preprocessing...")
    preprocess_images(input_dir="./raw_data/test_arina", output_dir="./processed/test_arina")
    print("Image preprocessing completed.")


if __name__ == "__main__":
    run()
