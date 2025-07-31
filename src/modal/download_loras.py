import modal

from src.app.download_loras import download_lora

image = (
    modal.Image
    .debian_slim(python_version="3.11")
    .pip_install(
        [
            "requests",
            "huggingface_hub",
            "huggingface_hub[hf-transfer]",
        ])
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    .add_local_python_source("src")
)
app = modal.App(name="model-downloader")


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("civitai-secret"),
    ],
    volumes={
        "/mnt/code/models": modal.Volume.from_name("models", create_if_missing=True),
        "/mnt/code/loras": modal.Volume.from_name("loras", create_if_missing=True),
    },
)
def download():
    loras_to_download = [
        "https://civitai.com/api/download/models/990315?type=Model&format=SafeTensor",
        "https://civitai.com/api/download/models/1943855?type=Model&format=SafeTensor",
        "https://civitai.com/api/download/models/1111847?type=Model&format=SafeTensor",
        "https://civitai.com/api/download/models/706528?type=Model&format=SafeTensor",
        "https://civitai.com/api/download/models/990315?type=Model&format=SafeTensor",
        "https://civitai.com/api/download/models/1177595?type=Model&format=SafeTensor",
        "https://civitai.com/api/download/models/1874953?type=Model&format=SafeTensor",
        "https://civitai.com/api/download/models/1423082?type=Model&format=SafeTensor",
        "https://civitai.com/api/download/models/194886?type=Model&format=SafeTensor",
    ]

    for lora in loras_to_download:
        download_lora(url=lora, destination_folder="/mnt/code/loras/common")


if __name__ == "__main__":
    download.remote()
