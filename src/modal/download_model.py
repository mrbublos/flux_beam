import modal

from huggingface_hub import snapshot_download

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    [
        "huggingface_hub",
        "huggingface_hub[hf-transfer]",
    ]).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
app = modal.App(name="model-downloader")

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/mnt/code/models": modal.Volume.from_name("models", create_if_missing=True),
    },
)
def download():
    snapshot_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        local_dir=f"/mnt/code/models/flux_dev"
    )
    snapshot_download(
        repo_id="microsoft/Florence-2-large",
        local_dir=f"/mnt/code/models/florence_2_large"
    )

if __name__ == "__main__":
    download.remote()