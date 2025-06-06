from beam import function, Volume, Image, env

if env.is_remote():
    from huggingface_hub import snapshot_download

VOLUME_PATH = "./models"

@function(
    image=Image(python_version="python3.11")
    .add_python_packages(
        [
            "huggingface_hub",
            "huggingface_hub[hf-transfer]",
        ]
    )
    .with_envs("HF_HUB_ENABLE_HF_TRANSFER=1"),
    memory="2Gi",
    cpu=0.125,
    secrets=["HF_TOKEN"],
    volumes=[Volume(name="models", mount_path=VOLUME_PATH)],
    )
def upload():
    snapshot_download(
        repo_id="black-forest-labs/FLUX.1-schnell",
        local_dir=f"{VOLUME_PATH}/flux_schnell"
    )

    print("Files downloaded successfully")

if __name__ == "__main__":
    upload()