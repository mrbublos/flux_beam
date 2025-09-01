import base64

import modal

from src.app.download_loras import download_lora
from src.app.inference import LoraStyle, inference, GenerateArgs
from src.app.join_images import combine_pil_images_to_bytes
from src.app.s3client import S3Client

# Define the Modal App
app = modal.App("CheckStyle")

# Define Network File Systems for persistent storage
volume_models = modal.Volume.from_name("models", create_if_missing=True)
volume_raw = modal.Volume.from_name("raw-data", create_if_missing=True)
volume_processed = modal.Volume.from_name("processed-data", create_if_missing=True)
volume_loras = modal.Volume.from_name("loras", create_if_missing=True)

# Define the container image
image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime")
    .pip_install(
        "accelerate",
        "annotated_types",
        "bitsandbytes==0.45.5",
        "click",
        "diffusers",
        "einops",
        "fastapi",
        "huggingface_hub",
        "loguru",
        "optimum-quanto==0.2.5",
        "peft",
        "protobuf",
        "pybase64",
        "pydantic",
        "pydantic_core",
        "pydash",
        "PyTurboJPEG",
        "quanto",
        "regex",
        "safetensors",
        "sentencepiece",
        "starlette",
        "tokenizers",
        "transformers==4.51.2",
        "triton",
        "typing_inspection",
        "uvicorn",
        "boto3",
        "para-attn",
    )
    .apt_install("build-essential")
    .pip_install("fastapi[standard]")
    .pip_install("requests")
    .pip_install(["huggingface_hub", "huggingface_hub[hf-transfer]"])
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "$CUDA_HOME/bin:$PATH",
        "HF_OFFLINE": "1",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        "TORCH_CUDA_ARCH_LIST": "Turing",
        "STYLES_FOLDER": "/mnt/loras/common",
        "USER_MODELS_FOLDER": "/mnt/loras",
        "MODEL_NAME": "/mnt/models/flux_dev",

    })
    .add_local_python_source("src")
)


@app.cls(
    image=image,
    gpu="L40S",
    volumes={
        "/mnt/models": volume_models,
        "/mnt/raw_data": volume_raw,
        "/mnt/processed": volume_processed,
        "/mnt/loras": volume_loras,
        # "/mnt/inference": modal.Volume.from_name("inference", create_if_missing=True),
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("s3-inference"),
    ],
    max_containers=1,
    scaledown_window=10,
)
class CheckStyle:
    @modal.enter()
    def load_model(self):
        self.STYLES = {}

        from src.app.inference import get_generator
        from src.app.logger import Logger

        self.logger = Logger(__name__)
        self.logger.info("Inference endpoint starting")
        self.generator = get_generator()
        self.logger.info("Generator loaded")
        self.s3_client = S3Client()

    @modal.method()
    def run(self, inputs: dict):
        try:
            user_id = inputs["user_id"]
            prompt = inputs["prompt"]
            num_steps = inputs["num_steps"] if "num_steps" in inputs else 28
            width = inputs["width"] if "width" in inputs else 1024
            height = inputs["height"] if "height" in inputs else 1024
            min_scale = inputs["min_scale"] if "min_scale" in inputs else 0.1
            max_scale = inputs["max_scale"] if "max_scale" in inputs else 1
            scale_step = inputs["scale_step"] if "scale_step" in inputs else 0.1
            style_link = inputs["style_link"]
            self.logger.info(f"Running inference for user {user_id} with prompt: {prompt} with styles: {style_link}")

            self.logger.info("Downloading style...")
            style_name, description_file_name = download_lora(style_link, "/mnt/loras/common")
            self.logger.info(f"Style downloaded {style_link}")

            images = []
            labels = []
            steps = int((max_scale - min_scale) / scale_step + 1)
            for scale in [round(min_scale + i * scale_step, 2) for i in range(steps)]:
                lora_styles = [LoraStyle(name="style", path=style_name, scale=scale)]
                pil_result, bytes_result = inference(GenerateArgs(
                    user_id=user_id,
                    lora_styles=lora_styles,
                    lora_personal=True,
                    num_steps=num_steps,
                    prompt=prompt,
                    width=width,
                    height=height,
                    guidance=3.5,
                ), self.generator)
                images.append(pil_result)
                labels.append(str(scale))

            self.logger.info("Combining images...")
            result = combine_pil_images_to_bytes(images, labels)
            self.logger.info("Images combined")
            self.logger.info("Converting to base64...")
            self.logger.info("Converted to base64")

            stored_file_name = f"style_checks/{style_name}.jpeg"
            self.s3_client.remove_object(object_name=stored_file_name)
            result = self.s3_client.upload_file(object_name=stored_file_name, data=result)
            return {
                "filename": stored_file_name,
                "user_id": user_id,
                "prompt": prompt,
                "num_steps": num_steps,
                "style_link": style_link,
                "status": "success",
            }
        except Exception as e:
            self.logger.error(f"Error running inference {e}")
            return {
                "error": str(e),
                "status": "error",
            }


@app.local_entrypoint()
def generate():
    user_id = "test_arina"
    prompt = "lady with a cart"

    # Run inference
    result = Inference().run.remote(user_id=user_id, prompt=prompt, num_steps=50)
    print(f"modal volume get inference {result['filename'].replace('inference/', '')}")
    bytes = base64.b64decode(result["image_data"])
    with open(f"output_{user_id}.jpeg", "wb") as f:
        f.write(bytes)

    print(result)
