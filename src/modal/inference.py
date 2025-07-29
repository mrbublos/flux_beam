from uuid import uuid4

import modal

from src.app.s3client import S3Client

# Define the Modal App
app = modal.App("Inference")

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
        modal.Secret.from_name("runpod-s3"),
    ],
    max_containers=5,
)
class Inference:
    @modal.enter()
    def load_model(self):
        self.STYLES = {}
        self.s3_client = S3Client()

        from src.app.inference import get_generator, LoraStyle
        from src.app.logger import Logger

        self.logger = Logger(__name__)
        self.logger.info("Inference endpoint starting")
        self.generator = get_generator()
        self.logger.info("Generator loaded")

        # Define styles after generator is loaded
        self.STYLES = {
            "disney": LoraStyle(
                path=f"/mnt/loras/common/Disney-Studios-Flux-000008.safetensors",
                scale=0.7,
                name="disney",
            ),
            "realistic": LoraStyle(
                path=f"/mnt/loras/common/amateurphoto-v6-forcu.safetensors",
                scale=0.7,
                name="realistic",
            ),
        }

    @modal.method()
    def run(self, data: dict):
        self.logger.info("Starting lora inference...")

        volume_loras.reload()

        user_id = data["user_id"]
        prompt = data["prompt"]
        num_steps = data["num_steps"] if "num_steps" in data else 50
        lora_styles = data["lora_styles"] if "lora_styles" in data else []
        guidance = data["guidance"] if "guidance" in data else 3.5

        from src.app.inference import inference, GenerateArgs

        selected_styles = [self.STYLES[style] for style in lora_styles if style in self.STYLES]

        self.logger.info(f"Running inference for user {user_id} with prompt: {prompt}")

        pil_result, bytes_result = inference(
            GenerateArgs(
                user_id=user_id,
                lora_styles=selected_styles,
                lora_personal=True,
                num_steps=num_steps,
                prompt=prompt,
                width=1024,
                height=1024,
                guidance=guidance,
            ),
            self.generator,
        )

        self.logger.info(f"Inference completed successfully for {user_id}.")

        stored_file_name = f"inference/{user_id}-{uuid4()}.jpeg"
        self.s3_client.remove_object(object_name=stored_file_name)
        result = self.s3_client.upload_file(object_name=stored_file_name, data=bytes_result)

        # with open("/mnt/" + stored_file_name, "wb") as f:
        #     f.write(bytes_result)

        return {
            "filename": stored_file_name,
            "user_id": user_id,
            "prompt": prompt,
            "num_steps": num_steps,
            "success": result,
        }


@app.local_entrypoint()
def generate():
    user_id = "test_arina"
    prompt = "lady with a cart"

    # Run inference
    result = Inference().run.remote(user_id=user_id, prompt=prompt, num_steps=50)
    print(f"modal volume get inference {result['filename'].replace('inference/', '')}")
    # S3Client().remove_object(result.filename)

    print(result)
