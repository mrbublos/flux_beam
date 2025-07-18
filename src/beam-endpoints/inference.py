from beam import Volume, Image, Output, task_queue

from src.app.inference import inference, GenerateArgs, get_generator, LoraStyle
from src.app.logger import Logger

VOLUME_PATH = "/mnt/code/models"
RAW_VOLUME_PATH = "/mnt/code/raw_data"
PROCESSED_VOLUME_PATH = "/mnt/code/processed"
LORAS_VOLUME_PATH = "/mnt/code/loras"

STYLES = ({
    "disney": LoraStyle(path=f"Disney-Studios-Flux-000008.safetensors", scale=0.7, name="disney"),
    "realistic": LoraStyle(path=f"amateurphoto-v6-forcu.safetensors", scale=0.7, name="realistic"),
})

logger = Logger(__name__)

generator = None

def on_start():
    logger.info("Inference endpoint started")
    global generator
    generator = get_generator()
    logger.info("Generator loaded")
    return generator

@task_queue(
    name="inference",
    image=Image(base_image="pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime")
    .add_python_packages(
        [
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
            "para-attn",
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
        ]
    ),
    gpu=["H100"],
    secrets=["HF_TOKEN"],
    env={
        "STYLES_FOLDER": "/mnt/code/loras/common",
        "USER_MODELS_FOLDER": "/mnt/code/loras",
        "MODEL_NAME": "/mnt/code/models/flux_dev",
        "HF_OFFLINE": "1",
    },
    volumes=[
        Volume(name="models", mount_path=VOLUME_PATH),
        Volume(name="raw_data", mount_path=RAW_VOLUME_PATH),
        Volume(name="processed", mount_path=PROCESSED_VOLUME_PATH),
        Volume(name="loras", mount_path=LORAS_VOLUME_PATH),
    ],
    on_start=on_start,
)
def run(context, **inputs):

    user_id = inputs["user_id"]
    prompt = inputs["prompt"]
    num_steps = inputs["num_steps"] if "num_steps" in inputs else 28
    lora_styles = [STYLES[x] for x in inputs["lora_styles"]] if "lora_styles" in inputs else []

    logger.info(f"Running inference for user {user_id} with prompt: {prompt}")

    logger.info("Starting lora inference...")

    pil_result, bytes_result = inference(GenerateArgs(
        user_id=user_id,
        lora_styles=lora_styles,
        lora_personal=True,
        num_steps=num_steps,
        prompt=prompt,
        width=1024,
        height=1024,
        guidance=3.5,
    ), context.on_start_value)

    output = Output.from_pil_image(pil_result)
    output.save()

    logger.info(f"Inference completed successfully for {user_id}.")

if __name__ == "__main__":
    run.put(user_id="test_arina", prompt="lady with a cat")