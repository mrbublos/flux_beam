from beam import function, Volume, Image, Output

from src.app.inference import inference, GenerateArgs, LoraStyle
from src.app.logger import Logger

VOLUME_PATH = "/mnt/code/models"
RAW_VOLUME_PATH = "/mnt/code/raw_data"
PROCESSED_VOLUME_PATH = "/mnt/code/processed"
LORAS_VOLUME_PATH = "/mnt/code/loras"

logger = Logger(__name__)

@function(
    image=Image(python_version="python3.11")
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
    gpu="H100",
    secrets=["HF_TOKEN"],
    env={
        "STYLES_FOLDER": "/mnt/code/loras/common",
        "USER_MODELS_FOLDER": "/mnt/code/loras",
        "MODEL_NAME": "/mnt/code/models/flux_schnell",
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
    """
    Execute the LoRA training and inference process.

    This function initializes the training environment and runs an inference
    operation to generate an image based on a given prompt and style settings.
    It utilizes multiple libraries for model handling, inference, and logging.
    The generated image is saved, and a public URL for the output is logged.

    Environment and settings:

    The function performs the following steps:
    1. Logs the start of the LoRA training.
    2. Defines user ID and LoRA styles.
    3. Calls the inference function with specified arguments to generate an image.
    4. Saves the generated image and retrieves its public URL.
    5. Logs successful completion and output URL.
    """

    logger.info("Starting lora train...")

    # files = os.listdir("/workspace/character_training")
    # print(files)

    user_id = "test_arina"
    lora_styles = [
        LoraStyle(path=f"Disney-Studios-Flux-000008.safetensors", scale=0.7, name="disney"),
        LoraStyle(path=f"amateurphoto-v6-forcu.safetensors", scale=0.7, name="realistic"),
    ]
    pil_result, bytes_result = inference(GenerateArgs(
        user_id=user_id,
        lora_styles=[],
        lora_personal=True,
        num_steps=5,
        prompt="High-fashion editorial style ultra-realistic photo in 8K. A young woman with extremely long, perfectly straight, jet-black hair stands gracefully in front of a clean matte pink background. Behind her, thin white geometric arches of various heights are arranged in a harmonious, rhythmic composition — evoking a futuristic, minimalist dreamscape.She wears a structured, all-white asymmetrical jumpsuit with exaggerated shoulders and a cinched waist — a fashion-forward silhouette inspired by futuristic runway looks (think Rick Owens x Loewe). Her expression is neutral, slightly melancholic, adding conceptual depth to the scene. She stands with her hands relaxed, one leg slightly bent. Natural light from the left creates soft shadows, emphasizing the contours of her face and outfit.Fashion mood: surreal minimalism meets future couture.",
        width=1024,
        height=1024,
        guidance=3.5,
    ))

    output = Output.from_pil_image(pil_result)
    output.save()

    # Retrieve pre-signed URL for output file
    url = output.public_url(expires=400)
    logger.info(f"Inference completed successfully. Output saved to: {url}")

if __name__ == "__main__":
    run()
