import base64
import hashlib

import runpod

from src.app.inference import inference, GenerateArgs, get_generator, LoraStyle
from src.app.logger import Logger
from src.runpod.style_check.join_images import combine_pil_images_to_bytes

LORAS_VOLUME_PATH = "/runpod-volume/loras/common"

logger = Logger(__name__)

generator = get_generator()

def run(event):
    inputs = event["input"]
    user_id = inputs["user_id"]
    prompt = inputs["prompt"]
    num_steps = inputs["num_steps"] if "num_steps" in inputs else 28
    style_link = inputs["style_link"]
    style_name = f"{hashlib.md5(str(style_link).encode()).hexdigest()}.safetensors"

    logger.info(f"Running inference for user {user_id} with prompt: {prompt} with styles: {style_link}")

    logger.info("Starting lora inference...")

    global generator

    images = []
    labels = []
    for scale in [round(x * 0.1, 1) for x in range(1, 11)]:
        lora_styles = [LoraStyle(name="style", path=f"{LORAS_VOLUME_PATH}/{style_name}", scale=scale)]
        pil_result, bytes_result = inference(GenerateArgs(
            user_id="test_arina",
            lora_styles=lora_styles,
            lora_personal=True,
            num_steps=num_steps,
            prompt=prompt,
            width=1024,
            height=1024,
            guidance=3.5,
        ), generator)
        images.append(pil_result)
        labels.append(str(scale))

    logger.info("Combining images...")
    result = combine_pil_images_to_bytes(images, labels)
    logger.info("Images combined")
    logger.info("Converting to base64...")
    base64_result = base64.b64encode(result).decode("utf-8")
    logger.info("Converted to base64")

    return {
        "result": base64_result,
        "user_id": user_id,
        "prompt": prompt,
        "num_steps": num_steps,
        "style_link": style_link,
        "success": True,
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": run})