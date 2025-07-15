import base64

import runpod

from src.app.inference import inference, GenerateArgs, get_generator, LoraStyle
from src.app.logger import Logger
from src.app.s3client import S3Client
from src.runpod.style_check.download import download_file
from src.runpod.style_check.join_images import combine_pil_images_to_bytes

logger = Logger(__name__)

get_generator()

print("Starting style check handler...")

s3_client = S3Client()

def run(event):
    try:
        inputs = event["input"]
        user_id = inputs["user_id"]
        prompt = inputs["prompt"]
        num_steps = inputs["num_steps"] if "num_steps" in inputs else 28
        style_link = inputs["style_link"]
        logger.info(f"Running inference for user {user_id} with prompt: {prompt} with styles: {style_link}")

        logger.info("Starting lora inference...")

        style_name, description_file_name = download_file(style_link)

        images = []
        labels = []
        for scale in [round(x * 0.1, 1) for x in range(1, 11)]:
            lora_styles = [LoraStyle(name="style", path=style_name, scale=scale)]
            pil_result, bytes_result = inference(GenerateArgs(
                user_id="test_arina",
                lora_styles=lora_styles,
                lora_personal=True,
                num_steps=num_steps,
                prompt=prompt,
                width=1024,
                height=1024,
                guidance=3.5,
            ), get_generator())
            images.append(pil_result)
            labels.append(str(scale))

        logger.info("Combining images...")
        result = combine_pil_images_to_bytes(images, labels)
        logger.info("Images combined")
        logger.info("Converting to base64...")
        logger.info("Converted to base64")

        stored_file_name = f"{style_name}.png"
        s3_client.remove_object(object_name=stored_file_name)
        result = s3_client.upload_file(object_name=stored_file_name, data=result)
        return {
            "filename": stored_file_name,
            "user_id": user_id,
            "prompt": prompt,
            "num_steps": num_steps,
            "style_link": style_link,
            "success": result,
        }
    except Exception as e:
        logger.error(f"Error running inference {e}")
        return {
            "error": str(e),
            "success": False,
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": run, "return_aggregate_stream": True})