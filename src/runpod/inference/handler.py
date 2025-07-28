from uuid import uuid4

import runpod

from src.app.inference import inference, GenerateArgs, get_generator, LoraStyle
from src.app.logger import Logger
from src.app.s3client import S3Client
from src.runpod.inference.download import download_file

logger = Logger(__name__)

get_generator()

s3_client = S3Client()

def run(event):
    try:
        inputs = event["input"]
        user_id = inputs["user_id"]
        prompt = inputs["prompt"]
        num_steps = inputs["num_steps"] if "num_steps" in inputs else 28
        width = inputs["width"] if "width" in inputs else 1024
        height = inputs["height"] if "height" in inputs else 1024
        styles = inputs["styles"] if "styles" in inputs else []
        logger.info(f"Running inference for user {user_id} with {inputs}")

        logger.info("Starting lora inference...")
        for style in styles:
            logger.info(f"Style: {style}")
            if "link" in style:
                logger.info(f"Downloading style {style['link']}")
                file_name, _ = download_file(style["link"])
                style["path"] = file_name
                style["name"] = file_name
                style["scale"] = style["weight"]


        lora_styles = [LoraStyle(name=dict["name"], path=dict["path"], scale=dict["scale"]) for dict in styles]
        pil_result, bytes_result = inference(GenerateArgs(
            user_id=user_id,
            lora_styles=lora_styles,
            lora_personal=True,
            num_steps=num_steps,
            prompt=prompt,
            width=width,
            height=height,
            guidance=3.5,
        ), get_generator())

        logger.info(f"Inference done for user {user_id}")

        stored_file_name = f"inference/{user_id}-{uuid4()}.jpeg"
        s3_client.remove_object(object_name=stored_file_name)
        result = s3_client.upload_file(object_name=stored_file_name, data=bytes_result)
        return {
            "filename": stored_file_name,
            "user_id": user_id,
            "prompt": prompt,
            "num_steps": num_steps,
            "success": result,
        }
    except Exception as e:
        logger.error(f"Error running inference {e}")
        return {
            "error": str(e),
            "success": False,
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": run})