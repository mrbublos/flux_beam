import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
from diffusers import FluxTransformer2DModel, FluxPipeline, AutoencoderKL

from src.app.logger import Logger

logger = Logger("model_loader")

MODEL_NAME = os.getenv("MODEL_NAME", "black-forest-labs/flux.1-dev")
HF_TOKEN = os.getenv("HF_TOKEN", None)
OFFLOAD_TO_CPU = os.getenv("OFFLOAD_TO_CPU", "false")

def offload_to_cpu():
    return OFFLOAD_TO_CPU == "true"

def load_models_concurrently(load_functions_map: dict) -> dict:
    model_id_to_model = {}
    with ThreadPoolExecutor(max_workers=len(load_functions_map)) as executor:
        future_to_model_id = {
            executor.submit(load_fn): model_id
            for model_id, load_fn in load_functions_map.items()
        }
        for future in as_completed(future_to_model_id.keys()):
            model_id_to_model[future_to_model_id[future]] = future.result()
    return model_id_to_model

def load_encoder():
    """Load the Flux encoder model"""
    logger.info(f"Loading encoder...{MODEL_NAME}")
    dtype = torch.bfloat16

    encoder = FluxPipeline.from_pretrained(
        MODEL_NAME,
        transformer=None,
        vae=None,
        torch_dtype=dtype,
        local_files_only=True,
        token=HF_TOKEN,
        device_map="balanced",
    )

    return encoder

def load_model():
    logger.info(f"Loading model...{MODEL_NAME}")
    dtype = torch.bfloat16

    # Load transformer with quantization
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_NAME,
        subfolder="transformer",
        quantization_config=None,  # DiffusersBitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=dtype,
        token=HF_TOKEN,
        local_files_only=True,
        device_map="auto",
    )

    # Initialize pipeline with transformer
    model = FluxPipeline.from_pretrained(
        MODEL_NAME,
        transformer=transformer,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        vae=None,
        torch_dtype=dtype,
        token=HF_TOKEN,
        local_files_only=True,
        device_map="auto",
    )

    apply_cache_on_pipe(model, residual_diff_threshold=0.12)
    model.to(memory_format=torch.channels_last)
    transformer.fuse_qkv_projections()

    return model

def load_autoencoder():
    logger.info(f"Loading autoencoder...{MODEL_NAME}")
    dtype = torch.bfloat16

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        MODEL_NAME,
        subfolder="vae",
        torch_dtype=dtype,
        token=HF_TOKEN,
        local_files_only=True,
        device_map="auto",
    )
    vae.to(memory_format=torch.channels_last)

    vae.fuse_qkv_projections()
    vae = torch.compile(vae)
    return vae
