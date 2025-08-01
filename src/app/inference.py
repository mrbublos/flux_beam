import gc
import io
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
from pydantic import BaseModel, Field

from src.app.logger import Logger
from src.app.model_loader import load_models_concurrently, load_encoder, load_model, load_autoencoder, offload_to_cpu

# Constants

MAX_RAND = 2 ** 32 - 1
STYLES_FOLDER = os.getenv("STYLES_FOLDER", "/lora_styles")
USER_MODELS = os.getenv("USER_MODELS_FOLDER", f"/user_models")
MODEL_NAME = os.getenv("MODEL_NAME", "black-forest-labs/flux.1-dev")
HF_TOKEN = os.getenv("HF_TOKEN", None)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

logger = Logger("FluxGenerator")

_generator_instance = None


class LoraStyle(BaseModel):
    path: str
    scale: float = Field(default=1.0)
    name: Optional[str] = None


class GenerateArgs(BaseModel):
    prompt: str
    width: Optional[int] = Field(default=1024)
    height: Optional[int] = Field(default=720)
    num_steps: Optional[int] = Field(default=28)
    guidance: Optional[float] = Field(default=3.5)
    seed: Optional[int] = Field(default_factory=lambda: np.random.randint(0, MAX_RAND), gt=0, lt=MAX_RAND)
    lora_personal: Optional[bool] = None
    lora_styles: Optional[list[LoraStyle]] = None
    user_id: str

def cuda_info():
    logger.info("CUDA info...")
    logger.info(str(torch.cuda.is_available()))
    logger.info(str(torch.cuda.device_count()))
    logger.info(str(torch.cuda.current_device()))

def flush():
    """Clear CUDA memory cache"""
    logger.info(f"Flushing CUDA cache...")
    gc.collect()
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device=device)
    torch.cuda.reset_peak_memory_stats(device=device)
    torch.cuda.synchronize(device=device)


class FluxGenerator:
    def __init__(self):
        flush()
        self.setup_environment()
        start_time = time.time()
        self.load_models()
        # self.load_models_async()
        end_time = time.time()
        logger.info(f"Models loaded in {end_time - start_time:.2f} seconds")
        os.makedirs(STYLES_FOLDER, exist_ok=True)

    def setup_environment(self):
        """Configure PyTorch and CUDA settings"""
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark_limit = 20
        torch.set_float32_matmul_precision("high")

    def load_models_async(self):
        logger.info(f"Loading async models...{MODEL_NAME}")

        components = load_models_concurrently({
            "encoder": load_encoder,
            "model": load_model,
            "autoencoder": load_autoencoder,
        })
        self.encoder = components["encoder"]
        self.model = components["model"]
        self.model.to("cpu" if offload_to_cpu() else "cuda")
        self.vae = components["autoencoder"]
        self.vae.to("cpu" if offload_to_cpu() else "cuda")

    def load_models(self):
        """Load all required models"""
        logger.info(f"Loading models...{MODEL_NAME}")
        dtype = torch.bfloat16

        # Load encoder
        self.encoder = FluxPipeline.from_pretrained(
            MODEL_NAME,
            transformer=None,
            vae=None,
            torch_dtype=dtype,
            local_files_only=True,
            token=HF_TOKEN,
        )
        self.encoder.to("cpu" if offload_to_cpu() else "cuda")

        # Load transformer with quantization
        transformer = FluxTransformer2DModel.from_pretrained(
            MODEL_NAME,
            subfolder="transformer",
            quantization_config=None,  # DiffusersBitsAndBytesConfig(load_in_8bit=True),
            torch_dtype=dtype,
            token=HF_TOKEN,
            local_files_only=True,
        )

        # Initialize pipeline with transformer
        self.model = FluxPipeline.from_pretrained(
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
        )
        self.model.to("cpu" if offload_to_cpu() else "cuda")

        apply_cache_on_pipe(self.model, residual_diff_threshold=0.12)
        self.model.to(memory_format=torch.channels_last)

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            MODEL_NAME,
            subfolder="vae",
            torch_dtype=dtype,
            token=HF_TOKEN,
            local_files_only=True,
        )
        self.vae.to("cpu" if offload_to_cpu() else "cuda")
        self.vae.to(memory_format=torch.channels_last)

        config = torch._inductor.config
        config.conv_1x1_as_mm = True
        transformer.fuse_qkv_projections()
        self.vae.fuse_qkv_projections()
        self.vae = torch.compile(self.vae)

        # Warmup
        logger.info("Performing warmup inference...")
        self.generate(GenerateArgs(
            prompt="Warmup inference",
            width=1024,
            height=1024,
            num_steps=4,
            guidance=3.5,
            seed=10,
            user_id="test",
            lora_personal=False,
            lora_styles=[],
        ))

    def generate(self, args: GenerateArgs) -> Tuple[any, bytes]:
        """Generate image from input parameters"""
        lora_names = []
        lora_scales = []

        flush()

        logger.info(f"Generating image for user {args}")

        try:

            logger.info(f"Prompt encoding prompt")
            self.encoder.to("cuda") if offload_to_cpu() else None
            # Encode prompt
            with torch.inference_mode():
                prompt_embeds, pooled_prompt_embeds, _ = self.encoder.encode_prompt(
                    args.prompt,
                    prompt_2=None,
                    max_sequence_length=512,
                    num_images_per_prompt=1
                )
            self.encoder.to("cpu") if offload_to_cpu() else None

            flush()

            # Handle LoRA loading
            if args.lora_personal:
                personal_lora = f"{USER_MODELS}/{args.user_id}/{args.user_id}.safetensors"
                logger.info(f"Using personal style {personal_lora}")
                self.model.load_lora_weights(personal_lora, adapter_name="user")
                lora_names.append("user")
                lora_scales.append(1.0)

            if args.lora_styles:
                for style in args.lora_styles:
                    if not style.path:
                        continue
                    style_path = f"{STYLES_FOLDER}/{style.path}"
                    logger.info(f"Using lora style {style_path} with scale {style.scale}")

                    self.model.load_lora_weights(style_path, adapter_name=style.name)
                    lora_names.append(style.name)
                    lora_scales.append(style.scale)

            if len(lora_names) > 0:
                logger.info(f"Applying {len(lora_names)} LoRA(s)")
                self.model.set_adapters(lora_names, adapter_weights=lora_scales)

            self.model.to("cuda") if offload_to_cpu() else None
            # Generate latents
            with torch.inference_mode():
                latents = self.model(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance,
                    height=args.height,
                    width=args.width,
                    output_type="latent"
                ).images
            self.model.to("cpu") if offload_to_cpu() else None

            logger.info("Decoding image")
            # Decode image
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

            self.vae.to("cuda") if offload_to_cpu() else None
            with torch.inference_mode():
                latents = FluxPipeline._unpack_latents(latents, args.height, args.width, vae_scale_factor)
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latents, return_dict=False)[0]
                image = image_processor.postprocess(image, output_type="pil")[0]
            self.vae.to("cpu") if offload_to_cpu() else None

            logger.info("Converting image")
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="JPEG", quality=95)
            return image, img_byte_arr.getvalue()

        except Exception as e:
            logger.error(f"Error generating image {str(e)}")
            raise RuntimeError(f"Image generation failed: {str(e)}")
        finally:
            # Unload LoRAs if used
            if lora_names is not None and len(lora_names) > 0:
                self.model.unload_lora_weights(reset_to_overwritten_params=True)
                logger.info(f"Unloaded LoRA weights: {lora_names}")

            flush()

def get_generator() -> FluxGenerator:
    logger.info("CUDA info...")
    logger.info(str(torch.cuda.is_available()))
    logger.info(str(torch.cuda.device_count()))
    logger.info(str(torch.cuda.current_device()))

    global _generator_instance
    if _generator_instance is None:
        logger.info("Initializing FluxGenerator...")
        _generator_instance = FluxGenerator()
    return _generator_instance

def inference(args: GenerateArgs, generator: FluxGenerator) -> Tuple[any, bytes]:
    """RunPod handler function"""
    try:
        logger.info(f"Running inference for user {args.user_id} with prompt: {args.prompt}")

        logger.info("Generating image")
        # Generate image
        pil_image, image_bytes = generator.generate(args)

        logger.info(f"Inference finished for {args.user_id} len: {len(image_bytes)}")
        return pil_image, image_bytes

    except Exception as e:
        logger.error(f"Error running inference {e}")
        return {
            "error": str(e)
        }
