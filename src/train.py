import os
import logging
from dataclasses import dataclass

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__)

@dataclass
class LoRATrainerConfig:
    base_model_id: str
    input_image_folder: str
    output_lora_folder: str
    training_params: dict
    use_offline_mode: bool = True

class LoRATrainer:
    def __init__(self, config: LoRATrainerConfig):
        self.config = config
        self.accelerator = Accelerator(
            mixed_precision=self.config.training_params.get("mixed_precision", None)
        )
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)

        if self.config.training_params.get("seed") is not None:
            set_seed(self.config.training_params["seed"])

        if self.accelerator.is_main_process:
            os.makedirs(self.config.output_lora_folder, exist_ok=True)

        # Check for xformers
        if self.config.training_params.get("enable_xformers_memory_efficient_attention", False):
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training. Please install 0.0.17 or higher. If you can't update to 0.0.17 or higher, consider downgrading to 0.0.15.x and disabling xformers."
                    )
                else:
                    self.enable_xformers = True
            else:
                logger.warn(
                    "xformers memory efficient attention is only available with xformers installed. Ignoring setting."
                )
                self.enable_xformers = False
        else:
            self.enable_xformers = False


    def _load_model_and_tokenizer(self):
        # Load scheduler, tokenizer and models.
        noise_scheduler = DDPMScheduler.from_pretrained(self.config.base_model_id, subfolder="scheduler", local_files_only=self.config.use_offline_mode)
        tokenizer = CLIPTokenizer.from_pretrained(self.config.base_model_id, subfolder="tokenizer", local_files_only=self.config.use_offline_mode)
        text_encoder = CLIPTextModel.from_pretrained(self.config.base_model_id, subfolder="text_encoder", local_files_only=self.config.use_offline_mode)
        vae = AutoencoderKL.from_pretrained(self.config.base_model_id, subfolder="vae", local_files_only=self.config.use_offline_mode)
        unet = UNet2DConditionModel.from_pretrained(self.config.base_model_id, subfolder="unet", local_files_only=self.config.use_offline_mode)

        # Freeze vae and text_encoder
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        # Freeze unet layers
        unet.requires_grad_(False)

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # if self.accelerator.mixed_precision == "fp16":
        #     text_encoder.to(self.accelerator.device, dtype=torch.float16)
        #     vae.to(self.accelerator.device, dtype=torch.float16)
        # elif self.accelerator.mixed_precision == "bf16":
        #     text_encoder.to(self.accelerator.device, dtype=torch.bfloat16)
        #     vae.to(self.accelerator.device, dtype=torch.bfloat16)

        # Move unet, vae and text_encoder to device and cast to desired data type
        unet.to(self.accelerator.device, dtype=torch.float32) # LoRA training typically uses full precision for UNet
        vae.to(self.accelerator.device, dtype=torch.float32)
        text_encoder.to(self.accelerator.device, dtype=torch.float32)


        if self.enable_xformers:
            unet.enable_xformers_memory_efficient_attention()

        # Add LoRA layers
        unet.enable_lora()
        text_encoder.enable_lora()


        return noise_scheduler, tokenizer, text_encoder, vae, unet

    def _prepare_dataset(self):
        class SelfieDataset(Dataset):
            def __init__(self, image_folder, tokenizer, size=512, center_crop=True):
                self.image_folder = image_folder
                self.image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                self.tokenizer = tokenizer
                self.size = size
                self.center_crop = center_crop
                self.transform = transforms.Compose([
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                self.caption = "a photo of a person" # Simple generic caption

            def __len__(self):
                return len(self.image_files)

            def __getitem__(self, i):
                image_path = self.image_files[i]
                image = Image.open(image_path).convert("RGB")
                image = self.transform(image)

                # Tokenize caption
                input_ids = self.tokenizer(
                    self.caption,
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
                return {"pixel_values": image, "input_ids": input_ids.squeeze()}

        dataset = SelfieDataset(
            image_folder=self.config.input_image_folder,
            tokenizer=self.tokenizer,
            size=self.config.training_params.get("resolution", 512),
            center_crop=self.config.training_params.get("center_crop", True),
        )

        dataloader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.config.training_params.get("train_batch_size", 1),
            num_workers=self.config.training_params.get("dataloader_num_workers", 0),
        )
        return dataloader

    def _save_lora_model(self, epoch):
        # Save only the LoRA layers
        unet_lora_state_dict = unet.lora_state_dict()
        text_encoder_lora_state_dict = text_encoder.lora_state_dict()

        StableDiffusionPipeline.save_lora_weights(
            save_directory=self.config.output_lora_folder,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=text_encoder_lora_state_dict,
            safe_serialization=True,
        )
        logger.info(f"LoRA weights saved to {self.config.output_lora_folder} at epoch {epoch}")


    def train(self):
        noise_scheduler, tokenizer, text_encoder, vae, unet = self._load_model_and_tokenizer()
        self.tokenizer = tokenizer # Store tokenizer for dataset preparation
        dataloader = self._prepare_dataset()

        # Optimizer
        # Only optimize LoRA layers
        lora_layers = list(unet.lora_layers.parameters()) + list(text_encoder.lora_layers.parameters())
        optimizer = torch.optim.AdamW(
            lora_layers,
            lr=self.config.training_params.get("learning_rate", 1e-4),
            betas=(self.config.training_params.get("adam_beta1", 0.9), self.config.training_params.get("adam_beta2", 0.999)),
            weight_decay=self.config.training_params.get("adam_weight_decay", 1e-4),
            eps=self.config.training_params.get("adam_epsilon", 1e-08),
        )

        # Scheduler
        lr_scheduler = get_scheduler(
            self.config.training_params.get("lr_scheduler", "constant"),
            optimizer=optimizer,
            num_warmup_steps=self.config.training_params.get("lr_warmup_steps", 500),
            num_training_steps=self.config.training_params.get("max_train_steps", len(dataloader) * self.config.training_params.get("num_train_epochs", 100)),
        )

        # Prepare with accelerator
        unet, text_encoder, optimizer, dataloader, lr_scheduler = self.accelerator.prepare(
            unet, text_encoder, optimizer, dataloader, lr_scheduler
        )

        # Train!
        total_batch_size = self.config.training_params.get("train_batch_size", 1) * self.accelerator.num_processes * self.config.training_params.get("gradient_accumulation_steps", 1)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(dataloader.dataset)}")
        logger.info(f"  Num Epochs = {self.config.training_params.get("num_train_epochs", 100)}")
        logger.info(f"  Instantaneous batch size per device = {self.config.training_params.get("train_batch_size", 1)}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.training_params.get("gradient_accumulation_steps", 1)}")
        logger.info(f"  Total optimization steps = {self.config.training_params.get("max_train_steps", len(dataloader) * self.config.training_params.get("num_train_epochs", 100))}")

        global_step = 0
        first_epoch = 0

        progress_bar = tqdm(range(global_step, self.config.training_params.get("max_train_steps", len(dataloader) * self.config.training_params.get("num_train_epochs", 100))), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(first_epoch, self.config.training_params.get("num_train_epochs", 100)):
            unet.train()
            text_encoder.train()
            for step, batch in enumerate(dataloader):
                with self.accelerator.accumulate(unet, text_encoder):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    self.accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.accelerator.log({"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

                    if global_step % self.config.training_params.get("save_steps", 500) == 0:
                        if self.accelerator.is_main_process:
                            self._save_lora_model(epoch)

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= self.config.training_params.get("max_train_steps", len(dataloader) * self.config.training_params.get("num_train_epochs", 100)):
                    break

            if self.accelerator.is_main_process:
                 if self.config.training_params.get("save_each_epoch", False):
                     self._save_lora_model(epoch)

        # Save the final model
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._save_lora_model(self.config.training_params.get("num_train_epochs", 100))

        self.accelerator.end_training()

if __name__ == "__main__":
    # Example Usage (replace with actual environment variable reading and config)
    # This part would typically be in a separate script or entry point
    # For demonstration, we'll use placeholder values and read from env vars
    import os

    input_folder = os.getenv("INPUT_IMAGE_FOLDER")
    output_folder = os.getenv("OUTPUT_LORA_FOLDER")
    base_model = os.getenv("BASE_MODEL_ID", "black-forest-labs/FLUX.1-schnell") # Allow overriding default

    if not input_folder:
        raise ValueError("INPUT_IMAGE_FOLDER environment variable not set.")
    if not output_folder:
        raise ValueError("OUTPUT_LORA_FOLDER environment variable not set.")

    # Example training parameters - these would ideally be configurable via CLI args or a config file
    training_config = {
        "num_train_epochs": 100,
        "train_batch_size": 1,
        "learning_rate": 1e-4,
        "resolution": 512,
        "seed": 42,
        "save_steps": 500,
        "save_each_epoch": False,
        "mixed_precision": "fp16", # or "bf16" or None
        "gradient_accumulation_steps": 1,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,
        "max_train_steps": None, # Set to None to train for num_train_epochs
        "center_crop": True,
        "dataloader_num_workers": 0,
        "enable_xformers_memory_efficient_attention": False, # Set to True if xformers is installed
    }

    # Determine offline mode from environment variable or default to True
    use_offline_mode = os.getenv("USE_OFFLINE_MODE", "true").lower() == "true"


    config = LoRATrainerConfig(
        base_model_id=base_model,
        input_image_folder=input_folder,
        output_lora_folder=output_folder,
        training_params=training_config,
        use_offline_mode=use_offline_mode,
    )

    trainer = LoRATrainer(config)
    trainer.train()
