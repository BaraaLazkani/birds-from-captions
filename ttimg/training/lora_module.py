"""Lightning module for the SD 1.5 LoRA fine-tune.

Deliberately mirrors :mod:`dit_module` -- same batch format, same frozen text
encoder, same cached latents.  The only real differences are the objective
(epsilon-prediction against SD's DDPM schedule, because that is what the
pretrained weights expect) and that ~99% of the parameters stay frozen.

Training on the precomputed 512px latents is what makes this fit: the VAE never
enters VRAM, so the UNet has the card almost to itself.
"""

from __future__ import annotations

import math

import torch
from lightning.pytorch import LightningModule

from ..diffusion.sampler import SD_TRAIN_TIMESTEPS, sd_alphas_cumprod
from ..models.encoders import TextConditioner, get_shared_vae
from ..models.sd15 import SD15UNet


class LoRATrainingModule(LightningModule):
    """Fine-tunes Stable Diffusion 1.5 on CUB with low-rank adapters.

    Args:
        rank: LoRA rank. 16 is a good default for a style/domain shift.
        alpha: LoRA scaling; defaults to ``rank``.
        learning_rate: 1e-4 is the usual starting point for LoRA -- much higher
            than full fine-tuning, because the adapters start at zero.
        caption_dropout: Null-embedding rate, keeping CFG well-calibrated.
        gradient_checkpointing: Trades ~30% speed for a large VRAM saving.
    """

    def __init__(
        self,
        rank: int = 16,
        alpha: int | None = None,
        dropout: float = 0.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 200,
        max_steps: int = 20_000,
        caption_dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        unet_dtype: torch.dtype = torch.float16,
        text_dtype: torch.dtype = torch.float16,
        latent_size: int = 64,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["unet_dtype", "text_dtype"])

        self.denoiser = SD15UNet(dtype=unet_dtype, device="cpu", latent_size=latent_size, trainable=True)
        self.denoiser.add_lora(rank=rank, alpha=alpha, dropout=dropout)
        if gradient_checkpointing:
            self.denoiser.enable_gradient_checkpointing()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps_ = max_steps
        self.caption_dropout = caption_dropout
        self.text_dtype = text_dtype

        self._conditioner: TextConditioner | None = None
        self.register_buffer("alphas_cumprod", sd_alphas_cumprod("cpu"), persistent=False)

        print(self.denoiser.describe())

    @property
    def conditioner(self) -> TextConditioner:
        if self._conditioner is None:
            self._conditioner = TextConditioner(device=self.device, dtype=self.text_dtype)
        return self._conditioner

    def get_vae(self):
        return get_shared_vae(dtype=torch.bfloat16, device=self.device)

    # -- training -------------------------------------------------------------

    def _step(self, batch: dict, dropout: float) -> torch.Tensor:
        latents = batch["latent"]
        batch_size = latents.shape[0]

        cond = self.conditioner.encode_with_dropout(batch["caption"], dropout)

        # SD's forward process: uniform discrete timesteps, as in its own training.
        indices = torch.randint(0, SD_TRAIN_TIMESTEPS, (batch_size,), device=latents.device)
        alpha = self.alphas_cumprod[indices].view(-1, 1, 1, 1)
        noise = torch.randn_like(latents)
        noisy = alpha.sqrt() * latents + (1 - alpha).sqrt() * noise

        # The adapter converts continuous t back to SD's discrete grid.
        t = indices.float() / (SD_TRAIN_TIMESTEPS - 1)
        prediction = self.denoiser(noisy, t, cond)
        return torch.nn.functional.mse_loss(prediction.float(), noise.float())

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._step(batch, self.caption_dropout)
        size = batch["latent"].shape[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=size)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._step(batch, dropout=0.0)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=batch["latent"].shape[0])
        return loss

    # -- optimisation ---------------------------------------------------------

    def configure_optimizers(self):
        params = [p for p in self.denoiser.parameters() if p.requires_grad]
        print(f"[lora] optimising {sum(p.numel() for p in params) / 1e6:.2f}M of "
              f"{self.denoiser.num_parameters / 1e6:.1f}M parameters")

        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, self.max_steps_ - self.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                "interval": "step",
            },
        }

    # -- checkpointing --------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Keep only the adapters.

        Lightning would otherwise write all 860M frozen UNet parameters into
        every checkpoint -- ~3.4 GB each, which this disk cannot absorb across a
        training run. The adapters alone are a few MB.
        """
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items() if "lora" in k.lower()
        }
