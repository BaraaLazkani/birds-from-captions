"""Caption in, image out -- for whichever denoiser you hand it.

This is the layer the CLI, the evaluation grids and the Gradio app all talk to.
It holds the three pieces (text encoder, denoiser, VAE) and knows nothing about
which denoiser it has: a from-scratch DiT and a fine-tuned SD UNet are used
through exactly the same calls, which is what makes the side-by-side comparison
in the GUI an honest one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .diffusion.sampler import sample
from .models.encoders import TextConditioner, decode_latents, load_vae
from .models.interface import Denoiser


@dataclass
class GenerationSettings:
    steps: int = 30
    guidance_scale: float = 4.0
    seed: int | None = None


class TextToImagePipeline:
    """Text encoder + denoiser + VAE decoder.

    Args:
        denoiser: Any :class:`Denoiser`.
        conditioner: Shared CLIP text encoder. Created if omitted.
        vae: Shared autoencoder. Created if omitted.
        device: Where to run.
        dtype: Compute dtype for sampling. bfloat16 is the default because SD's
            VAE overflows in fp16, and Blackwell runs bf16 at full speed.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        conditioner: TextConditioner | None = None,
        vae=None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.denoiser = denoiser.to(self.device).eval()
        self.conditioner = conditioner or TextConditioner(device=device, dtype=dtype)
        self.vae = vae if vae is not None else load_vae(dtype=dtype, device=device)

    @torch.no_grad()
    def generate_latents(
        self,
        prompts: list[str],
        settings: GenerationSettings | None = None,
        latent_size: int | None = None,
    ) -> torch.Tensor:
        settings = settings or GenerationSettings()
        cond = self.conditioner.encode(prompts).to(self.dtype)
        uncond = self.conditioner.null_embedding(len(prompts)).to(self.dtype)

        return sample(
            self.denoiser,
            cond=cond,
            uncond=uncond if settings.guidance_scale != 1.0 else None,
            steps=settings.steps,
            guidance_scale=settings.guidance_scale,
            latent_size=latent_size,
            seed=settings.seed,
            device=self.device,
            dtype=self.dtype,
        )

    @torch.no_grad()
    def generate(
        self,
        prompts: str | list[str],
        steps: int = 30,
        guidance_scale: float = 4.0,
        seed: int | None = None,
        latent_size: int | None = None,
    ) -> list[Image.Image]:
        """Generate one image per prompt.

        A fixed ``seed`` gives every model the same starting noise, so
        differences between them are differences in the models rather than in
        the draw.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        latents = self.generate_latents(
            prompts, GenerationSettings(steps, guidance_scale, seed), latent_size
        )
        images = decode_latents(self.vae, latents).float().cpu().numpy()
        return [Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8)) for img in images]

    def to(self, device: str | torch.device) -> TextToImagePipeline:
        self.device = torch.device(device)
        self.denoiser.to(device)
        self.vae.to(device)
        self.conditioner.to(device)
        return self


def make_grid(images: list[Image.Image], columns: int | None = None, padding: int = 4) -> Image.Image:
    """Tile images into a single contact sheet, for logging and README figures."""
    if not images:
        raise ValueError("no images to grid")

    columns = columns or min(len(images), 4)
    rows = (len(images) + columns - 1) // columns
    width, height = images[0].size

    canvas = Image.new(
        "RGB",
        (columns * width + (columns + 1) * padding, rows * height + (rows + 1) * padding),
        (255, 255, 255),
    )
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        canvas.paste(
            image, (padding + column * (width + padding), padding + row * (height + padding))
        )
    return canvas


def save_grid(images: list[Image.Image], path: str | Path, columns: int | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_grid(images, columns).save(path)
    return path
