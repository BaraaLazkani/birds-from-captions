"""Reference grids built from ground-truth data.

Used by the overfit gate: to judge whether the model has memorised its ten
training images you need those images beside the samples, decoded through the
same VAE so the comparison is not confounded by autoencoder loss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..models.encoders import decode_latents
from ..pipeline import make_grid


@torch.no_grad()
def reference_grid(
    dataset,
    vae,
    path: str | Path,
    columns: int = 4,
    device: str | torch.device = "cuda",
) -> Path:
    """Decode a dataset's cached latents into a contact sheet.

    Deliberately decodes the *cached latents* rather than reading the source
    JPEGs: it shows the ceiling the model is actually training toward, since no
    denoiser can beat the VAE's own reconstruction.
    """
    images: list[Image.Image] = []
    for index in range(len(dataset)):
        latent = dataset[index]["latent"].unsqueeze(0).to(device=device, dtype=vae.dtype)
        decoded = decode_latents(vae, latent)[0].float().cpu().numpy()
        images.append(Image.fromarray((decoded.transpose(1, 2, 0) * 255).astype(np.uint8)))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_grid(images, columns=columns).save(path)
    print(f"[eval] wrote reference grid ({len(images)} images) -> {path}")
    return path


def side_by_side(top: list[Image.Image], bottom: list[Image.Image], path: str | Path) -> Path:
    """Stack two equal-length image rows, for before/after comparisons."""
    if len(top) != len(bottom):
        raise ValueError(f"row lengths differ: {len(top)} vs {len(bottom)}")

    width, height = top[0].size
    canvas = Image.new("RGB", (len(top) * width, 2 * height), (255, 255, 255))
    for index, (upper, lower) in enumerate(zip(top, bottom)):
        canvas.paste(upper, (index * width, 0))
        canvas.paste(lower, (index * width, height))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path
