"""CLIP score: does the generated image actually match the caption?

Deliberately uses a *different* CLIP (ViT-B/16) from the ViT-L/14 encoder that
conditions the models.  Scoring with the same encoder that produced the
conditioning would let a model score well simply by satisfying that particular
encoder's quirks -- the metric would be grading its own homework.

CLIP score measures alignment only, not image quality: a model producing crisp
birds that ignore the caption and one producing blurry birds that follow it can
land in the same place.  Always read it next to the sample grids.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

CLIP_SCORE_REPO = "openai/clip-vit-base-patch16"


class CLIPScorer:
    """Cosine similarity between CLIP image and text embeddings."""

    def __init__(
        self,
        repo: str = CLIP_SCORE_REPO,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(repo)
        self.model = CLIPModel.from_pretrained(repo, torch_dtype=dtype)
        self.model.requires_grad_(False).eval().to(device)
        self.device = torch.device(device)

    @torch.no_grad()
    def score(self, images: list[Image.Image], prompts: list[str]) -> torch.Tensor:
        """Per-pair similarity in [-1, 1]; typical good values are 0.25-0.35."""
        if len(images) != len(prompts):
            raise ValueError(f"got {len(images)} images and {len(prompts)} prompts")

        inputs = self.processor(
            text=prompts, images=images, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = self.model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
        )

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return (image_features * text_features).sum(dim=-1).float().cpu()

    @torch.no_grad()
    def score_pipeline(
        self,
        pipeline,
        prompts: list[str],
        steps: int = 30,
        guidance_scale: float = 4.0,
        seed: int = 0,
        batch_size: int = 8,
    ) -> dict[str, float]:
        """Generate for each prompt and score. Returns mean/std/min/max."""
        scores: list[float] = []
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            images = pipeline.generate(
                chunk, steps=steps, guidance_scale=guidance_scale, seed=seed + start
            )
            scores.extend(self.score(images, chunk).tolist())

        array = np.array(scores)
        return {
            "clip_score_mean": float(array.mean()),
            "clip_score_std": float(array.std()),
            "clip_score_min": float(array.min()),
            "clip_score_max": float(array.max()),
            "n_prompts": len(prompts),
        }
