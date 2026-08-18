#!/usr/bin/env python
"""Render the same prompts through every trained model, side by side.

    python scripts/compare_models.py
    python scripts/compare_models.py --guidance 3.0 --steps 40

This is the payoff of the shared-latent-space design: because the text encoder,
the VAE, the sampler and the starting noise are all identical across models, the
only thing that differs between columns is the denoiser itself. Any other setup
would be comparing pipelines, not models.

Also scores each model with CLIP (ViT-B/16 -- deliberately a different CLIP from
the ViT-L/14 that conditions the models, so nothing grades its own homework).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ttimg.models.encoders import TextConditioner  # noqa: E402
from ttimg.models.registry import MODEL_LABELS, build_pipeline  # noqa: E402

DEFAULT_PROMPTS = [
    "this bird has a bright red belly with black wings and a short pointed beak",
    "a small yellow bird with a grey head and a thin sharp bill",
    "a large white seabird with long narrow wings and a hooked orange beak",
    "this small bird has an iridescent green throat and a very long thin bill",
    "a brown speckled bird with a white throat and striped wingbars",
    "this bird is completely black with a long tail and a stout bill",
]

LABEL_BAND = 30


def available_models(dit_ckpt: Path, lora_path: Path) -> list[tuple[str, dict]]:
    models: list[tuple[str, dict]] = [("sd15", {})]
    if lora_path.exists():
        models.append(("sd15_lora", {"lora_path": lora_path}))
    if dit_ckpt.exists():
        models.append(("dit", {"checkpoint": dit_ckpt}))
    return models


def label_row(images: list[Image.Image], text: str, width: int) -> Image.Image:
    """One model's outputs in a row, with a caption band above."""
    size = images[0].size[0]
    row = Image.new("RGB", (width, size + LABEL_BAND), "white")
    draw = ImageDraw.Draw(row)
    draw.rectangle([0, 0, width, LABEL_BAND], fill=(25, 25, 30))
    draw.text((10, 9), text, fill=(255, 255, 255))
    for i, im in enumerate(images):
        row.paste(im.resize((size, size)), (i * size, LABEL_BAND))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit-checkpoint", type=Path, default=Path("checkpoints/dit_small/last.ckpt"))
    parser.add_argument("--lora", type=Path, default=Path("checkpoints/sd15_lora_cub/lora.safetensors"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=Path("outputs/final/model_comparison.png"))
    parser.add_argument("--score", action="store_true", help="also compute CLIP scores")
    parser.add_argument("--score-prompts", type=int, default=24)
    args = parser.parse_args()

    models = available_models(args.dit_checkpoint, args.lora)
    print(f"comparing {len(models)} model(s): {[m for m, _ in models]}")

    # Shared across every pipeline, so switching models does not reload it.
    conditioner = TextConditioner(device="cuda", dtype=torch.bfloat16)

    scorer = None
    score_prompts: list[str] = []
    if args.score:
        from ttimg.data.dataset import LatentCaptionDataset
        from ttimg.eval.clip_score import CLIPScorer

        val = LatentCaptionDataset("datasets/cache", 256, split="val", random_caption=False)
        score_prompts = val.fixed_prompts(args.score_prompts, seed=7)
        scorer = CLIPScorer(device="cuda")
        print(f"scoring against {len(score_prompts)} held-out val captions")

    size = None
    rows: list[Image.Image] = []
    results: dict[str, dict] = {}

    for kind, kwargs in models:
        print(f"\n--- {MODEL_LABELS[kind]} ---")
        pipeline = build_pipeline(
            kind=kind, device="cuda", dtype=torch.bfloat16, conditioner=conditioner, **kwargs
        )

        images = pipeline.generate(
            DEFAULT_PROMPTS, steps=args.steps, guidance_scale=args.guidance, seed=args.seed
        )
        # Normalise to the first model's resolution so rows line up (SD renders
        # at 512, the DiT at 256).
        size = size or images[0].size[0]
        images = [im.resize((size, size)) for im in images]
        rows.append(label_row(images, MODEL_LABELS[kind], size * len(images)))

        entry: dict = {"label": MODEL_LABELS[kind]}
        if scorer is not None:
            stats = scorer.score_pipeline(
                pipeline, score_prompts, steps=args.steps,
                guidance_scale=args.guidance, seed=args.seed,
            )
            entry.update(stats)
            print(f"  CLIP score: {stats['clip_score_mean']:.4f} +/- {stats['clip_score_std']:.4f}")
        results[kind] = entry

        del pipeline
        torch.cuda.empty_cache()

    canvas = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"\nwrote {args.out}")

    if results:
        summary = args.out.with_suffix(".json")
        summary.write_text(json.dumps(
            {"guidance": args.guidance, "steps": args.steps, "seed": args.seed,
             "prompts": DEFAULT_PROMPTS, "models": results}, indent=2))
        print(f"wrote {summary}")


if __name__ == "__main__":
    main()
