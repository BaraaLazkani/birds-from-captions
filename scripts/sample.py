#!/usr/bin/env python
"""Generate images from captions.

    python scripts/sample.py prompt="a small blue bird with a white belly"
    python scripts/sample.py model=dit checkpoint=checkpoints/dit_small/last.ckpt
    python scripts/sample.py mode=sweep_guidance
    python scripts/sample.py mode=sweep_attribute

Beyond plain generation this offers the two diagnostics that matter for a
from-scratch model: a guidance sweep (is the CFG scale in a sane range?) and an
attribute sweep (does changing one word in the caption change the right thing?).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ttimg.models.registry import build_pipeline  # noqa: E402
from ttimg.pipeline import save_grid  # noqa: E402


def _stamped(out_dir: Path, name: str) -> Path:
    return out_dir / f"{datetime.now():%Y%m%d-%H%M%S}_{name}.png"


def run_single(pipeline, cfg: DictConfig, out_dir: Path) -> Path:
    prompts = [cfg.prompt] * cfg.n_images
    images = pipeline.generate(
        prompts, steps=cfg.steps, guidance_scale=cfg.guidance_scale, seed=cfg.seed
    )
    path = _stamped(out_dir, f"{cfg.model}_single")
    save_grid(images, path, columns=min(4, len(images)))
    print(f"\n  prompt: {cfg.prompt}\n  -> {path}")
    return path


def run_guidance_sweep(pipeline, cfg: DictConfig, out_dir: Path) -> Path:
    """Same prompt and seed at increasing CFG scales.

    Low guidance gives washed-out, generic samples; too high oversaturates and
    collapses diversity. The usable window differs between the two models, so it
    is worth measuring rather than assuming.
    """
    images = []
    for scale in cfg.guidance_values:
        image = pipeline.generate(
            [cfg.prompt], steps=cfg.steps, guidance_scale=float(scale), seed=cfg.seed
        )[0]
        images.append(image)
        print(f"  cfg={float(scale):4.1f} done")

    path = _stamped(out_dir, f"{cfg.model_kind}_guidance_sweep")
    save_grid(images, path, columns=len(images))
    print(f"\n  guidance {list(cfg.guidance_values)} (left to right)\n  -> {path}")
    return path


def run_attribute_sweep(pipeline, cfg: DictConfig, out_dir: Path) -> Path:
    """Minimal caption pairs at a fixed seed.

    This is the sharpest test CUB affords: because the two prompts share every
    word but one, and the starting noise is identical, any difference in the
    output is attributable to that word.
    """
    images = []
    for left, right in cfg.attribute_pairs:
        pair = pipeline.generate(
            [left, right], steps=cfg.steps, guidance_scale=cfg.guidance_scale, seed=cfg.seed
        )
        images.extend(pair)
        print(f"  {left!r}\n  {right!r}\n")

    path = _stamped(out_dir, f"{cfg.model_kind}_attribute_sweep")
    save_grid(images, path, columns=2)
    print(f"  each row is one minimal pair\n  -> {path}")
    return path


@hydra.main(version_base=None, config_path="../configs", config_name="sample")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(
        kind=cfg.model_kind,
        checkpoint=cfg.checkpoint,
        lora_path=cfg.lora_path,
        device=cfg.device,
        dtype=torch.bfloat16,
    )
    print(f"\n{pipeline.denoiser.describe()}\n")

    runners = {
        "single": run_single,
        "sweep_guidance": run_guidance_sweep,
        "sweep_attribute": run_attribute_sweep,
    }
    if cfg.mode not in runners:
        raise SystemExit(f"unknown mode {cfg.mode!r}; expected one of {sorted(runners)}")

    runners[cfg.mode](pipeline, cfg, out_dir)


if __name__ == "__main__":
    main()
