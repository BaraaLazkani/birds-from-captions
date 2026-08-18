#!/usr/bin/env python
"""Download CUB-200-2011 and build the shared VAE latent caches.

    python scripts/prepare_data.py
    python scripts/prepare_data.py resolutions=[256] overwrite=true

Run this once before any training.  It ends with a reconstruction check, which
is the gate for the whole project: if a cached latent does not decode back into
a recognisable bird, nothing downstream can possibly work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ttimg.data.prepare import build_latent_cache  # noqa: E402

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@hydra.main(version_base=None, config_path="../configs", config_name="prepare")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    data = cfg.data

    for resolution in cfg.resolutions:
        print(f"\n{'=' * 70}\n  Building latent cache at {resolution}px\n{'=' * 70}")
        build_latent_cache(
            data_root=data.root,
            out_dir=data.cache_dir,
            resolution=int(resolution),
            batch_size=cfg.vae_batch_size,
            object_ratio=data.object_ratio,
            split_mode=data.split_mode,
            val_fraction=data.val_fraction,
            n_val_classes=data.n_val_classes,
            seed=data.seed,
            device=cfg.device,
            overwrite=cfg.overwrite,
            num_workers=cfg.loader_workers,
            vae_dtype=DTYPES[cfg.vae_dtype],
        )

    print(f"\n{'=' * 70}\n  Verifying reconstruction quality\n{'=' * 70}")
    from ttimg.data.verify import verify_latent_cache

    ok = True
    for resolution in cfg.resolutions:
        ok &= verify_latent_cache(
            data_root=data.root,
            cache_dir=data.cache_dir,
            resolution=int(resolution),
            object_ratio=data.object_ratio,
            device=cfg.device,
        )

    if not ok:
        raise SystemExit("latent cache failed verification -- do not proceed to training")
    print("\nLatent caches ready.")


if __name__ == "__main__":
    main()
