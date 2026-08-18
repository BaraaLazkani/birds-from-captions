"""Encode CUB images into a VAE latent cache, once.

The single most important optimisation in this project.  Running the VAE inside
the training loop would cost VRAM we do not have and repeat identical work every
epoch; encoding up front turns training into a stream of small tensors and lets
the 8 GB card spend all of its memory on the denoiser.

Layout written to ``<out_dir>/``:

    latents_256.npy   float16 [N, 2, 4, 32, 32]   -- dim 1 is [original, h-flip]
    index_256.json    per-record captions, class, split, row offset

Latents are stored as a numpy ``.npy`` and read back with ``mmap_mode="r"``.
That matters for a specific reason: DataLoader workers are separate processes,
so an in-RAM tensor would be duplicated per worker (8 workers x 760 MB at 512px
would exhaust this machine).  A memory-mapped file is backed by the shared OS
page cache instead, so every worker reads the same physical pages.

Both orientations are stored because horizontal flip is the one augmentation
that survives precomputation -- you cannot flip a latent and get the latent of
the flipped image, so it has to be encoded twice, here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..models.encoders import LATENT_CHANNELS, VAE_DOWNSCALE, encode_images, load_vae
from .cub import CubRecord, build_index, crop_box, split_records


class _CropDataset(torch.utils.data.Dataset):
    """Loads and crops images in DataLoader workers.

    Decoding a JPEG and bicubic-resizing it costs about as much as the VAE
    forward pass itself, and it is pure single-threaded CPU work.  Pushing it
    into worker processes overlaps it with GPU time and roughly halves wall
    clock for the whole precompute.
    """

    def __init__(self, records: list[CubRecord], resolution: int, object_ratio: float) -> None:
        self.records = records
        self.resolution = resolution
        self.object_ratio = object_ratio

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(_load_and_crop(self.records[idx], self.resolution, self.object_ratio))


def _encode_pair(vae, batch: torch.Tensor) -> torch.Tensor:
    """Encode a pixel batch and its mirror -> [B, 2, C, h, w] fp16 on CPU."""
    original = encode_images(vae, batch)
    flipped = encode_images(vae, torch.flip(batch, dims=[3]))
    return torch.stack([original, flipped], dim=1).to(torch.float16).cpu()


def _encode_adaptive(vae, batch: torch.Tensor, max_chunk: int) -> tuple[torch.Tensor, int]:
    """Encode, halving the chunk size until it fits in VRAM.

    The VAE encoder's activations scale with resolution squared: a batch that is
    comfortable at 256px needs 4x the memory at 512px and will OOM on an 8 GB
    card.  Rather than hard-code a per-resolution batch size that would break on
    different hardware, back off on failure and report the size that worked so
    subsequent batches skip straight to it.
    """
    chunk = max(1, max_chunk)
    while True:
        try:
            return torch.cat([_encode_pair(vae, part) for part in batch.split(chunk)]), chunk
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk == 1:
                raise
            chunk = max(1, chunk // 2)
            print(f"\n[prepare] VRAM tight -- retrying with VAE chunk size {chunk}")


def _load_and_crop(record: CubRecord, resolution: int, object_ratio: float) -> np.ndarray:
    """One image -> float32 CHW array in [-1, 1], square and bird-centred."""
    with Image.open(record.path) as image:
        image = image.convert("RGB")  # a handful of CUB files are greyscale
        image = image.crop(crop_box(record.bbox, *image.size, object_ratio=object_ratio))
        image = image.resize((resolution, resolution), Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return array.transpose(2, 0, 1)


@torch.no_grad()
def build_latent_cache(
    data_root: str | Path,
    out_dir: str | Path,
    resolution: int = 256,
    batch_size: int = 16,
    object_ratio: float = 0.75,
    split_mode: str = "by_image",
    val_fraction: float = 0.05,
    n_val_classes: int = 50,
    seed: int = 42,
    device: str = "cuda",
    overwrite: bool = False,
    num_workers: int = 8,
    vae_dtype: torch.dtype = torch.bfloat16,
) -> Path:
    """Encode the whole dataset at one resolution.  Returns the ``.npy`` path."""
    data_root, out_dir = Path(data_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latent_path = out_dir / f"latents_{resolution}.npy"
    index_path = out_dir / f"index_{resolution}.json"
    if latent_path.exists() and index_path.exists() and not overwrite:
        print(f"[prepare] {latent_path.name} already built; pass overwrite=True to rebuild")
        return latent_path

    records = build_index(data_root)
    train, val = split_records(
        records, mode=split_mode, val_fraction=val_fraction, n_val_classes=n_val_classes, seed=seed
    )
    split_of = {id(r): "train" for r in train} | {id(r): "val" for r in val}

    latent_size = resolution // VAE_DOWNSCALE
    # bf16 rather than fp32: measured against fp32 on this dataset it costs
    # -0.04 dB of reconstruction PSNR (i.e. nothing -- the two decoded outputs
    # agree to 54 dB) while running 1.8x faster in half the VRAM. The precision
    # would be discarded anyway, since latents are stored as fp16.
    vae = load_vae(dtype=vae_dtype, device=device)

    # Allocate on disk and fill incrementally -- the full 512px cache is ~760 MB
    # and this machine has 15 GB of RAM shared with everything else.
    array = np.lib.format.open_memmap(
        latent_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(records), 2, LATENT_CHANNELS, latent_size, latent_size),
    )

    index: list[dict] = []
    vae_chunk = batch_size
    print(
        f"[prepare] encoding {len(records)} images at {resolution}px "
        f"-> {latent_size}x{latent_size} latents ({vae_dtype}, {num_workers} loader workers)"
    )

    loader = torch.utils.data.DataLoader(
        _CropDataset(records, resolution, object_ratio),
        batch_size=batch_size,
        shuffle=False,  # row order must match the index we build below
        num_workers=num_workers,
        pin_memory=True,
    )

    started = time.time()
    start = 0

    for batch in loader:
        chunk = records[start : start + batch.shape[0]]
        stacked, vae_chunk = _encode_adaptive(vae, batch.to(device, non_blocking=True), vae_chunk)
        array[start : start + len(chunk)] = stacked.numpy()

        for offset, record in enumerate(chunk):
            index.append(
                {
                    "row": start + offset,
                    "image_id": record.image_id,
                    "class_id": record.class_id,
                    "class_name": record.class_name,
                    "captions": record.captions,
                    "split": split_of[id(record)],
                }
            )

        start += len(chunk)
        if start % (batch_size * 20) < batch_size:
            rate = start / max(time.time() - started, 1e-6)
            eta = (len(records) - start) / max(rate, 1e-6)
            print(
                f"\r[prepare] {start}/{len(records)} ({100 * start / len(records):.1f}%) "
                f"{rate:.1f} img/s  ETA {eta / 60:.1f} min",
                end="",
                flush=True,
            )

    array.flush()
    del array
    print(f"\r[prepare] {len(records)}/{len(records)} (100.0%)")

    index_path.write_text(
        json.dumps(
            {
                "resolution": resolution,
                "latent_size": latent_size,
                "latent_channels": LATENT_CHANNELS,
                "object_ratio": object_ratio,
                "split_mode": split_mode,
                "seed": seed,
                "records": index,
            }
        )
    )

    size_mb = latent_path.stat().st_size / 1e6
    print(f"[prepare] wrote {latent_path.name} ({size_mb:.1f} MB) and {index_path.name}")
    return latent_path
