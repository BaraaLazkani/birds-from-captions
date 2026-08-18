"""Gate on the latent cache before any training starts.

Every model in this project learns to produce latents, and is judged only after
those latents pass through the VAE decoder.  If the cache is wrong -- bad
scaling factor, mismatched crop, transposed flip index -- training will still
produce a falling loss curve while generating garbage, and the cause is very
hard to find from that side.  So we check it here, where it is cheap and
unambiguous.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..models.encoders import decode_latents, load_vae
from .cub import build_index, crop_box
from .prepare import _load_and_crop

MIN_ACCEPTABLE_PSNR = 20.0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio between two [0, 1] arrays, in dB."""
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10.0 * float(np.log10(1.0 / mse))


@torch.no_grad()
def verify_latent_cache(
    data_root: str | Path,
    cache_dir: str | Path,
    resolution: int = 256,
    object_ratio: float = 0.75,
    n_samples: int = 8,
    device: str = "cuda",
    out_dir: str | Path = "outputs/verify",
    seed: int = 0,
) -> bool:
    """Decode cached latents and compare against the source images.

    Checks two things:

    1. **Reconstruction fidelity.**  Decoded latents should match the original
       crop closely; the VAE is lossy but not by much.  Failure here usually
       means the scaling factor or the crop pipeline disagrees between encode
       and decode.
    2. **Flip bookkeeping.**  Latent index 1 must decode to the mirror of index
       0.  This catches an off-by-one in the cache layout that would otherwise
       show up only as a model that never learns left/right structure.

    Writes a side-by-side PNG so the result can be judged by eye too, and
    returns whether the cache passed.
    """
    data_root, cache_dir, out_dir = Path(data_root), Path(cache_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((cache_dir / f"index_{resolution}.json").read_text())
    latents = np.load(cache_dir / f"latents_{resolution}.npy", mmap_mode="r")
    records = {r.image_id: r for r in build_index(data_root)}

    rng = random.Random(seed)
    chosen = rng.sample(meta["records"], min(n_samples, len(meta["records"])))

    vae = load_vae(dtype=torch.float32, device=device)

    originals, reconstructions, scores, flip_scores = [], [], [], []

    for entry in chosen:
        record = records[entry["image_id"]]
        original = _load_and_crop(record, resolution, object_ratio)  # [-1,1] CHW
        original01 = (original + 1.0) / 2.0

        cached = torch.from_numpy(np.asarray(latents[entry["row"]], dtype=np.float32)).to(device)
        decoded = decode_latents(vae, cached)  # [2,3,H,W] in [0,1]

        recon = decoded[0].cpu().numpy()
        scores.append(psnr(original01, recon))

        # The flipped entry should reconstruct the mirrored image.
        flipped = decoded[1].cpu().numpy()
        flip_scores.append(psnr(original01[:, :, ::-1].copy(), flipped))

        originals.append(original01)
        reconstructions.append(recon)

    mean_psnr = float(np.mean(scores))
    mean_flip_psnr = float(np.mean(flip_scores))

    grid = np.concatenate(
        [np.concatenate(originals, axis=2), np.concatenate(reconstructions, axis=2)], axis=1
    )
    grid_path = out_dir / f"reconstruction_{resolution}.png"
    Image.fromarray((grid.transpose(1, 2, 0) * 255).astype(np.uint8)).save(grid_path)

    passed = mean_psnr >= MIN_ACCEPTABLE_PSNR and mean_flip_psnr >= MIN_ACCEPTABLE_PSNR
    status = "PASS" if passed else "FAIL"

    print(f"[verify] {resolution}px  reconstruction PSNR {mean_psnr:5.2f} dB (min per-sample {min(scores):.2f})")
    print(f"[verify] {resolution}px  h-flip alignment PSNR {mean_flip_psnr:5.2f} dB")
    print(f"[verify] {resolution}px  top row = original, bottom row = decoded -> {grid_path}")
    print(f"[verify] {resolution}px  {status} (threshold {MIN_ACCEPTABLE_PSNR} dB)")

    return passed
