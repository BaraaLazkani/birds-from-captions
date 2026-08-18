"""Dataset over the precomputed latent cache.

Both training runs -- the LoRA fine-tune and the from-scratch DiT -- consume this
same class; only the resolution differs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentCaptionDataset(Dataset):
    """(latent, caption) pairs, with the two augmentations that survive caching.

    Args:
        cache_dir: Directory written by :func:`build_latent_cache`.
        resolution: Which cache to open (256 or 512).
        split: ``"train"``, ``"val"``, or ``"all"``.
        random_flip: Pick between the cached original and h-flipped latent.
        random_caption: Pick one of the image's 10 captions per access.  This is
            the project's main source of data diversity: with 10 captions each,
            CUB's 11,788 images become 117,880 distinct image-text pairs, which
            is what makes a transformer trainable on a dataset this small.
        max_records: Truncate the split -- used by the 10-image overfit test.
        repeat: Present the records this many times per epoch. Only useful with
            ``max_records``: a 10-image epoch is a single optimiser step, and
            Lightning's per-epoch overhead (dataloader restart, validation,
            callbacks) then costs ~2 s to wrap 20 ms of compute. Repeating packs
            real steps into each epoch and leaves the GPU actually busy.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        resolution: int = 256,
        split: str = "train",
        random_flip: bool = True,
        random_caption: bool = True,
        max_records: int | None = None,
        repeat: int = 1,
    ) -> None:
        cache_dir = Path(cache_dir)
        latent_path = cache_dir / f"latents_{resolution}.npy"
        index_path = cache_dir / f"index_{resolution}.json"

        if not latent_path.exists():
            raise FileNotFoundError(
                f"no latent cache at {latent_path}. Run: python scripts/prepare_data.py"
            )

        meta = json.loads(index_path.read_text())
        records = meta["records"]
        if split != "all":
            records = [r for r in records if r["split"] == split]

        if max_records is not None:
            records = records[:max_records]

        self.records = records
        self.latent_path = latent_path
        self.resolution = resolution
        self.latent_size = meta["latent_size"]
        self.random_flip = random_flip
        self.random_caption = random_caption
        self.repeat = max(1, repeat)

        # Opened lazily: a memmap handle created before fork does not survive
        # cleanly into DataLoader worker processes, so each worker makes its own.
        self._latents: np.ndarray | None = None

    def _array(self) -> np.ndarray:
        if self._latents is None:
            self._latents = np.load(self.latent_path, mmap_mode="r")
        return self._latents

    def __len__(self) -> int:
        return len(self.records) * self.repeat

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx % len(self.records)]
        flip = 1 if (self.random_flip and random.random() < 0.5) else 0

        latent = np.asarray(self._array()[record["row"], flip], dtype=np.float32)

        captions = record["captions"]
        caption = random.choice(captions) if self.random_caption else captions[0]

        return {
            "latent": torch.from_numpy(latent),
            "caption": caption,
            "class_id": record["class_id"],
        }

    def fixed_prompts(self, n: int = 8, seed: int = 0) -> list[str]:
        """A stable prompt sample for progress grids, independent of training."""
        rng = random.Random(seed)
        return [rng.choice(r["captions"]) for r in rng.sample(self.records, min(n, len(self.records)))]
