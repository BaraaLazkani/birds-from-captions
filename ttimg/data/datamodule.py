"""Lightning DataModule over the latent cache, shared by both training runs."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import LatentCaptionDataset


class LatentDataModule(LightningDataModule):
    """Serves cached latents and captions.

    Args:
        cache_dir: Directory produced by ``scripts/prepare_data.py``.
        resolution: 256 for the DiT, 512 for the SD LoRA.
        overfit_records: If set, train and validate on the same tiny subset.
            This is the correctness gate for the from-scratch model -- a
            diffusion model that cannot memorise ten images has a bug, and it is
            far cheaper to find that here than after a long run.
    """

    def __init__(
        self,
        cache_dir: str | Path = "./datasets/cache",
        resolution: int = 256,
        batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        random_flip: bool = True,
        random_caption: bool = True,
        overfit_records: int | None = None,
        overfit_repeat: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_set: LatentCaptionDataset | None = None
        self.val_set: LatentCaptionDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        hp = self.hparams
        common = dict(
            cache_dir=hp.cache_dir,
            resolution=hp.resolution,
            random_flip=hp.random_flip,
            random_caption=hp.random_caption,
        )

        if hp.overfit_records:
            # Same handful of images for both splits: the point is to watch the
            # model reproduce them, not to measure generalisation.
            self.train_set = LatentCaptionDataset(
                **{**common, "random_flip": False, "random_caption": False},
                split="train",
                max_records=hp.overfit_records,
                repeat=hp.overfit_repeat,
            )
            # Validate over the records once, not once per repeat.
            self.val_set = LatentCaptionDataset(
                **{**common, "random_flip": False, "random_caption": False},
                split="train",
                max_records=hp.overfit_records,
            )
        else:
            self.train_set = LatentCaptionDataset(**common, split="train")
            self.val_set = LatentCaptionDataset(
                **{**common, "random_flip": False, "random_caption": False}, split="val"
            )

        print(f"[data] {len(self.train_set)} train / {len(self.val_set)} val at {hp.resolution}px")

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        hp = self.hparams
        workers = hp.num_workers
        return DataLoader(
            dataset,
            batch_size=hp.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=hp.pin_memory,
            drop_last=shuffle,
            persistent_workers=workers > 0,
            prefetch_factor=hp.prefetch_factor if workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, shuffle=False)
