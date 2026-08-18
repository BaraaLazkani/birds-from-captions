#!/usr/bin/env python
"""Train either model.

    python scripts/train.py --config-name train_dit
    python scripts/train.py --config-name train_lora
    python scripts/train.py --config-name train_dit overfit=10

Both tasks share this entrypoint because they share everything that matters --
the dataset, the text encoder, the sample-grid callback and the loggers.  Only
the LightningModule differs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ttimg.data.datamodule import LatentDataModule  # noqa: E402
from ttimg.training.callbacks import EMACallback, SampleGridCallback  # noqa: E402

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def build_module(cfg: DictConfig):
    """Construct the LightningModule for the requested task."""
    optim = cfg.optim

    if cfg.task == "dit":
        from ttimg.training.dit_module import DiTTrainingModule

        model_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.model).items()
                        if k not in ("name", "_target_")}
        return DiTTrainingModule(
            model=model_kwargs,
            flow={k: v for k, v in OmegaConf.to_container(cfg.diffusion).items() if k != "name"},
            learning_rate=optim.learning_rate,
            weight_decay=optim.weight_decay,
            warmup_steps=optim.warmup_steps,
            max_steps=cfg.max_steps,
            caption_dropout=optim.caption_dropout,
            ema_decay=optim.ema_decay,
            ema_warmup=optim.ema_warmup,
        )

    if cfg.task == "lora":
        from ttimg.training.lora_module import LoRATrainingModule

        return LoRATrainingModule(
            rank=cfg.model.rank,
            alpha=cfg.model.alpha,
            dropout=cfg.model.dropout,
            learning_rate=optim.learning_rate,
            weight_decay=optim.weight_decay,
            warmup_steps=optim.warmup_steps,
            max_steps=cfg.max_steps,
            caption_dropout=optim.caption_dropout,
            gradient_checkpointing=cfg.model.gradient_checkpointing,
            latent_size=cfg.model.latent_size,
            unet_dtype=DTYPES[cfg.model.get("unet_dtype", "bfloat16")],
        )

    raise ValueError(f"unknown task {cfg.task!r}; expected 'dit' or 'lora'")


def build_loggers(cfg: DictConfig) -> list:
    """TensorBoard and/or CSV -- both local, nothing phones home."""
    loggers = []
    if cfg.logger.tensorboard:
        loggers.append(TensorBoardLogger(cfg.logger.save_dir, name=cfg.run_name, default_hp_metric=False))
    if cfg.logger.csv:
        loggers.append(CSVLogger(cfg.logger.save_dir, name=cfg.run_name))
    return loggers


@hydra.main(version_base=None, config_path="../configs", config_name="train_dit")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")  # enable TF32 paths on Blackwell

    datamodule = LatentDataModule(
        cache_dir=cfg.data.cache_dir,
        resolution=cfg.data.resolution,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor,
        random_flip=cfg.data.random_flip,
        random_caption=cfg.data.random_caption,
        overfit_records=cfg.overfit,
        overfit_repeat=cfg.get("overfit_repeat", 100),
    )

    module = build_module(cfg)

    # In overfit mode the useful preview is the captions of the very images
    # being memorised -- then the sample grid can be compared directly against
    # the originals, which is the whole point of the gate.
    sample_prompts = list(cfg.sampling.prompts)
    if cfg.overfit:
        datamodule.setup("fit")
        sample_prompts = [record["captions"][0] for record in datamodule.train_set.records]
        print(f"[train] overfit gate: sampling the {len(sample_prompts)} memorised captions")

        # The targets, decoded through the same VAE -- this is the ceiling the
        # model is training toward, and what the sample grids get compared to.
        from ttimg.eval.grids import reference_grid
        from ttimg.models.encoders import get_shared_vae

        # Explicit device: the module has not been moved to the accelerator yet.
        device = "cuda" if cfg.accelerator == "gpu" else "cpu"
        reference_grid(
            # val_set, not train_set: the training copy repeats each record to
            # pack epochs, and iterating it would render the same 10 birds 100
            # times over.
            datamodule.val_set,
            get_shared_vae(dtype=torch.bfloat16, device=device),
            Path("outputs/samples") / cfg.run_name / "reference.png",
            columns=cfg.sampling.columns,
            device=device,
        )

    run_dir = Path(cfg.checkpoint.dir) / cfg.run_name
    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir,
            # The metric is logged as "val/loss"; referencing "val_loss" here
            # silently formats as 0.0000 (ranking still uses the real value).
            filename="{epoch:04d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=cfg.checkpoint.save_top_k,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        SampleGridCallback(
            prompts=sample_prompts,
            every_n_epochs=cfg.sampling.every_n_epochs,
            steps=cfg.sampling.steps,
            guidance_scale=cfg.sampling.guidance_scale,
            seed=cfg.sampling.seed,
            columns=cfg.sampling.columns,
            out_dir=Path("outputs/samples") / cfg.run_name,
            use_ema=cfg.task == "dit",
        ),
    ]
    if cfg.task == "dit":
        callbacks.append(EMACallback())

    trainer = Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        max_epochs=cfg.max_epochs,
        max_steps=cfg.max_steps,
        gradient_clip_val=cfg.gradient_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        val_check_interval=cfg.val_check_interval,
        logger=build_loggers(cfg),
        callbacks=callbacks,
        log_every_n_steps=25,
    )

    # Save the fully resolved config next to the checkpoints, so any run can be
    # reproduced from its own directory without guessing at overrides.
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    trainer.fit(module, datamodule=datamodule)

    if cfg.task == "lora":
        module.denoiser.save_lora(run_dir / "lora.safetensors")

    print(f"\nDone. Checkpoints in {run_dir}\n  tensorboard --logdir {cfg.logger.save_dir}")


if __name__ == "__main__":
    main()
