"""Training callbacks shared by both runs.

The important one is :class:`SampleGridCallback`.  For a generative model the
loss curve is close to uninformative -- flow-matching loss drifts down slowly
and correlates poorly with whether the pictures look like birds.  What actually
tells you the run is working is a grid of samples from the *same* prompts and
the *same* seed at every epoch, so the only thing changing is the model.
Logged as images, TensorBoard's step slider then plays it back as an animation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer

from ..pipeline import TextToImagePipeline, make_grid


class SampleGridCallback(Callback):
    """Generate a fixed prompt set each epoch and log it as an image.

    Args:
        prompts: Captions to render. Kept fixed for the whole run.
        every_n_epochs: Sampling costs a full denoising loop; every epoch is
            fine for short runs, raise it for long ones.
        steps: Sampler steps for the preview.
        guidance_scale: CFG strength for the preview.
        seed: Fixed, so successive grids are comparable frame to frame.
        out_dir: Also written to disk here, for the README.
        use_ema: Draw from the EMA weights when the module has them -- that is
            what the final model will be, so it is what should be monitored.
    """

    def __init__(
        self,
        prompts: list[str],
        every_n_epochs: int = 1,
        steps: int = 30,
        guidance_scale: float = 4.0,
        seed: int = 1234,
        out_dir: str | Path = "outputs/samples",
        use_ema: bool = True,
        columns: int = 4,
    ) -> None:
        super().__init__()
        self.prompts = list(prompts)
        self.every_n_epochs = max(1, every_n_epochs)
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.out_dir = Path(out_dir)
        self.use_ema = use_ema
        self.columns = columns
        self._pipeline: TextToImagePipeline | None = None

    def _pipe(self, module: LightningModule) -> TextToImagePipeline:
        if self._pipeline is None:
            self._pipeline = TextToImagePipeline(
                denoiser=module.denoiser,
                conditioner=module.conditioner,
                vae=module.get_vae(),
                device=module.device,
                dtype=torch.bfloat16,
            )
        # The denoiser may have moved between devices; keep the handle current.
        self._pipeline.denoiser = module.denoiser
        return self._pipeline

    @torch.no_grad()
    def _render(self, module: LightningModule) -> np.ndarray:
        pipeline = self._pipe(module)
        was_training = module.denoiser.training
        module.denoiser.eval()
        try:
            images = pipeline.generate(
                self.prompts, steps=self.steps, guidance_scale=self.guidance_scale, seed=self.seed
            )
        finally:
            if was_training:
                module.denoiser.train()
        grid = make_grid(images, columns=self.columns)
        return np.asarray(grid).transpose(2, 0, 1)  # CHW for TensorBoard

    def on_validation_epoch_end(self, trainer: Trainer, module: LightningModule) -> None:
        if trainer.sanity_checking or (trainer.current_epoch % self.every_n_epochs):
            return

        # Sampling runs the denoiser at double batch (classifier-free guidance)
        # and loads the VAE, on top of memory the training loop has already
        # reserved. On an 8 GB card that can tip over -- and losing an
        # overnight run because a *preview* failed would be absurd. Release the
        # cache first, and treat any failure here as non-fatal.
        torch.cuda.empty_cache()
        try:
            ema = getattr(module, "ema", None)
            if self.use_ema and ema is not None:
                with ema.averaged(module.denoiser):
                    grid = self._render(module)
            else:
                grid = self._render(module)
        except torch.OutOfMemoryError:
            print("\n[samples] skipped this epoch: out of memory during sampling (training continues)")
            torch.cuda.empty_cache()
            return
        except Exception as exc:  # noqa: BLE001 - previews must never kill a run
            print(f"\n[samples] skipped this epoch: {type(exc).__name__}: {exc} (training continues)")
            return
        finally:
            torch.cuda.empty_cache()

        path = self.out_dir / f"epoch_{trainer.current_epoch:04d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.fromarray(grid.transpose(1, 2, 0)).save(path)

        for logger in trainer.loggers:
            experiment = getattr(logger, "experiment", None)
            if hasattr(experiment, "add_image"):  # TensorBoard; CSVLogger has none
                experiment.add_image("samples", grid, global_step=trainer.global_step)

        print(f"\n[samples] epoch {trainer.current_epoch}: wrote {path}")


class EMACallback(Callback):
    """Advance the module's EMA after every optimiser step."""

    def on_train_batch_end(self, trainer: Trainer, module: LightningModule, *args) -> None:
        ema = getattr(module, "ema", None)
        if ema is not None:
            ema.update(module.denoiser)
            module.log("ema_decay", ema.current_decay(), on_step=True, on_epoch=False, prog_bar=False)
