"""Lightning module for the from-scratch DiT.

The training step is short, and deliberately so -- all the expensive machinery
(VAE encoding, dataset construction) happened during preparation, so what
remains is: read cached latents, embed the caption, corrupt along a straight
path, predict the velocity, take the MSE.
"""

from __future__ import annotations

import math

import torch
from lightning.pytorch import LightningModule

from ..diffusion.flow import FlowMatching
from ..models.dit import ScratchDiT
from ..models.encoders import TextConditioner, get_shared_vae
from .ema import EMA


class DiTTrainingModule(LightningModule):
    """Trains :class:`ScratchDiT` with a rectified-flow objective.

    Args:
        model: Kwargs for :class:`ScratchDiT`.
        flow: Kwargs for :class:`FlowMatching`.
        learning_rate: Peak LR after warmup.
        weight_decay: AdamW decay; not applied to norms/biases/embeddings.
        warmup_steps: Linear LR warmup. Transformers destabilise without it.
        max_steps: Horizon for the cosine decay.
        caption_dropout: Fraction of captions replaced by the null embedding,
            which is what makes classifier-free guidance possible at sample time.
        ema_decay: Decay for the weight average used when sampling.
    """

    def __init__(
        self,
        model: dict | None = None,
        flow: dict | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100_000,
        caption_dropout: float = 0.1,
        ema_decay: float = 0.9999,
        ema_warmup: int = 1000,
        text_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        # torch dtypes do not round-trip through YAML, and Lightning warns on
        # every checkpoint if they are in hparams.
        self.save_hyperparameters(ignore=["text_dtype"])

        self.denoiser = ScratchDiT(**(model or {}))
        self.flow = FlowMatching(**(flow or {}))
        self.caption_dropout = caption_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps_ = max_steps
        self.text_dtype = text_dtype

        # Frozen, and plain objects rather than submodules -- see encoders.py.
        self._conditioner: TextConditioner | None = None
        self.ema = EMA(self.denoiser, decay=ema_decay, warmup_steps=ema_warmup)

        print(self.denoiser.describe())

    # -- frozen components, created lazily so they land on the right device ---

    @property
    def conditioner(self) -> TextConditioner:
        if self._conditioner is None:
            self._conditioner = TextConditioner(device=self.device, dtype=self.text_dtype)
        return self._conditioner

    def get_vae(self):
        return get_shared_vae(dtype=torch.bfloat16, device=self.device)

    def setup(self, stage: str) -> None:
        self.ema.to(self.device)

    # -- training -------------------------------------------------------------

    def _step(self, batch: dict, dropout: float) -> torch.Tensor:
        latents = batch["latent"]
        cond = self.conditioner.encode_with_dropout(batch["caption"], dropout).to(latents.dtype)

        x_t, t, target = self.flow.training_pair(latents)
        prediction = self.denoiser(x_t, t, cond)
        return self.flow.loss(prediction, target)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._step(batch, self.caption_dropout)
        size = batch["latent"].shape[0]  # explicit: batches mix tensors and strings
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=size)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # No caption dropout: validation should measure conditional modelling.
        loss = self._step(batch, dropout=0.0)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=batch["latent"].shape[0])
        return loss

    # -- optimisation ---------------------------------------------------------

    def configure_optimizers(self):
        # Weight decay on matmul weights only. Applying it to norms, biases and
        # the modulation tables pulls them toward zero, which for adaLN
        # parameters means erasing the conditioning the model is trying to learn.
        decay, no_decay = [], []
        for name, param in self.denoiser.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith("scale_shift_table") or "pos_embed" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
            betas=(0.9, 0.95),  # lower beta2 than default: standard for transformers
            eps=1e-8,
        )

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, self.max_steps_ - self.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                "interval": "step",
            },
        }

    # -- checkpointing --------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
