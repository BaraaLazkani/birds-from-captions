"""The seam between the two models.

Everything downstream -- sampler, pipeline, GUI, evaluation -- is written
against ``Denoiser`` and never asks which model it is holding.  That is what
lets a 75M from-scratch transformer and a fine-tuned 860M Stable Diffusion UNet
be swapped by changing one line of config.

Two conventions make the seam work:

**Time is continuous and normalised.**  Callers always pass ``t`` in [0, 1],
where 0 is clean data and 1 is pure noise.  SD's UNet wants discrete timesteps
in 0..999 and the DiT wants a continuous value; converting is each model's own
business, not the sampler's.

**Models declare what they predict.**  The from-scratch model is trained with a
rectified-flow objective and predicts a velocity; SD 1.5 predicts the noise.
Rather than force one into the other's shape, each reports its
``prediction_type`` and the sampler dispatches once.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn

PredictionType = Literal["flow", "epsilon"]


class Denoiser(nn.Module, ABC):
    """A conditional denoiser operating in SD 1.5's latent space."""

    #: What ``forward`` returns -- a velocity field or the noise estimate.
    prediction_type: PredictionType = "flow"

    #: Channel count of the latents; 4 for SD 1.5's VAE.
    latent_channels: int = 4

    #: Latent grid size the model was trained at (32 => 256px, 64 => 512px).
    latent_size: int = 32

    @abstractmethod
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict at noisy latents ``x_t``.

        Args:
            x_t:  [B, C, H, W] noisy latents.
            t:    [B] continuous noise level in [0, 1]; 0 = clean, 1 = pure noise.
            cond: [B, L, D] text embedding sequence from :class:`TextConditioner`.

        Returns:
            [B, C, H, W] -- a velocity if ``prediction_type == "flow"``,
            otherwise the predicted noise.
        """

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        total, trainable = self.num_parameters, self.num_trainable_parameters
        return (
            f"{type(self).__name__}: {total / 1e6:.1f}M params "
            f"({trainable / 1e6:.1f}M trainable), predicts '{self.prediction_type}', "
            f"latents {self.latent_channels}x{self.latent_size}x{self.latent_size}"
        )
