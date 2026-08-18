"""Rectified flow matching -- the training objective for the from-scratch model.

Flow matching replaces DDPM's discrete noise schedule with a straight line
between data and noise:

    x_t = (1 - t) * x_0 + t * eps,        t in [0, 1]

Differentiating gives a velocity that does not depend on t at all:

    v = dx_t/dt = eps - x_0

so the model is trained on a plain MSE against that target, and sampling is
ordinary ODE integration from t=1 (noise) down to t=0 (data).  There is no
beta schedule, no alpha-bar table, no posterior variance -- which removes most
of the places a hand-written diffusion implementation usually goes subtly wrong.

This is the formulation behind SD3, Flux and SANA, and it is the reason the
sampler in this project is about thirty lines.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FlowMatching:
    """Noise schedule, training targets, and loss for rectified flow.

    Args:
        timestep_sampling: ``"logit_normal"`` or ``"uniform"``.
        logit_mean: Mean of the pre-sigmoid normal, when logit-normal.
        logit_std: Std of the pre-sigmoid normal, when logit-normal.
    """

    timestep_sampling: str = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0

    def sample_timesteps(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        """Draw ``t`` in [0, 1], one per batch element.

        Defaults to logit-normal rather than uniform, following SD3's finding.
        Uniform sampling spends much of its budget near t=0 and t=1, where the
        task is nearly trivial -- almost-clean latents, or almost-pure noise.
        Pushing a sigmoid through a normal concentrates samples in the middle of
        the trajectory, which is where the model actually has something to learn
        and where sample quality is decided.
        """
        if self.timestep_sampling == "uniform":
            return torch.rand(batch_size, device=device)
        if self.timestep_sampling == "logit_normal":
            normal = torch.randn(batch_size, device=device) * self.logit_std + self.logit_mean
            return torch.sigmoid(normal)
        raise ValueError(f"unknown timestep_sampling {self.timestep_sampling!r}")

    @staticmethod
    def _broadcast(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Reshape [B] -> [B, 1, 1, 1] so it broadcasts over image latents."""
        return t.view(-1, *([1] * (like.ndim - 1)))

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Interpolate along the straight path: ``x_t = (1-t) x_0 + t eps``."""
        t = self._broadcast(t, x0)
        return (1.0 - t) * x0 + t * noise

    @staticmethod
    def velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """The regression target, ``eps - x_0``.

        Constant along each path, which is exactly why these trajectories are
        straight and why few-step sampling works well.
        """
        return noise - x0

    def training_pair(
        self, x0: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build one training example set from clean latents.

        Returns ``(x_t, t, target)`` ready to feed the model and its loss.
        """
        noise = torch.empty_like(x0).normal_(generator=generator)
        t = self.sample_timesteps(x0.shape[0], x0.device)
        return self.add_noise(x0, noise, t), t, self.velocity_target(x0, noise)

    @staticmethod
    def loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Plain MSE, in fp32 for numerical stability under mixed precision."""
        return torch.nn.functional.mse_loss(prediction.float(), target.float())
