"""Exponential moving average of model weights.

Not optional for diffusion.  The raw weights at any given step bounce around
enough that samples drawn from them are visibly noisier than samples drawn from
their running average -- EMA is routinely worth more visual quality than an
architecture change, and it costs one extra copy of the parameters.

For a 55M-parameter DiT that copy is ~220 MB in fp32, which fits comfortably
alongside training on an 8 GB card.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn


class EMA:
    """Tracks a decayed average of a model's parameters.

    Args:
        model: Model whose parameters to track.
        decay: Target decay. 0.9999 is standard for long diffusion runs.
        warmup_steps: Ramp the decay in from 0 over roughly this many steps.
            Without it the average stays anchored to the random initialisation
            for the first several thousand updates, which wastes early training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 1000) -> None:
        self.decay = decay
        self.warmup_steps = max(warmup_steps, 1)
        self.step = 0
        self.shadow = {
            name: param.detach().clone().float()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def current_decay(self) -> float:
        """Decay ramped in over warmup, so early steps track the model closely."""
        return min(self.decay, (1 + self.step) / (self.warmup_steps + self.step))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        decay = self.current_decay()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(decay).add_(param.detach().float(), alpha=1.0 - decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.dtype))

    @contextmanager
    def averaged(self, model: nn.Module):
        """Temporarily swap EMA weights in, e.g. to draw a progress grid."""
        backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if name in self.shadow
        }
        try:
            self.copy_to(model)
            yield model
        finally:
            for name, param in model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "warmup_steps": self.warmup_steps, "step": self.step, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.warmup_steps = state["warmup_steps"]
        self.step = state["step"]
        self.shadow = {k: v.float() for k, v in state["shadow"].items()}

    def to(self, device: str | torch.device) -> EMA:
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self
