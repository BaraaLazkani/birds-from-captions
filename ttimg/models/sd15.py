"""Stable Diffusion 1.5's UNet, wrapped to satisfy the shared Denoiser interface.

Two jobs:

1. **Adapt.**  SD's UNet wants discrete timesteps in 0..999 and returns a noise
   estimate; the rest of this project speaks continuous t in [0, 1].  The
   conversion lives here so the sampler never has to know.

2. **Attach LoRA.**  Low-rank adapters on the attention projections, so the
   fine-tune trains ~1% of the parameters and produces a ~10 MB checkpoint
   instead of a 3.4 GB one.  On an 8 GB card this is the difference between a
   comfortable run and one that does not start.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .encoders import SD15_REPO
from .interface import Denoiser

# The attention projections inside every SD UNet block. Cross-attention (to_k,
# to_v on the text side) is where caption-to-image association lives, and
# self-attention carries spatial style -- adapting both is what lets a LoRA pick
# up CUB's subject matter and its photographic look together.
LORA_TARGET_MODULES = ("to_q", "to_k", "to_v", "to_out.0")


class SD15UNet(Denoiser):
    """Stable Diffusion 1.5's denoiser, optionally with LoRA adapters."""

    prediction_type = "epsilon"

    def __init__(
        self,
        repo: str = SD15_REPO,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
        latent_size: int = 64,  # 64 -> 512px, SD 1.5's native resolution
        trainable: bool = False,
    ) -> None:
        super().__init__()
        from diffusers import UNet2DConditionModel

        self.unet = UNet2DConditionModel.from_pretrained(repo, subfolder="unet", torch_dtype=dtype)
        self.latent_size = latent_size
        self.latent_channels = self.unet.config.in_channels
        self.has_lora = False

        if not trainable:
            self.unet.requires_grad_(False).eval()
        self.unet.to(device)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Continuous [0, 1] -> SD's discrete 0..999 grid.
        timesteps = (t.float() * 999.0).round().long().clamp(0, 999)
        return self.unet(x_t.to(self.unet.dtype), timesteps, encoder_hidden_states=cond.to(self.unet.dtype)).sample

    def add_lora(self, rank: int = 16, alpha: int | None = None, dropout: float = 0.0) -> None:
        """Attach LoRA adapters and freeze everything else."""
        from peft import LoraConfig

        config = LoraConfig(
            r=rank,
            lora_alpha=alpha if alpha is not None else rank,
            lora_dropout=dropout,
            target_modules=list(LORA_TARGET_MODULES),
            init_lora_weights="gaussian",
        )
        self.unet.add_adapter(config)
        self.has_lora = True

        # Adapters must be fp32 even when the frozen base is fp16: their updates
        # are small, and fp16 gradients at that magnitude round to zero.
        for param in self.unet.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

    def enable_gradient_checkpointing(self) -> None:
        self.unet.enable_gradient_checkpointing()

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        from peft import get_peft_model_state_dict

        return get_peft_model_state_dict(self.unet)

    def save_lora(self, path: str | Path) -> Path:
        """Write adapters as safetensors.

        Deliberately not ``torch.save``: that format unpickles arbitrary Python
        on load, and adapter files are exactly the kind of artefact that gets
        shared around. safetensors holds tensors and nothing executable.
        """
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v.to(torch.float16).cpu().contiguous() for k, v in self.lora_state_dict().items()}
        save_file(state, str(path))
        total = sum(v.numel() for v in state.values())
        print(f"[sd15] saved LoRA ({len(state)} tensors, {total / 1e6:.2f}M params) -> {path}")
        return path

    def load_lora(self, path: str | Path, rank: int = 16, alpha: int | None = None) -> None:
        """Attach adapters (if absent) and load weights from ``save_lora``."""
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        if not self.has_lora:
            self.add_lora(rank=rank, alpha=alpha)

        state = load_file(str(path))
        dtype = next(p for p in self.unet.parameters() if p.requires_grad).dtype
        set_peft_model_state_dict(self.unet, {k: v.to(dtype) for k, v in state.items()})
        print(f"[sd15] loaded LoRA from {path}")
