"""Load any of the project's three models by name.

    sd15       -- stock Stable Diffusion 1.5, the untouched baseline
    sd15_lora  -- SD 1.5 with the CUB LoRA adapters applied
    dit        -- the from-scratch diffusion transformer

The CLI, the evaluation code and the Gradio app all go through here, so there is
exactly one place that knows how each model is constructed, and all three come
back as a :class:`TextToImagePipeline` that is used identically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..pipeline import TextToImagePipeline
from .encoders import TextConditioner, get_shared_vae
from .interface import Denoiser

MODEL_KINDS = ("sd15", "sd15_lora", "dit")

# Human-readable labels for the GUI.
MODEL_LABELS = {
    "sd15": "Stable Diffusion 1.5 (stock)",
    "sd15_lora": "SD 1.5 + CUB LoRA (fine-tuned)",
    "dit": "Scratch DiT (trained from zero)",
}


def _safe_load(checkpoint: str | Path) -> dict:
    """Load a Lightning checkpoint without unpickling arbitrary objects.

    ``torch.load`` with ``weights_only=False`` executes whatever is pickled in
    the file, so a downloaded checkpoint could run arbitrary code.  Our
    checkpoints hold tensors and primitive containers, which ``weights_only=True``
    handles.  If one does contain a custom class, fail loudly with instructions
    rather than silently opening the unsafe path.
    """
    try:
        return torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - re-raised with guidance below
        raise RuntimeError(
            f"could not load {checkpoint} in safe mode ({exc}).\n"
            "It contains pickled objects beyond tensors and plain containers. "
            "Only if you trust its origin, re-run with "
            "TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 set explicitly."
        ) from exc


def load_dit(checkpoint: str | Path, device: str = "cuda", use_ema: bool = True) -> Denoiser:
    """Rebuild the scratch DiT from a Lightning checkpoint.

    Prefers the EMA weights when present: those are the ones the sample grids
    were drawn from and the ones that actually look good.
    """
    from .dit import ScratchDiT

    state = _safe_load(checkpoint)
    hparams = state.get("hyper_parameters", {})
    model_kwargs: dict[str, Any] = hparams.get("model") or {}

    denoiser = ScratchDiT(**model_kwargs)

    if use_ema and "ema" in state:
        shadow = state["ema"]["shadow"]
        missing = denoiser.load_state_dict(
            {k: v for k, v in shadow.items()}, strict=False
        )
        if missing.missing_keys:
            print(f"[registry] EMA missing {len(missing.missing_keys)} keys; falling back to raw weights")
            use_ema = False
        else:
            print(f"[registry] loaded EMA weights (step {state['ema']['step']})")

    if not (use_ema and "ema" in state):
        weights = {
            k.removeprefix("denoiser."): v
            for k, v in state["state_dict"].items()
            if k.startswith("denoiser.")
        }
        denoiser.load_state_dict(weights, strict=False)
        print("[registry] loaded raw (non-EMA) weights")

    return denoiser.to(device).eval()


def build_pipeline(
    kind: str,
    checkpoint: str | Path | None = None,
    lora_path: str | Path | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    conditioner: TextConditioner | None = None,
    lora_rank: int = 16,
) -> TextToImagePipeline:
    """Build a ready-to-use pipeline for one of :data:`MODEL_KINDS`.

    The text encoder and VAE are shared across every pipeline built here -- pass
    a ``conditioner`` in (or rely on the VAE singleton) so that switching models
    in the GUI does not reload 400 MB of frozen weights each time.
    """
    if kind not in MODEL_KINDS:
        raise ValueError(f"unknown model kind {kind!r}; expected one of {MODEL_KINDS}")

    conditioner = conditioner or TextConditioner(device=device, dtype=dtype)
    vae = get_shared_vae(dtype=dtype, device=device)

    if kind == "dit":
        if checkpoint is None:
            raise ValueError("kind='dit' requires a checkpoint path")
        denoiser = load_dit(checkpoint, device=device)
        denoiser = denoiser.to(dtype)
    else:
        from .sd15 import SD15UNet

        denoiser = SD15UNet(dtype=torch.float16, device=device, latent_size=64)
        if kind == "sd15_lora":
            if lora_path is None:
                raise ValueError("kind='sd15_lora' requires a lora_path")
            denoiser.load_lora(lora_path, rank=lora_rank)
            denoiser.unet.to(device)

    return TextToImagePipeline(denoiser, conditioner=conditioner, vae=vae, device=device, dtype=dtype)


def find_latest_checkpoint(run_dir: str | Path, pattern: str = "*.ckpt") -> Path | None:
    """Most recent checkpoint in a run directory, or None."""
    candidates = sorted(Path(run_dir).glob(pattern), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None
