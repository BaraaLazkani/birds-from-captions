"""Sampling, shared by both models.

One entry point, :func:`sample`, drives either denoiser.  It dispatches once on
``denoiser.prediction_type`` and then runs the matching integrator:

* ``"flow"``    -- Euler integration of the rectified-flow ODE (the scratch DiT).
* ``"epsilon"`` -- DDIM (Stable Diffusion's UNet).

Both are implemented here rather than delegated to ``diffusers`` schedulers, for
two reasons: the from-scratch model needs its own sampler anyway, and having the
DDIM update written out makes the comparison between the two objectives legible
instead of hidden behind a library.

Classifier-free guidance is applied identically in both paths, so a guidance
scale means the same thing on either model and the GUI can compare them fairly.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..models.interface import Denoiser

# Stable Diffusion 1.5's noise schedule.
SD_BETA_START = 0.00085
SD_BETA_END = 0.012
SD_TRAIN_TIMESTEPS = 1000


def sd_alphas_cumprod(device: torch.device | str = "cpu") -> torch.Tensor:
    """SD's "scaled_linear" schedule as a cumulative-alpha table, [1000]."""
    betas = (
        torch.linspace(SD_BETA_START**0.5, SD_BETA_END**0.5, SD_TRAIN_TIMESTEPS, dtype=torch.float64) ** 2
    )
    return torch.cumprod(1.0 - betas, dim=0).to(device=device, dtype=torch.float32)


def _guided(
    denoiser: Denoiser,
    x: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
    uncond: torch.Tensor | None,
    guidance_scale: float,
) -> torch.Tensor:
    """One denoiser call with classifier-free guidance folded in.

    The conditional and unconditional passes are concatenated into a single
    batch rather than run in sequence -- one kernel launch instead of two, which
    roughly halves sampling latency on a laptop GPU.
    """
    if uncond is None or guidance_scale == 1.0:
        return denoiser(x, t, cond)

    both = denoiser(
        torch.cat([x, x]),
        torch.cat([t, t]),
        torch.cat([uncond, cond]),
    )
    pred_uncond, pred_cond = both.chunk(2)
    return pred_uncond + guidance_scale * (pred_cond - pred_uncond)


@torch.no_grad()
def sample_flow(
    denoiser: Denoiser,
    cond: torch.Tensor,
    uncond: torch.Tensor | None,
    shape: tuple[int, ...],
    steps: int,
    guidance_scale: float,
    generator: torch.Generator | None,
    device: torch.device | str,
    dtype: torch.dtype,
    callback: Callable[[int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Euler integration of dx/dt = v(x, t) from t=1 (noise) to t=0 (data).

    Because rectified-flow paths are straight, plain Euler is a good integrator
    here -- this is why the model stays usable at 20-30 steps rather than needing
    the 50+ a curved DDPM trajectory would.
    """
    x = torch.randn(shape, generator=generator, device=device, dtype=torch.float32).to(dtype)
    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)

    for i in range(steps):
        t_now, t_next = timesteps[i], timesteps[i + 1]
        t_batch = t_now.expand(x.shape[0])
        velocity = _guided(denoiser, x, t_batch, cond, uncond, guidance_scale)
        # t_next < t_now, so this steps backwards along the path toward the data.
        x = x + (t_next - t_now) * velocity.to(x.dtype)
        if callback is not None:
            callback(i, x)

    return x


@torch.no_grad()
def sample_ddim(
    denoiser: Denoiser,
    cond: torch.Tensor,
    uncond: torch.Tensor | None,
    shape: tuple[int, ...],
    steps: int,
    guidance_scale: float,
    generator: torch.Generator | None,
    device: torch.device | str,
    dtype: torch.dtype,
    eta: float = 0.0,
    callback: Callable[[int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """DDIM sampling for epsilon-prediction models (Stable Diffusion).

    At each step the predicted noise is used to estimate the clean latent, which
    is then re-noised to the next, lower noise level.  ``eta=0`` makes this
    deterministic.
    """
    alphas = sd_alphas_cumprod(device)
    # Evenly spaced discrete timesteps, descending from 999.
    indices = torch.linspace(SD_TRAIN_TIMESTEPS - 1, 0, steps + 1, device=device).round().long()

    x = torch.randn(shape, generator=generator, device=device, dtype=torch.float32).to(dtype)

    for i in range(steps):
        index, index_prev = indices[i], indices[i + 1]
        alpha_t, alpha_prev = alphas[index], alphas[index_prev]

        # The adapter expects continuous t in [0, 1] and converts internally.
        t_batch = (index.float() / (SD_TRAIN_TIMESTEPS - 1)).expand(x.shape[0])
        eps = _guided(denoiser, x, t_batch, cond, uncond, guidance_scale).float()

        x32 = x.float()
        pred_x0 = (x32 - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()

        sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).clamp(min=0).sqrt()
        direction = (1 - alpha_prev - sigma**2).clamp(min=0).sqrt() * eps
        x32 = alpha_prev.sqrt() * pred_x0 + direction
        if eta > 0 and index_prev > 0:
            x32 = x32 + sigma * torch.randn(x32.shape, generator=generator, device=device)

        x = x32.to(dtype)
        if callback is not None:
            callback(i, x)

    return x


@torch.no_grad()
def sample(
    denoiser: Denoiser,
    cond: torch.Tensor,
    uncond: torch.Tensor | None = None,
    steps: int = 30,
    guidance_scale: float = 4.0,
    latent_size: int | None = None,
    seed: int | None = None,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
    callback: Callable[[int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Generate latents for a batch of text embeddings.

    Args:
        denoiser: Any :class:`Denoiser`; its ``prediction_type`` picks the integrator.
        cond: [B, L, D] text embeddings.
        uncond: [B, L, D] null embeddings for guidance; ``None`` disables CFG.
        steps: Integration steps.
        guidance_scale: CFG strength. 1.0 = no guidance.
        latent_size: Override the denoiser's native latent grid size.
        seed: Seed for the initial noise; makes runs comparable across models.

    Returns:
        [B, C, H, W] latents, ready for :func:`decode_latents`.
    """
    batch = cond.shape[0]
    size = latent_size or denoiser.latent_size
    shape = (batch, denoiser.latent_channels, size, size)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    kwargs = dict(
        denoiser=denoiser,
        cond=cond,
        uncond=uncond,
        shape=shape,
        steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        device=device,
        dtype=dtype,
        callback=callback,
    )

    if denoiser.prediction_type == "flow":
        return sample_flow(**kwargs)
    if denoiser.prediction_type == "epsilon":
        return sample_ddim(**kwargs)
    raise ValueError(f"unknown prediction_type {denoiser.prediction_type!r}")
