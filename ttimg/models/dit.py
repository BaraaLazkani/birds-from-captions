"""A PixArt-style diffusion transformer, written from scratch.

This is the model the project exists to build.  It operates on SD 1.5's latents
(4 x 32 x 32 for 256px images), patchifies them into 256 tokens, and runs a
stack of transformer blocks that alternate:

    self-attention   -- how do the patches relate to each other?
    cross-attention  -- what did the caption ask for?
    MLP              -- per-token processing

conditioned on the diffusion timestep through adaptive layer norm.  It is the
architecture behind PixArt-alpha, SD3, Flux and SANA, at a size that fits an
8 GB laptop GPU.

Three choices worth calling out, because each is load-bearing:

**Cross-attention, not pooled conditioning.**  The cheap way to condition a DiT
is to squash the caption into one vector and feed it through adaLN.  That would
be actively wrong for CUB: "this bird has a red belly and black wings" would
collapse into an averaged "bird", and the model could never place the red on the
belly rather than the wings.  Attending over all 77 token embeddings is what
makes per-attribute control possible.

**adaLN-single.**  Standard DiT gives every block its own 6-way modulation MLP,
costing 6*d^2 parameters per block -- at d=512, depth 12, that is 19M parameters
(a third of the model) spent on timestep conditioning alone.  PixArt's trick is
one shared modulation MLP plus a small learned per-block offset table, which
buys back those parameters for actual capacity.

**QK normalisation.**  RMSNorm on queries and keys before attention. Attention
logits in a from-scratch model trained in bf16 can grow until softmax saturates
and gradients vanish; this is the cheap, now-standard fix (SD3, Flux) and costs
essentially nothing.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .interface import Denoiser


def build_2d_sincos_pos_embed(dim: int, grid_size: int) -> torch.Tensor:
    """Fixed 2D sine-cosine position embeddings, [grid_size**2, dim].

    Fixed rather than learned: with ~12k images a learned table is one more
    thing to fit, and the standard DiT/PixArt recipe shows nothing is gained.
    """
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4 for 2D sincos, got {dim}")

    def axis_embed(positions: np.ndarray) -> np.ndarray:
        omega = 1.0 / 10000 ** (np.arange(dim // 4, dtype=np.float64) / (dim / 4.0))
        out = positions.reshape(-1)[:, None] * omega[None, :]
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    coords = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(coords, coords)  # (w, h) order, per the DiT reference
    embed = np.concatenate([axis_embed(grid[0]), axis_embed(grid[1])], axis=1)
    return torch.from_numpy(embed).float()


class TimestepEmbedder(nn.Module):
    """Continuous t in [0, 1] -> a vector the blocks can modulate with."""

    def __init__(self, hidden_dim: int, frequency_dim: int = 256, max_period: int = 10000) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Scale to ~[0, 1000] so the sinusoidal bands cover a useful range --
        # the frequencies below are the ones DDPM chose for discrete timesteps.
        t = t.float() * 1000.0
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)


class Attention(nn.Module):
    """Multi-head attention, self or cross, with QK-norm and flash SDPA."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        context_dim: int | None = None,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.is_cross = context_dim is not None
        kv_dim = context_dim if self.is_cross else dim

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_kv = nn.Linear(kv_dim, dim * 2, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)

        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        kv_source = context if self.is_cross else x

        q = self.to_q(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = (
            self.to_kv(kv_source)
            .view(batch, kv_source.shape[1], 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )

        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.to_out(out)


class DiTBlock(nn.Module):
    """Self-attention -> cross-attention -> MLP, with adaLN-single modulation."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        context_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads, qk_norm=qk_norm)

        # Cross-attention is deliberately *not* gated by the timestep: the
        # caption is equally relevant at every noise level, and PixArt shows a
        # plain residual works better here than a modulated one.
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        self.cross_attn = Attention(dim, num_heads, context_dim=context_dim, qk_norm=qk_norm)

        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))

        # The per-block half of adaLN-single: a learned offset added to the
        # shared, timestep-derived modulation. 6*dim params instead of 6*dim^2.
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, context: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t_mod.reshape(x.shape[0], 6, -1)
        ).chunk(6, dim=1)

        x = x + gate_sa * self.attn(self.norm1(x) * (1 + scale_sa) + shift_sa)
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + gate_mlp * self.mlp(self.norm3(x) * (1 + scale_mlp) + shift_mlp)
        return x


class FinalLayer(nn.Module):
    """Project tokens back to patch-shaped latent values."""

    def __init__(self, dim: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.scale_shift_table = nn.Parameter(torch.randn(2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        return self.linear(self.norm(x) * (1 + scale) + shift)


class ScratchDiT(Denoiser):
    """Text-conditional diffusion transformer trained from scratch.

    Args:
        latent_size: Latent grid side (32 for 256px images).
        patch_size: Latent cells per token. 2 gives 256 tokens from a 32x32 grid.
        dim: Model width.
        depth: Number of blocks.
        num_heads: Attention heads.
        context_dim: Text embedding width; 768 for CLIP ViT-L.
        mlp_ratio: MLP expansion.
        qk_norm: RMSNorm on queries/keys.
    """

    prediction_type = "flow"

    def __init__(
        self,
        latent_size: int = 32,
        latent_channels: int = 4,
        patch_size: int = 2,
        dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        context_dim: int = 768,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        if latent_size % patch_size != 0:
            raise ValueError(f"latent_size {latent_size} must be divisible by patch_size {patch_size}")

        self.latent_size = latent_size
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.dim = dim
        self.grid_size = latent_size // patch_size

        self.patch_embed = nn.Conv2d(latent_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.register_buffer(
            "pos_embed", build_2d_sincos_pos_embed(dim, self.grid_size)[None], persistent=False
        )

        self.t_embedder = TimestepEmbedder(dim)
        # The shared half of adaLN-single: one MLP for the whole network.
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

        # Captions arrive as CLIP embeddings; a LayerNorm here keeps their scale
        # from destabilising cross-attention early in training.
        self.context_norm = nn.LayerNorm(context_dim, eps=1e-6)

        self.blocks = nn.ModuleList(
            DiTBlock(dim, num_heads, context_dim, mlp_ratio, qk_norm) for _ in range(depth)
        )
        self.final_layer = FinalLayer(dim, patch_size, latent_channels)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(basic_init)

        # Patchify like a linear layer, not a conv.
        nn.init.xavier_uniform_(self.patch_embed.weight.view(self.patch_embed.weight.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero the output projection so the network starts as the identity
        # function on its residual stream: every block initially contributes
        # nothing, and learns how much to contribute. This is what "adaLN-Zero"
        # buys, and it is the difference between a stable start and a diverging one.
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, p*p*C] -> [B, C, H, W]."""
        batch, _, _ = x.shape
        p, g, c = self.patch_size, self.grid_size, self.latent_channels
        x = x.view(batch, g, g, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(batch, c, g * p, g * p)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x_t).flatten(2).transpose(1, 2)  # [B, T, D]
        x = x + self.pos_embed.to(x.dtype)

        t_emb = self.t_embedder(t)
        t_mod = self.t_block(t_emb)
        context = self.context_norm(cond)

        for block in self.blocks:
            x = block(x, context, t_mod)

        return self.unpatchify(self.final_layer(x, t_emb))
