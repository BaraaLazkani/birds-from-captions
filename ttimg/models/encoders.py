"""The frozen halves of the system: SD 1.5's VAE and CLIP text encoder.

Both denoisers -- the fine-tuned SD UNet and the from-scratch DiT -- operate in
*this* latent space and consume *these* text embeddings.  Sharing them is the
decision that makes the project affordable on 8 GB:

* one VAE and one text encoder on disk instead of two of each, and
* latents can be precomputed once and reused by both training runs.

Nothing in this module is ever trained.
"""

from __future__ import annotations

import torch

SD15_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# SD 1.5's VAE emits latents with roughly unit variance only after this scaling.
# Skipping it is a classic silent failure: training "works" but samples are mush.
VAE_SCALING_FACTOR = 0.18215
VAE_DOWNSCALE = 8  # pixels per latent cell
LATENT_CHANNELS = 4
TEXT_EMBED_DIM = 768  # CLIP ViT-L/14 hidden size
MAX_TEXT_TOKENS = 77


def load_vae(
    repo: str = SD15_REPO,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cuda",
):
    """Load the frozen autoencoder.

    Defaults to fp32 because the one place this runs at scale -- the one-off
    latent precompute -- is worth doing accurately, and SD 1.5's VAE is the
    component notorious for producing NaNs in fp16.  For interactive sampling
    pass ``torch.bfloat16``: it has fp32's exponent range, so it sidesteps that
    overflow entirely while halving memory.
    """
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", torch_dtype=dtype)
    vae.requires_grad_(False).eval()
    return vae.to(device)


@torch.no_grad()
def encode_images(vae, images: torch.Tensor, sample: bool = False) -> torch.Tensor:
    """Images in [-1, 1], shape [B,3,H,W] -> scaled latents [B,4,H/8,W/8].

    ``sample=False`` takes the posterior mean, which is what we want for a cache:
    deterministic, and the stochasticity the VAE would add is noise we would only
    have to learn to ignore.
    """
    posterior = vae.encode(images.to(vae.dtype)).latent_dist
    latents = posterior.sample() if sample else posterior.mean
    return latents * VAE_SCALING_FACTOR


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Scaled latents -> images in [0, 1], ready for saving or display."""
    images = vae.decode(latents.to(vae.dtype) / VAE_SCALING_FACTOR).sample
    return (images / 2 + 0.5).clamp(0, 1)


_VAE_CACHE: dict[tuple, object] = {}


def get_shared_vae(dtype: torch.dtype = torch.bfloat16, device: str | torch.device = "cuda"):
    """Process-wide VAE singleton.

    Sampling callbacks, evaluation and the GUI all need a decoder, and loading
    several copies of it would waste both disk reads and ~160 MB of VRAM each.
    Deliberately kept out of any ``LightningModule``: the VAE is frozen and
    training runs on cached latents, so it must not appear in checkpoints or
    parameter groups.
    """
    key = (str(dtype), str(device))
    if key not in _VAE_CACHE:
        _VAE_CACHE[key] = load_vae(dtype=dtype, device=device)
    return _VAE_CACHE[key]


class TextConditioner:
    """Frozen CLIP ViT-L/14 text encoder, shared by both denoisers.

    Intentionally *not* an ``nn.Module``.  It holds one internally, but it is
    never trained, and subclassing would register 123M frozen parameters into
    whatever ``LightningModule`` referenced it -- inflating every checkpoint and
    inviting them into optimiser parameter groups.

    Text embeddings are computed on the fly rather than cached.  Caching them
    would cost ~14 GB for CUB's 117,880 captions at 77x768 in fp16 -- more disk
    than this machine has free -- while a forward pass through a 123M-parameter
    encoder is a couple of milliseconds and disappears behind the denoiser step.
    Latents get the opposite treatment for the same reason, inverted: expensive
    to compute, cheap to store.
    """

    def __init__(
        self,
        repo: str = SD15_REPO,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        max_length: int = MAX_TEXT_TOKENS,
    ) -> None:
        from transformers import CLIPTextModel, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
        self.model = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=dtype)
        self.model.requires_grad_(False).eval()
        self.model.to(device)

        self.device_ = torch.device(device)
        self.dtype = dtype
        self.max_length = max_length
        self._null_cache: torch.Tensor | None = None

    @property
    def embed_dim(self) -> int:
        return self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, prompts: list[str]) -> torch.Tensor:
        """List of captions -> [B, 77, 768] embedding sequence."""
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        return self.model(tokens.input_ids.to(self.device_))[0]

    @torch.no_grad()
    def null_embedding(self, batch_size: int = 1) -> torch.Tensor:
        """The unconditional embedding used by classifier-free guidance.

        This is the empty string put through the same encoder, matching how
        Stable Diffusion defines its null condition -- so the scratch DiT and the
        SD UNet share one notion of "unconditional" and can be driven by the same
        guidance code.  Cached, since it never changes.
        """
        if self._null_cache is None:
            self._null_cache = self.encode([""])
        return self._null_cache.expand(batch_size, -1, -1)

    @torch.no_grad()
    def encode_with_dropout(self, prompts: list[str], dropout_prob: float = 0.1) -> torch.Tensor:
        """Encode, replacing a fraction of captions with the null embedding.

        This is how a model learns to be steerable: it must see the
        unconditional case often enough to model it, or classifier-free guidance
        at sample time extrapolates along a direction it never learned.
        """
        embeddings = self.encode(prompts)
        if dropout_prob <= 0:
            return embeddings

        drop = torch.rand(len(prompts), device=embeddings.device) < dropout_prob
        if drop.any():
            embeddings[drop] = self.null_embedding(1).to(embeddings.dtype)
        return embeddings

    def to(self, device: str | torch.device) -> TextConditioner:
        self.model.to(device)
        self.device_ = torch.device(device)
        self._null_cache = None  # cached on the old device; recompute on demand
        return self

    def __call__(self, prompts: list[str]) -> torch.Tensor:  # convenience
        return self.encode(prompts)
