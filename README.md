# Birds from captions

Two text-to-image models over one shared latent space, on the CUB-200-2011 bird
dataset:

1. **Stable Diffusion 1.5 fine-tuned with LoRA** — the pragmatic path.
2. **A diffusion transformer trained from scratch** — ~56M parameters, no
   pretrained denoiser weights, rectified-flow objective.

Both run behind one Gradio app that can put them side by side on the same prompt
and the same starting noise.

Built to run on a single **8 GB laptop GPU** (RTX 5050, Blackwell `sm_120`).

---

## The idea

The expensive, generic parts of a text-to-image system — the image autoencoder
and the text encoder — are *shared*, and only the denoiser differs:

```
caption → CLIP ViT-L/14 text encoder ─┐
                                      ├→ [ denoiser ] → latents → SD VAE → image
noise ───────────────────────────────┘        ↑
                                     ┌────────┴────────┐
                                SD 1.5 UNet        Scratch DiT
                                  (+ LoRA)        (flow matching)
```

Everything else is common: the precomputed latent cache, the text embeddings,
the classifier-free guidance loop, the decoder, the evaluation harness and the
GUI. The from-scratch model is a *second module behind one interface*
(`ttimg/models/interface.py`), not a second project.

This is also what makes the comparison honest. When the GUI renders both models
on one prompt, the noise, text encoder, sampler and decoder are identical — the
only thing that differs is the denoiser.

## Why these choices

| Decision | Reasoning |
|---|---|
| **SD 1.5**, not SDXL | 2 GB in fp16 vs 7 GB. Fits 8 GB at 512px with room to train. |
| **Precompute latents, encode text live** | Latents: expensive to compute, cheap to store (950 MB). Text: the reverse — caching 117,880 captions at 77×768 would be **14 GB**, while a CLIP text pass is ~2 ms. So the VAE never occupies VRAM during training. |
| **Rectified flow** for the scratch model | Straight paths between noise and data mean Euler integration is *exact* (verified in `tests/`), no beta schedules or posterior variances to get wrong, and good samples at 20–30 steps. |
| **Cross-attention**, not pooled conditioning | Pooling CLIP's 77 tokens into one vector destroys exactly the per-attribute detail that makes CUB interesting. "Red belly, black wings" must be able to place the red on the belly. |
| **adaLN-single** (PixArt) | Standard DiT spends `6·d²` params per block on timestep modulation — at d=512, depth 12, that is 19M params, a third of the model. One shared MLP plus small per-block offsets buys that back as capacity. |
| **Bounding-box crop** | CUB birds are often small in cluttered frames. Cropping so the bird fills ~75% of the frame (StackGAN convention) stops a small model wasting capacity on branches and sky. |
| **QK-norm, EMA, logit-normal timesteps** | Cheap, standard, and each worth more than they cost: bounded attention logits in bf16, cleaner samples, and training effort spent mid-trajectory where the task is hard. |

