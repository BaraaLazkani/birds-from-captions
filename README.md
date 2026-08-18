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

